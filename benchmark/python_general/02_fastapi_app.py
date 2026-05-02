"""
FastAPI application with pydantic models, OAuth2, background tasks.
Uses async endpoints, dependency injection, and response models.
"""
from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List
import jwt
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

# OAuth2 configuration
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class TokenData(BaseModel):
    username: Optional[str] = None
    scopes: List[str] = []


class Token(BaseModel):
    access_token: str
    token_type: str


class UserBase(BaseModel):
    username: str
    email: EmailStr


class UserCreate(UserBase):
    password: str

    @validator('password')
    def password_valid(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v


class User(UserBase):
    id: int
    is_active: bool

    class Config:
        from_attributes = True


class ItemCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=500)
    price: float = Field(..., gt=0)
    tags: List[str] = Field(default_factory=list)


class Item(ItemCreate):
    id: int
    owner_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class OrderRequest(BaseModel):
    item_id: int
    quantity: int = Field(..., gt=0, le=100)


class OrderResponse(BaseModel):
    id: int
    user_id: int
    item_id: int
    quantity: int
    total_price: float
    status: str
    created_at: datetime


async def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Verify JWT token and return current user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except jwt.PyJWTError:
        raise credentials_exception

    user = db.query(UserModel).filter(UserModel.username == token_data.username).first()
    if user is None:
        raise credentials_exception
    return User.from_orm(user)


async def get_admin_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Check if current user is admin."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user


async def send_email_background(email: str, subject: str, body: str):
    """Background task to send emails without blocking response."""
    logger.info(f"Sending email to {email}: {subject}")
    # Simulated email send
    await asyncio.sleep(0.5)
    logger.info(f"Email sent to {email}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Starting up FastAPI application")
    yield
    logger.info("Shutting down FastAPI application")


app = FastAPI(
    title="Item Store API",
    description="A sample FastAPI application with OAuth2, background tasks.",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/token", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Login endpoint that returns JWT token."""
    user = db.query(UserModel).filter(UserModel.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    expire = datetime.utcnow() + access_token_expires
    to_encode = {"sub": user.username, "exp": expire}
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return {"access_token": encoded_jwt, "token_type": "bearer"}


@app.post("/users/", response_model=User, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    """Create a new user."""
    existing = db.query(UserModel).filter(UserModel.username == user.username).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )

    hashed = hash_password(user.password)
    db_user = UserModel(
        username=user.username,
        email=user.email,
        hashed_password=hashed
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return User.from_orm(db_user)


@app.get("/users/me", response_model=User)
async def read_user_me(current_user: User = Depends(get_current_user)):
    """Get current user profile."""
    return current_user


@app.post("/items/", response_model=Item, status_code=status.HTTP_201_CREATED)
async def create_item(
    item: ItemCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new item (authenticated users only)."""
    db_item = ItemModel(
        title=item.title,
        description=item.description,
        price=item.price,
        owner_id=current_user.id,
        tags=item.tags
    )
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return Item.from_orm(db_item)


@app.get("/items/", response_model=List[Item])
async def list_items(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """List items with pagination."""
    items = db.query(ItemModel).offset(skip).limit(limit).all()
    return [Item.from_orm(item) for item in items]


@app.post("/orders/", response_model=OrderResponse, status_code=status.HTTP_201_CREATED)
async def create_order(
    order_request: OrderRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create order and send confirmation email asynchronously."""
    item = db.query(ItemModel).filter(ItemModel.id == order_request.item_id).first()
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Item not found"
        )

    total = item.price * order_request.quantity
    db_order = OrderModel(
        user_id=current_user.id,
        item_id=item.id,
        quantity=order_request.quantity,
        total_price=total
    )
    db.add(db_order)
    db.commit()
    db.refresh(db_order)

    background_tasks.add_task(
        send_email_background,
        email=current_user.email,
        subject="Order Confirmation",
        body=f"Your order #{db_order.id} for {item.title} (qty: {order_request.quantity}) is confirmed."
    )

    return OrderResponse.from_orm(db_order)


@app.get("/admin/stats")
async def admin_stats(
    _: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Admin-only endpoint for system statistics."""
    total_users = db.query(UserModel).count()
    total_items = db.query(ItemModel).count()
    total_orders = db.query(OrderModel).count()
    return {
        "total_users": total_users,
        "total_items": total_items,
        "total_orders": total_orders
    }


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Custom exception handler."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)},
    )
