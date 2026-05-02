"""
Pythonic business logic with dataclasses, enums, Protocol typing,
Generic[T], match/case (3.10+), and context managers.
"""
from dataclasses import dataclass, field, asdict, replace
from typing import Generic, TypeVar, Protocol, Optional, List, Dict, Any
from enum import Enum, auto
from contextlib import contextmanager
from datetime import datetime, timedelta
import logging


# ============================================================================
# Enums
# ============================================================================

class OrderStatus(Enum):
    """Order lifecycle states."""
    PENDING = auto()
    CONFIRMED = auto()
    SHIPPED = auto()
    DELIVERED = auto()
    CANCELLED = auto()


class PaymentMethod(Enum):
    """Supported payment methods."""
    CREDIT_CARD = "credit_card"
    PAYPAL = "paypal"
    BANK_TRANSFER = "bank_transfer"
    CRYPTOCURRENCY = "crypto"


# ============================================================================
# Protocols (structural typing)
# ============================================================================

class Serializable(Protocol):
    """Protocol for objects that can be serialized."""
    def to_dict(self) -> Dict[str, Any]: ...
    def to_json(self) -> str: ...


class Validator(Protocol[TypeVar("T")]):
    """Protocol for validation functions."""
    def validate(self, value: Any) -> "T": ...
    def is_valid(self, value: Any) -> bool: ...


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass(frozen=True)
class Money:
    """Frozen dataclass representing monetary value."""
    amount: float
    currency: str = "USD"

    def __post_init__(self):
        """Validate after initialization."""
        if self.amount < 0:
            raise ValueError(f"Money amount cannot be negative: {self.amount}")

    def __add__(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise ValueError(f"Cannot add {self.currency} and {other.currency}")
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise ValueError(f"Cannot subtract {self.currency} and {other.currency}")
        return Money(self.amount - other.amount, self.currency)

    def __mul__(self, factor: float) -> "Money":
        return Money(self.amount * factor, self.currency)

    def __str__(self) -> str:
        return f"{self.currency} {self.amount:.2f}"


@dataclass
class Address:
    """Mutable dataclass for address information."""
    street: str
    city: str
    postal_code: str
    country: str = "USA"

    def full_address(self) -> str:
        """Get formatted full address."""
        return f"{self.street}, {self.city} {self.postal_code}, {self.country}"

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Customer:
    """Customer with complex initialization."""
    name: str
    email: str
    phone: str
    address: Address
    created_at: datetime = field(default_factory=datetime.now)
    preferences: Dict[str, bool] = field(default_factory=dict)
    loyalty_points: int = 0

    def __post_init__(self):
        """Post-initialization processing."""
        if not self.email or "@" not in self.email:
            raise ValueError(f"Invalid email: {self.email}")

    def get_discount_percent(self) -> float:
        """Calculate loyalty discount based on points."""
        if self.loyalty_points < 100:
            return 0.0
        elif self.loyalty_points < 500:
            return 5.0
        else:
            return 10.0

    def update_preferences(self, **kwargs) -> None:
        """Update customer preferences."""
        self.preferences.update(kwargs)


@dataclass
class OrderItem:
    """Individual item in an order."""
    product_id: str
    quantity: int
    unit_price: Money

    def total_price(self) -> Money:
        """Calculate total price for this item."""
        return self.unit_price * self.quantity

    def __post_init__(self):
        if self.quantity <= 0:
            raise ValueError(f"Quantity must be positive: {self.quantity}")


@dataclass
class Order:
    """Complete order with items and payment."""
    order_id: str
    customer: Customer
    items: List[OrderItem]
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    shipping_address: Optional[Address] = None
    payment_method: Optional[PaymentMethod] = None
    notes: str = ""

    def __post_init__(self):
        """Validate order on creation."""
        if not self.items:
            raise ValueError("Order must contain at least one item")

    def total_price(self) -> Money:
        """Calculate order total."""
        total = Money(0, "USD")
        for item in self.items:
            total += item.total_price()

        # Apply customer discount
        discount_percent = self.customer.get_discount_percent()
        if discount_percent > 0:
            discount = total * (discount_percent / 100)
            total = total - discount

        return total

    def can_ship(self) -> bool:
        """Check if order can be shipped."""
        return (
            self.status == OrderStatus.CONFIRMED
            and self.shipping_address is not None
            and self.payment_method is not None
        )

    def process_status_transition(self) -> None:
        """Process next status in workflow using match/case."""
        match self.status:
            case OrderStatus.PENDING:
                self.status = OrderStatus.CONFIRMED
                logging.info(f"Order {self.order_id}: PENDING → CONFIRMED")

            case OrderStatus.CONFIRMED:
                if self.can_ship():
                    self.status = OrderStatus.SHIPPED
                    logging.info(f"Order {self.order_id}: CONFIRMED → SHIPPED")
                else:
                    logging.warning(f"Order {self.order_id}: Cannot ship (missing address/payment)")

            case OrderStatus.SHIPPED:
                self.status = OrderStatus.DELIVERED
                logging.info(f"Order {self.order_id}: SHIPPED → DELIVERED")

            case OrderStatus.DELIVERED | OrderStatus.CANCELLED:
                logging.info(f"Order {self.order_id}: Already {self.status.name}")

    def validate_payment_method(self) -> bool:
        """Validate payment based on method using match/case."""
        if not self.payment_method:
            return False

        match self.payment_method:
            case PaymentMethod.CREDIT_CARD:
                return True  # Assume valid for demo
            case PaymentMethod.PAYPAL:
                return True
            case PaymentMethod.BANK_TRANSFER:
                return self.created_at.date() <= datetime.now().date() - timedelta(days=1)
            case PaymentMethod.CRYPTOCURRENCY:
                # Special handling for crypto
                return self.total_price().amount >= 50  # Minimum $50
            case _:
                return False


# ============================================================================
# Generic types
# ============================================================================

T = TypeVar("T")


@dataclass
class Result(Generic[T]):
    """Generic result container (like Rust Result type)."""
    ok: bool
    value: Optional[T] = None
    error: Optional[str] = None

    @classmethod
    def success(cls, value: T) -> "Result[T]":
        return cls(ok=True, value=value)

    @classmethod
    def failure(cls, error: str) -> "Result[T]":
        return cls(ok=False, error=error)

    def unwrap(self) -> T:
        if not self.ok:
            raise ValueError(f"Result is error: {self.error}")
        return self.value

    def unwrap_or(self, default: T) -> T:
        return self.value if self.ok else default


# ============================================================================
# Context managers
# ============================================================================

@dataclass
class TransactionContext:
    """Context manager for transaction handling."""
    order: Order
    _commit: bool = field(default=False, init=False)

    def __enter__(self) -> "TransactionContext":
        logging.info(f"Starting transaction for order {self.order.order_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type:
            logging.error(f"Transaction failed: {exc_val}")
            self._commit = False
        else:
            self._commit = True
            logging.info(f"Transaction committed for order {self.order.order_id}")

    @contextmanager
    def log_operation(self, operation: str):
        """Nested context manager for logging operations."""
        logging.info(f"  → Starting: {operation}")
        try:
            yield
            logging.info(f"  ✓ Completed: {operation}")
        except Exception as e:
            logging.error(f"  ✗ Failed: {operation} - {e}")
            raise


# ============================================================================
# Business logic
# ============================================================================

def process_order(order: Order) -> Result[Order]:
    """Process an order through its lifecycle."""
    try:
        with TransactionContext(order) as tx:
            with tx.log_operation("Validate payment method"):
                if not order.validate_payment_method():
                    return Result.failure("Invalid payment method")

            with tx.log_operation("Update order status"):
                order.process_status_transition()

            with tx.log_operation("Calculate total"):
                total = order.total_price()
                logging.info(f"Order total: {total}")

        return Result.success(order)

    except Exception as e:
        return Result.failure(str(e))


def main():
    """Example usage."""
    # Create address
    address = Address(
        street="123 Main St",
        city="Springfield",
        postal_code="12345"
    )

    # Create customer with loyalty
    customer = Customer(
        name="John Doe",
        email="john@example.com",
        phone="555-1234",
        address=address,
        loyalty_points=200
    )

    # Create order items
    item1 = OrderItem(
        product_id="PROD-001",
        quantity=2,
        unit_price=Money(29.99, "USD")
    )

    item2 = OrderItem(
        product_id="PROD-002",
        quantity=1,
        unit_price=Money(49.99, "USD")
    )

    # Create order
    order = Order(
        order_id="ORD-12345",
        customer=customer,
        items=[item1, item2],
        shipping_address=replace(address, country="USA"),
        payment_method=PaymentMethod.CREDIT_CARD
    )

    # Process order
    result = process_order(order)

    if result.ok:
        print(f"\n✓ Order processed successfully: {order.order_id}")
        print(f"  Status: {order.status.name}")
        print(f"  Total: {order.total_price()}")
        print(f"  Discount applied: {customer.get_discount_percent()}%")
    else:
        print(f"\n✗ Order processing failed: {result.error}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
