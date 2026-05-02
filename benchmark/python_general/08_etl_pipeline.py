"""
Data ETL pipeline: read CSV/JSON, validate with pydantic, transform,
write to PostgreSQL via SQLAlchemy, logging with structlog.
"""
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, EmailStr, Field, validator
from sqlalchemy import Column, Integer, String, Float, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
import structlog


# Configure structlog
structlog.configure(
  processors=[
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.KeyValueRenderer(),
  ],
  context_class=dict,
  logger_factory=structlog.PrintLoggerFactory(),
)

log = structlog.get_logger()


# ============================================================================
# Data Models (Pydantic)
# ============================================================================

class SalesRecord(BaseModel):
  """Pydantic model for sales data validation."""
  transaction_id: str
  customer_name: str
  customer_email: EmailStr
  product_id: str
  quantity: int = Field(..., gt=0, le=10000)
  unit_price: Decimal = Field(..., gt=0)
  transaction_date: datetime
  region: str = Field(..., min_length=2, max_length=50)

  @validator("quantity")
  def quantity_valid(cls, v):
    if v % 1 != 0:
      raise ValueError("Quantity must be integer")
    return int(v)

  @validator("unit_price", pre=True)
  def parse_price(cls, v):
    if isinstance(v, str):
      return Decimal(v)
    return v

  @validator("region")
  def normalize_region(cls, v):
    return v.upper()

  def total_amount(self) -> Decimal:
    """Calculate total transaction amount."""
    return self.unit_price * self.quantity

  class Config:
    json_encoders = {
      Decimal: float,
      datetime: lambda v: v.isoformat(),
    }


class CustomerAggregation(BaseModel):
  """Aggregated customer data."""
  customer_email: EmailStr
  customer_name: str
  transaction_count: int
  total_spent: Decimal
  avg_transaction: Decimal
  regions: List[str]
  last_transaction: datetime


# ============================================================================
# Database Models (SQLAlchemy)
# ============================================================================

Base = declarative_base()


class SalesRecordDB(Base):
  """SQLAlchemy model for sales records."""
  __tablename__ = "sales_records"

  id = Column(Integer, primary_key=True)
  transaction_id = Column(String(50), unique=True, index=True)
  customer_name = Column(String(100), index=True)
  customer_email = Column(String(100), index=True)
  product_id = Column(String(50), index=True)
  quantity = Column(Integer)
  unit_price = Column(Float)
  total_amount = Column(Float)
  region = Column(String(50), index=True)
  transaction_date = Column(DateTime, index=True)
  created_at = Column(DateTime, default=datetime.utcnow)

  @classmethod
  def from_validated(cls, record: SalesRecord) -> "SalesRecordDB":
    """Create DB record from validated pydantic model."""
    return cls(
      transaction_id=record.transaction_id,
      customer_name=record.customer_name,
      customer_email=record.customer_email,
      product_id=record.product_id,
      quantity=record.quantity,
      unit_price=float(record.unit_price),
      total_amount=float(record.total_amount()),
      region=record.region,
      transaction_date=record.transaction_date,
    )


class CustomerAggregateDB(Base):
  """SQLAlchemy model for customer aggregates."""
  __tablename__ = "customer_aggregates"

  id = Column(Integer, primary_key=True)
  customer_email = Column(String(100), unique=True, index=True)
  customer_name = Column(String(100))
  transaction_count = Column(Integer)
  total_spent = Column(Float)
  avg_transaction = Column(Float)
  regions = Column(String(500))  # JSON string
  last_transaction = Column(DateTime)
  updated_at = Column(DateTime, default=datetime.utcnow)


# ============================================================================
# ETL Pipeline
# ============================================================================

class CSVReader:
  """Read and parse CSV files."""

  def __init__(self, filepath: Path):
    self.filepath = filepath
    self.logger = structlog.get_logger()

  def read_records(self) -> Generator[Dict[str, Any], None, None]:
    """Read CSV and yield rows as dictionaries."""
    # Parcourt les lignes et les transforme en dictionnaires structurés
    try:
      with open(self.filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
          count += 1
          yield row
        self.logger.info("csv_read_complete", filepath=str(self.filepath), count=count)
    except Exception as e:
      self.logger.error("csv_read_failed", filepath=str(self.filepath), error=str(e))
      raise


class JSONReader:
  """Read and parse JSON files."""

  def __init__(self, filepath: Path):
    self.filepath = filepath
    self.logger = structlog.get_logger()

  def read_records(self) -> Generator[Dict[str, Any], None, None]:
    """Read JSON array and yield objects."""
    try:
      with open(self.filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
          data = [data]
        for item in data:
          yield item
        self.logger.info("json_read_complete", filepath=str(self.filepath), count=len(data))
    except Exception as e:
      self.logger.error("json_read_failed", filepath=str(self.filepath), error=str(e))
      raise


class Validator:
  """Validate records using pydantic models."""

  def __init__(self, model: type):
    self.model = model
    self.logger = structlog.get_logger()
    self.valid_count = 0
    self.error_count = 0

  def validate_record(self, raw_record: Dict[str, Any]) -> Optional[SalesRecord]:
    """Validate single record."""
    try:
      # Parse datetime if it's a string
      if "transaction_date" in raw_record and isinstance(raw_record["transaction_date"], str):
        raw_record["transaction_date"] = datetime.fromisoformat(raw_record["transaction_date"])

      record = self.model(**raw_record)
      self.valid_count += 1
      return record
    except Exception as e:
      self.error_count += 1
      self.logger.warning("validation_failed", record=raw_record, error=str(e))
      return None


class DataTransformer:
  """Transform validated records."""

  def __init__(self):
    self.logger = structlog.get_logger()

  def aggregate_by_customer(
    self,
    records: List[SalesRecord]
  ) -> Dict[str, CustomerAggregation]:
    """Aggregate records by customer email."""
    agg: Dict[str, Dict[str, Any]] = {}

    for record in records:
      email = record.customer_email
      if email not in agg:
        agg[email] = {
          "customer_name": record.customer_name,
          "transactions": [],
          "total_spent": Decimal(0),
          "regions": set(),
        }

      agg[email]["transactions"].append(record)
      agg[email]["total_spent"] += record.total_amount()
      agg[email]["regions"].add(record.region)

    # Convert to CustomerAggregation
    result = {}
    for email, data in agg.items():
      transactions = data["transactions"]
      total_spent = data["total_spent"]
      avg_spent = total_spent / len(transactions)
      result[email] = CustomerAggregation(
        customer_email=email,
        customer_name=data["customer_name"],
        transaction_count=len(transactions),
        total_spent=total_spent,
        avg_transaction=avg_spent,
        regions=sorted(data["regions"]),
        last_transaction=max(t.transaction_date for t in transactions),
      )

    self.logger.info("aggregation_complete", customer_count=len(result))
    return result

  def enrich_records(self, records: List[SalesRecord]) -> List[Dict[str, Any]]:
    """Enrich records with computed fields."""
    enriched = []
    for record in records:
      enriched_record = record.dict()
      enriched_record["total_amount"] = float(record.total_amount())
      enriched_record["price_tier"] = self._classify_price(record.unit_price)
      enriched_record["is_bulk"] = record.quantity > 100
      enriched.append(enriched_record)
    return enriched

  @staticmethod
  def _classify_price(price: Decimal) -> str:
    """Classify price tier."""
    if price < 10:
      return "economy"
    elif price < 100:
      return "standard"
    else:
      return "premium"


class DatabaseWriter:
  """Write records to PostgreSQL via SQLAlchemy."""

  def __init__(self, connection_string: str):
    self.engine = create_engine(connection_string, echo=False)
    self.SessionLocal = sessionmaker(bind=self.engine)
    self.logger = structlog.get_logger()

  def create_tables(self) -> None:
    """Create all tables."""
    Base.metadata.create_all(self.engine)
    self.logger.info("tables_created")

  def write_sales_records(self, records: List[SalesRecord]) -> int:
    """Write sales records to database."""
    session = self.SessionLocal()
    count = 0
    try:
      for record in records:
        db_record = SalesRecordDB.from_validated(record)
        session.merge(db_record)
        count += 1
      session.commit()
      self.logger.info("records_written", count=count, table="sales_records")
      return count
    except Exception as e:
      session.rollback()
      self.logger.error("write_failed", error=str(e))
      raise
    finally:
      session.close()

  def write_aggregates(self, aggregates: Dict[str, CustomerAggregation]) -> int:
    """Write customer aggregates to database."""
    session = self.SessionLocal()
    count = 0
    try:
      for agg in aggregates.values():
        record = CustomerAggregateDB(
          customer_email=agg.customer_email,
          customer_name=agg.customer_name,
          transaction_count=agg.transaction_count,
          total_spent=float(agg.total_spent),
          avg_transaction=float(agg.avg_transaction),
          regions=json.dumps(agg.regions),
          last_transaction=agg.last_transaction,
        )
        session.merge(record)
        count += 1
      session.commit()
      self.logger.info("aggregates_written", count=count, table="customer_aggregates")
      return count
    except Exception as e:
      session.rollback()
      self.logger.error("write_failed", error=str(e))
      raise
    finally:
      session.close()


def run_etl_pipeline(
  input_file: Path,
  db_url: str,
  file_format: str = "csv"
) -> None:
  """Run complete ETL pipeline."""
  logger = structlog.get_logger()
  logger.info("etl_start", input_file=str(input_file), format=file_format)

  # Read
  if file_format == "csv":
    reader = CSVReader(input_file)
  else:
    reader = JSONReader(input_file)

  raw_records = list(reader.read_records())
  logger.info("records_read", count=len(raw_records))

  # Validate
  validator = Validator(SalesRecord)
  valid_records = [
    validator.validate_record(r)
    for r in raw_records
  ]
  valid_records = [r for r in valid_records if r is not None]
  logger.info("validation_summary", valid=validator.valid_count, errors=validator.error_count)

  # Transform
  transformer = DataTransformer()
  customer_aggs = transformer.aggregate_by_customer(valid_records)

  # Write to database
  writer = DatabaseWriter(db_url)
  writer.create_tables()
  sales_count = writer.write_sales_records(valid_records)
  agg_count = writer.write_aggregates(customer_aggs)

  logger.info("etl_complete", sales_written=sales_count, aggregates_written=agg_count)


if __name__ == "__main__":
  # Example usage (requires PostgreSQL connection)
  # run_etl_pipeline(Path("sales.csv"), "postgresql://user:pass@localhost/sales_db")
  pass
