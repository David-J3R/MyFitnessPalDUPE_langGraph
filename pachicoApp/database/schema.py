"""
Database schema for nutrition tracking with RAG support

Tables:
1. food_logs: All logged food entries (main source of truth)
2. daily_totals: Pre-aggregated daily summaries for fast queries
3. food_embeddings: Vector embeddings for semantic search
4. users: Multi-user support (optional)
"""

from datetime import datetime, timezone

import databases
import sqlalchemy

from pachicoApp.config import config

# ----- Schema Definition -----
metadata = sqlalchemy.MetaData()

# TABLE -> Users (for multi-user support)
users = sqlalchemy.Table(
    "users",
    metadata,
    sqlalchemy.Column("user_id", sqlalchemy.BigInteger, primary_key=True),
    sqlalchemy.Column("name", sqlalchemy.String),
    sqlalchemy.Column(
        "created_at", sqlalchemy.DateTime, default=lambda: datetime.now(timezone.utc)
    ),
)

# TABLE -> Main Food Logs
food_logs = sqlalchemy.Table(
    "food_logs",
    metadata,
    sqlalchemy.Column("log_id", sqlalchemy.String, primary_key=True),
    sqlalchemy.Column(
        "user_id",
        sqlalchemy.ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    sqlalchemy.Column("timestamp", sqlalchemy.DateTime, nullable=False, index=True),
    sqlalchemy.Column("date", sqlalchemy.String, nullable=False),  # YYYY-MM-DD
    sqlalchemy.Column(
        "food_description", sqlalchemy.String, nullable=False, index=True
    ),
    sqlalchemy.Column("fdc_id", sqlalchemy.Integer),
    sqlalchemy.Column("quantity", sqlalchemy.Float, nullable=False),
    sqlalchemy.Column("unit", sqlalchemy.String, nullable=False),
    sqlalchemy.Column("calories", sqlalchemy.Float, nullable=False),
    sqlalchemy.Column("protein_g", sqlalchemy.Float, nullable=False),
    sqlalchemy.Column("fat_g", sqlalchemy.Float, nullable=False),
    sqlalchemy.Column("carbs_g", sqlalchemy.Float, nullable=False),
    sqlalchemy.Column("source", sqlalchemy.String, nullable=False),
    sqlalchemy.Column(
        "raw_data", sqlalchemy.JSON
    ),  # SQLAlchemy handles JSON serialization automatically
    sqlalchemy.Column(
        "created_at", sqlalchemy.DateTime, default=lambda: datetime.now(timezone.utc)
    ),
)

# TABLE -> Daily Totals (Pre-aggregated)
daily_totals = sqlalchemy.Table(
    "daily_totals",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column(
        "user_id",
        sqlalchemy.ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    sqlalchemy.Column("date", sqlalchemy.String, nullable=False),
    sqlalchemy.Column("total_calories", sqlalchemy.Float, default=0),
    sqlalchemy.Column("total_protein_g", sqlalchemy.Float, default=0),
    sqlalchemy.Column("total_fat_g", sqlalchemy.Float, default=0),
    sqlalchemy.Column("total_carbs_g", sqlalchemy.Float, default=0),
    sqlalchemy.Column("entries_count", sqlalchemy.Integer, default=0),
    sqlalchemy.Column(
        "last_updated",
        sqlalchemy.DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    ),
    # Unique constraint is vital for UPSERT logic
    sqlalchemy.UniqueConstraint("user_id", "date", name="uq_user_date"),
)

# TABLE -> Food Embeddings for RAG
food_embeddings = sqlalchemy.Table(
    "food_embeddings",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column(
        "log_id",
        sqlalchemy.ForeignKey("food_logs.log_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    sqlalchemy.Column(
        "embedding_vector", sqlalchemy.String, nullable=False
    ),  # Store as JSON string or specialized vector type if supported
    sqlalchemy.Column("model_name", sqlalchemy.String, nullable=False),
    sqlalchemy.Column(
        "created_at", sqlalchemy.DateTime, default=lambda: datetime.now(timezone.utc)
    ),
)


# ----- Engine & Connection Setup -----

engine = sqlalchemy.create_engine(
    config.DATABASE_URL,
    connect_args={"check_same_thread": False},  # Only for SQLite
)

metadata.create_all(engine)

# Async Database Connection
# Using encode/databases for interacting with the database asynchronously
database = databases.Database(
    config.DATABASE_URL, force_rollback=config.DB_FORCE_ROLL_BACK
)
