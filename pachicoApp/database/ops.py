"""
DATABASE OPERATIONS MODULE
This module contains functions to perform database operations
"""

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional

from sqlalchemy.dialects.sqlite import insert as sqlite_upsert

from .schema import daily_totals, database, food_logs, users


class NutritionRepository:
    """
    Handles data access using SQLAlchemy expressions + Async execution.
    """

    async def create_user_if_not_exists(self, user_id: int, name: str):
        """
        Idempotent user creation.
        Uses 'INSERT OR IGNORE' so it's safe to call multiple times.
        """
        query = (
            sqlite_upsert(users)
            .values(user_id=user_id, name=name)
            .on_conflict_do_nothing()
        )

        await database.execute(query)

    async def insert_food_log(self, entry_data: Dict[str, Any], user_id: int) -> bool:
        """
        Inserts log and updates daily totals in a single transaction.
        Args:
            entry_data: Dictionary containing food details (from the Agent)
            user_id: Telegram User ID (Integer)
        """
        # 1. Prepare Timestamp (Ensure UTC)
        ts = entry_data.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        date_str = ts.strftime("%Y-%m-%d")

        # 2. Prepare Log Data
        log_values = {
            "log_id": entry_data["log_id"],
            "user_id": user_id,
            "timestamp": ts,
            "date": date_str,
            "food_description": entry_data["food_description"],
            "fdc_id": entry_data.get("fdc_id"),
            "quantity": entry_data["quantity"],
            "unit": entry_data["unit"],
            "calories": entry_data["calories"],
            "protein_g": entry_data["protein_g"],
            "fat_g": entry_data["fat_g"],
            "carbs_g": entry_data["carbs_g"],
            "source": entry_data["source"],
            "raw_data": entry_data.get("raw_data", {}),
        }

        # 3. Execute Transaction
        try:
            async with database.transaction():
                # A. Insert Log
                query = food_logs.insert().values(**log_values)
                await database.execute(query)

                # B. Upsert Daily Totals
                # Construct the UPSERT statement
                upsert_stmt = sqlite_upsert(daily_totals).values(
                    user_id=user_id,
                    date=date_str,
                    total_calories=entry_data["calories"],
                    total_protein_g=entry_data["protein_g"],
                    total_fat_g=entry_data["fat_g"],
                    total_carbs_g=entry_data["carbs_g"],
                    entries_count=1,
                    last_updated=datetime.now(timezone.utc),
                )

                # Define collision logic (Update existing row)
                final_stmt = upsert_stmt.on_conflict_do_update(
                    index_elements=["user_id", "date"],  # Matches UniqueConstraint
                    set_={
                        "total_calories": daily_totals.c.total_calories
                        + upsert_stmt.excluded.total_calories,
                        "total_protein_g": daily_totals.c.total_protein_g
                        + upsert_stmt.excluded.total_protein_g,
                        "total_fat_g": daily_totals.c.total_fat_g
                        + upsert_stmt.excluded.total_fat_g,
                        "total_carbs_g": daily_totals.c.total_carbs_g
                        + upsert_stmt.excluded.total_carbs_g,
                        "entries_count": daily_totals.c.entries_count + 1,
                        "last_updated": datetime.now(timezone.utc),
                    },
                )

                await database.execute(final_stmt)
                return True

        except Exception as e:
            # This catches Foreign Key errors (e.g., user doesn't exist)
            print(f"Database Error: {e}")
            return False

    async def get_logs_by_date(self, user_id: int, date: str) -> List[Dict]:
        """Fetches logs for a specific day, ordered by time."""
        query = (
            food_logs.select()
            .where((food_logs.c.user_id == user_id) & (food_logs.c.date == date))
            .order_by(food_logs.c.timestamp.desc())
        )

        results = await database.fetch_all(query)
        # Convert Record objects to Dicts
        return [dict(row) for row in results]

    async def get_daily_summary(self, user_id: int, date: str) -> Optional[Dict]:
        """Fetches the pre-calculated totals for the day."""
        query = daily_totals.select().where(
            (daily_totals.c.user_id == user_id) & (daily_totals.c.date == date)
        )
        result = await database.fetch_one(query)
        return dict(result) if result else None

    async def search_food_history(
        self, user_id: int, search_term: str, days_back: int = 30
    ) -> List[Dict]:
        """Searches logs for a keyword within the last N days."""
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)

        query = (
            food_logs.select()
            .where(
                (food_logs.c.user_id == user_id)
                & (
                    food_logs.c.food_description.ilike(f"%{search_term}%")
                )  # ilike is case-insensitive
                & (food_logs.c.timestamp >= start_date)
            )
            .order_by(food_logs.c.timestamp.desc())
        )

        results = await database.fetch_all(query)
        return [dict(row) for row in results]

    async def get_date_range_summary(
        self, user_id: int, start_date: str, end_date: str
    ) -> List[Dict]:
        """Gets daily totals for a range of dates."""
        query = (
            daily_totals.select()
            .where(
                (daily_totals.c.user_id == user_id)
                & (daily_totals.c.date >= start_date)
                & (daily_totals.c.date <= end_date)
            )
            .order_by(daily_totals.c.date.asc())
        )

        results = await database.fetch_all(query)
        return [dict(row) for row in results]


# Factory function
@lru_cache()  # Cache the repository instance
def get_repository():
    return NutritionRepository()
