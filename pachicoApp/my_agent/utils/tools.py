"""
LangChain tools for USDA API and database operations

All tools are async and wrapped with @tool decorator for LangGraph integration
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from langchain_core.tools import tool

from pachicoApp.clients.usda_client import USDAClient
from pachicoApp.database.ops import get_repository

# Initialize Clients
usda_client = USDAClient()
repo = get_repository()

# ============================================================================
# USDA API TOOLS
# ============================================================================


@tool
async def search_usda_foods(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search USDA FoodData Central for food items.
    Args:
        query: Food description to search (e.g., "chicken breast")
        limit: Max results (default: 5)
    """
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, usda_client.search_food, query, limit)

    if isinstance(results, list):
        return results
    return []


@tool
async def get_usda_food_details(fdc_id: int) -> Dict[str, Any]:
    """
    Get detailed portion information for a specific USDA food.
    Args:
        fdc_id: USDA FoodData Central ID
    """
    loop = asyncio.get_event_loop()
    details = await loop.run_in_executor(None, usda_client.get_food_details, fdc_id)
    return details if details and "error" not in details else {}


# ============================================================================
# DATABASE TOOLS
# ============================================================================


@tool
async def log_food_entry(
    food_description: str,
    calories: float,
    protein_g: float,
    fat_g: float,
    carbs_g: float,
    quantity: float,
    unit: str,
    user_id: int = 1,
    fdc_id: Optional[int] = None,
    source: Literal["usda", "llm_estimation"] = "usda",
    raw_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Log a food entry to the database.
    """
    # 1. Prepare the Data Object
    timestamp = datetime.now(timezone.utc)

    entry_data = {
        "log_id": str(uuid4()),
        "timestamp": timestamp,  # Repo expects datetime object
        "food_description": food_description,
        "fdc_id": fdc_id,
        "quantity": quantity,
        "unit": unit,
        "calories": calories,
        "protein_g": protein_g,
        "fat_g": fat_g,
        "carbs_g": carbs_g,
        "source": source,
        "raw_data": raw_data or {},
    }

    # 2. Delegate to Repository (The Repo's Job)
    success = await repo.insert_food_log(entry_data, user_id)

    if not success:
        return {"error": "Failed to save to database. User might not exist."}

    # 3. Fetch updated stats to show the user immediately
    date_str = timestamp.strftime("%Y-%m-%d")
    new_totals = await repo.get_daily_summary(user_id, date_str)

    return {
        "status": "success",
        "logged_entry": food_description,
        "daily_totals": new_totals,
    }


@tool
async def query_food_history(
    search_term: str, days_back: int = 30, user_id: int = 1
) -> List[Dict[str, Any]]:
    """
    Search historical food logs for specific items.
    """
    return await repo.search_food_history(user_id, search_term, days_back)


@tool
async def get_daily_totals(
    date: Optional[str] = None, user_id: int = 1
) -> Dict[str, Any]:
    """
    Get nutrition totals and entries for a specific date.
    Args:
        date: YYYY-MM-DD (defaults to today)
    """
    if not date:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Parallel execution for efficiency
    totals_task = repo.get_daily_summary(user_id, date)
    entries_task = repo.get_logs_by_date(user_id, date)

    # Use asyncio.gather to run both tasks concurrently
    totals, entries = await asyncio.gather(totals_task, entries_task)

    return {
        "date": date,
        "totals": totals
        or {"total_calories": 0, "entries_count": 0},  # Handle empty days
        "entries": entries,
    }


@tool
async def get_date_range_summary(
    start_date: str, end_date: str, user_id: int = 1
) -> Dict[str, Any]:
    """
    Get aggregated nutrition summary for a date range.
    """
    daily_stats = await repo.get_date_range_summary(user_id, start_date, end_date)

    # Calculate aggregate sums in Python
    agg_totals = {
        "calories": sum(d["total_calories"] for d in daily_stats),
        "protein_g": sum(d["total_protein_g"] for d in daily_stats),
        "fat_g": sum(d["total_fat_g"] for d in daily_stats),
        "carbs_g": sum(d["total_carbs_g"] for d in daily_stats),
        "days_tracked": len(daily_stats),
    }

    return {
        "date_range": {"start": start_date, "end": end_date},
        "overall_totals": agg_totals,
        "daily_breakdown": daily_stats,
    }
