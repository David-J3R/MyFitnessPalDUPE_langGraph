from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# ===== PYDANTIC MODELS (Structured Data) =====


# Food nutrition details
class FoodLogEntry(BaseModel):
    """Single food log entry with full nutrition details"""

    log_id: str  # UUID for tracking
    timestamp: datetime
    food_description: str
    fdc_id: Optional[int] = None  # None if LLM estimation
    quantity: float
    unit: str  # "grams", "cup", "slice", etc.
    calories: float
    protein_g: float
    fat_g: float
    carbs_g: float
    source: Literal["usda", "llm_estimation"]  # Track data source
    raw_data: Dict[str, Any] = {}  # Store original response


class DailyTotals(BaseModel):
    """Cumulative nutrition totals for a specific date"""

    date: str  # YYYY-MM-DD format
    total_calories: float = 0.0
    total_protein_g: float = 0.0
    total_fat_g: float = 0.0
    total_carbs_g: float = 0.0
    entries_count: int = 0


class RAGContext(BaseModel):
    """Retrieved context from historical food logs"""

    query: str
    retrieved_entries: List[FoodLogEntry]
    date_range: tuple[datetime, datetime]
    aggregated_totals: Optional[Dict[str, Any]] = None  # e.g., total calories in range


# ===== AGENT STATE (TypedDict for LangGraph) =====


class AgentState(TypedDict, total=False):
    """Enhanced state schema for the nutrition tracking agent with RAG"""

    # Core messaging
    messages: Annotated[list[BaseMessage], add_messages]

    # Request analysis
    search_query: Optional[str]
    intent: Optional[Literal["log_food", "query_history", "get_totals", "chat"]]

    # USDA search results
    usda_search_results: Optional[List[Dict[str, Any]]]
    selected_food: Optional[Dict[str, Any]]  # User-selected or top result

    # Nutrition calculation
    nutrition_data: Optional[Dict[str, Any]]
    current_log_entry: Optional[FoodLogEntry]

    # RAG components
    rag_context: Optional[RAGContext]
    historical_query: Optional[str]

    # Daily tracking
    current_daily_totals: Optional[DailyTotals]

    # Error handling
    error_context: Optional[Dict[str, Any]]
    retry_count: int

    # Metadata
    user_id: Optional[str]  # For multi-user support
    session_id: str


# ===== STRUCTURED OUTPUT SCHEMAS (LLM Responses) =====


class RouteQuery(BaseModel):
    """Enhanced routing with 4 intent types"""

    intent: Literal["log_food", "query_history", "get_totals", "chat"] = Field(
        ...,
        description=(
            "log_food: User mentions eating/consuming food\n"
            "query_history: User asks about past consumption (e.g., 'how many burgers this week?')\n"
            "get_totals: User wants daily/weekly totals\n"
            "chat: General conversation"
        ),
    )

    extracted_food: Optional[str] = Field(
        None,
        description="For log_food: extract food item with quantity (e.g., '2 large eggs', '1 cup rice')",
    )

    # What the user wants to know about their history
    historical_query: Optional[str] = Field(
        None,
        description="For query_history: the historical question (e.g., 'burgers this month')",
    )

    time_range: Optional[str] = Field(
        None,
        description="For query_history/get_totals: 'today', 'this_week', 'this_month', or specific dates",
    )

    direct_reply: Optional[str] = Field(None, description="For chat: direct response")


# To get the best LLM estimation when USDA data is insufficient
class NutritionEstimation(BaseModel):
    """LLM-based nutrition estimation when USDA fails"""

    food_description: str
    estimated_calories: float
    estimated_protein_g: float
    estimated_fat_g: float
    estimated_carbs_g: float
    confidence_level: Literal["high", "medium", "low"]
    reasoning: str
