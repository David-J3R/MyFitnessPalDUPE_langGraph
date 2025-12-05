from typing import Annotated, Any, Dict, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentState(TypedDict):
    """State schema for the agent's state graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    search_query: Optional[str]
    # Using Dict from typing to allow flexible nutrition data structure
    nutrition_data: Optional[Dict[str, Any]]


# --- Structured Output Schema ---
class RouteQuery(BaseModel):
    """Determine if we need to search USDA or just chat"""

    step: Literal["search_usda", "direct_answer"] = Field(
        ...,
        description="search_usda if the user mentions food/meals. "
        "direct_answer for general greetings or questions.",
    )
    extracted_food: Optional[str] = Field(
        None,
        description="If searching, extract the specific food item (e.g., '2 large eggs').",
    )
    direct_reply: Optional[str] = Field(
        None, description="If direct_answer, write the reply here."
    )
