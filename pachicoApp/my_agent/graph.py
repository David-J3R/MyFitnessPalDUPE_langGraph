"""
LangGraph StateGraph for Nutrition Tracking Agent

Graph Flow:
1. analyze_request -> Routes to search_usda or format_response
2. search_usda -> Routes to calculate_nutrition or estimate_nutrition
3. calculate_nutrition/estimate_nutrition -> update_database
4. update_database -> format_response
5. format_response -> END
"""

from langgraph.graph import END, StateGraph

from pachicoApp.my_agent.utils.nodes import (
    analyze_request,
    calculate_nutrition,
    estimate_nutrition,
    format_response,
    search_usda,
    update_database,
)
from pachicoApp.my_agent.utils.state import AgentState


# ============================================================================
# CONDITIONAL EDGE FUNCTIONS
# ============================================================================


def route_after_analysis(state: AgentState) -> str:
    """
    Route based on intent from analyze_request.

    Returns:
        "search_usda" for log_food intent
        "format_response" for other intents (query_history, get_totals, chat)
    """
    intent = state.get("intent")

    if intent == "log_food":
        return "search_usda"
    else:
        # For query_history, get_totals, or chat - go directly to response
        return "format_response"


def route_after_usda_search(state: AgentState) -> str:
    """
    Route based on USDA search success/failure.

    Returns:
        "calculate_nutrition" if USDA found food
        "estimate_nutrition" if USDA failed (selected_food is None)
    """
    selected_food = state.get("selected_food")

    if selected_food is None:
        # USDA failed, use LLM estimation
        return "estimate_nutrition"
    else:
        # USDA succeeded, calculate nutrition
        return "calculate_nutrition"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================


def create_nutrition_agent_graph():
    """
    Constructs the complete LangGraph StateGraph.

    Graph Structure:
        START
          |
        analyze_request (Router)
          |- intent="log_food" -> search_usda
          |                        |- success -> calculate_nutrition -> update_database -> format_response
          |                        |- failure -> estimate_nutrition -> update_database -> format_response
          |- intent="chat|query_history|get_totals" -> format_response
          |
        END

    Returns:
        Compiled StateGraph ready for execution
    """
    # Initialize graph with AgentState schema
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("analyze_request", analyze_request)
    graph.add_node("search_usda", search_usda)
    graph.add_node("estimate_nutrition", estimate_nutrition)
    graph.add_node("calculate_nutrition", calculate_nutrition)
    graph.add_node("update_database", update_database)
    graph.add_node("format_response", format_response)

    # Set entry point
    graph.set_entry_point("analyze_request")

    # Conditional edge from analyze_request (intent-based routing)
    graph.add_conditional_edges(
        "analyze_request",
        route_after_analysis,
        {
            "search_usda": "search_usda",
            "format_response": "format_response",
        },
    )

    # Conditional edge from search_usda (USDA success/failure routing)
    graph.add_conditional_edges(
        "search_usda",
        route_after_usda_search,
        {
            "calculate_nutrition": "calculate_nutrition",
            "estimate_nutrition": "estimate_nutrition",
        },
    )

    # Linear edges (sequential flow)
    graph.add_edge("calculate_nutrition", "update_database")
    graph.add_edge("estimate_nutrition", "update_database")
    graph.add_edge("update_database", "format_response")
    graph.add_edge("format_response", END)

    # Compile graph (no memory saver for now - can add later)
    return graph.compile()


# ============================================================================
# EXPORT
# ============================================================================

# Create and export the compiled graph
agent_graph = create_nutrition_agent_graph()
