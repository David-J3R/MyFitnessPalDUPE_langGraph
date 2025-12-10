import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from pachicoApp.clients.ai_engine import get_model
from pachicoApp.my_agent.utils.state import AgentState, RouteQuery
from pachicoApp.my_agent.utils.tools import log_food_entry, search_usda_foods


# --- NODE 1: ANALYZE REQUEST (ROUTER) ---
async def analyze_request(state: AgentState, config: RunnableConfig):
    """
    Decides intent and extracts structured data.
    """
    model = get_model(temperature=0)

    # We use the Structured Output directly on the model
    # method="json_mode" forces raw JSON output (no markdown wrapping)
    structured_llm = model.with_structured_output(RouteQuery, method="json_mode")

    system_prompt = """You are a nutrition assistant.
    Analyze the user's message.
    
    1. If they ate something: set intent='log_food' and extract the food text.
    2. If they ask about history (e.g., "what did I eat yesterday"): set intent='query_history'.
    3. If they ask for today's stats: set intent='get_totals'.
    4. Otherwise: set intent='chat'.
    """

    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None

    if not last_message:
        return {"intent": "chat", "search_query": None, "historical_query": None}

    response: RouteQuery = await structured_llm.ainvoke(
        [  # type: ignore[assignment]
            SystemMessage(content=system_prompt),
            last_message,
        ]
    )

    # Update state with the decision
    return {
        "intent": response.intent,
        "search_query": response.extracted_food,
        # If the user asked a question, keep it for RAG
        "historical_query": last_message.content,
    }


# --- NODE 2: SEARCH USDA ---
async def search_usda(state: AgentState):
    """
    Searches USDA. If empty, marks state for fallback.
    """
    query = state.get("search_query")
    if not query:
        return {"error": "No query"}

    # Call the tool
    results = await search_usda_foods.ainvoke({"query": query, "limit": 3})

    if not results:
        # This signals the graph to go to 'estimate_nutrition'
        return {"selected_food": None}

    # We default to the first result for now
    return {"selected_food": results[0]}


# --- NODE 3: ESTIMATE NUTRITION (Fallback) ---
async def estimate_nutrition(state: AgentState):
    """
    Uses LLM to estimate the calories if USDA fails.
    """
    model = get_model(temperature=0.4)  # Higher temp for creativity
    query = state.get("search_query", "")

    # Simple prompt - let LLM estimate the nutrition
    system_prompt = """You are a nutrition expert. Estimate nutritional content for common foods.
    Return accurate JSON with these exact fields: food_description, quantity, unit, calories, protein_g, fat_g, carbs_g"""

    user_prompt = f"""Estimate nutrition for: "{query}"

    Return JSON with realistic values for a standard serving."""

    # Get raw response and parse manually since FoodLogEntry requires fields LLM can't provide
    response = await model.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    # Parse the response content as JSON
    try:
        # Extract string content from AIMessage
        content_str = str(response.content)

        # The response might have markdown wrapping, so clean it
        if "```json" in content_str:
            content_str = content_str.split("```json")[1].split("```")[0].strip()
        elif "```" in content_str:
            content_str = content_str.split("```")[1].split("```")[0].strip()

        data = json.loads(content_str)

        # Ensure all required fields exist with defaults
        final_data = {
            "food_description": data.get("food_description", f"Estimated {query}"),
            "quantity": data.get("quantity", 1),
            "unit": data.get("unit", "serving"),
            "calories": data.get("calories", 100),
            "protein_g": data.get("protein_g", 5),
            "fat_g": data.get("fat_g", 2),
            "carbs_g": data.get("carbs_g", 10),
            "source": "llm_estimation"
        }

        return {"nutrition_data": final_data}
    except (json.JSONDecodeError, KeyError):
        # Fallback if parsing fails
        return {
            "nutrition_data": {
                "food_description": f"Estimated {query}",
                "quantity": 1,
                "unit": "serving",
                "calories": 100,
                "protein_g": 5,
                "fat_g": 2,
                "carbs_g": 10,
                "source": "llm_estimation"
            }
        }


# --- NODE 4: CALCULATE NUTRITION (The Math) ---
async def calculate_nutrition(state: AgentState):
    """
    Standardizes the USDA data into our Food Log format.
    """
    usda_item = state.get("selected_food")

    # If we already have final data (from estimation), skip
    if state.get("nutrition_data"):
        return {}

    # Handle None case
    if not usda_item:
        return {}

    # Extract Nutrients (USDA structure is nested)
    nutrients = usda_item.get("nutrients", {})

    # Simple logic: Assume 1 serving = 100g (since USDA search returns 100g)
    # *Upgrade Path:* Call 'get_usda_food_details' here if you want exact portions

    final_data = {
        "food_description": usda_item.get("description"),
        "fdc_id": usda_item.get("fdc_id"),
        "calories": nutrients.get("Energy", {}).get("value", 0),
        "protein_g": nutrients.get("Protein", {}).get("value", 0),
        "fat_g": nutrients.get("Total lipid (fat)", {}).get("value", 0),
        "carbs_g": nutrients.get("Carbohydrate, by difference", {}).get("value", 0),
        "quantity": 100,
        "unit": "g",
        "source": "usda",
    }

    return {"nutrition_data": final_data}


# --- NODE 5: UPDATE DATABASE ---
async def update_database(state: AgentState, config: RunnableConfig):
    """
    Persists data. CRITICAL: Gets user_id from config.
    """
    # 1. Get User ID from the Runtime Config (passed from Telegram)
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return {"messages": [AIMessage(content="Error: No user ID found.")]}

    data = state.get("nutrition_data")
    if not data:
        return {"messages": [AIMessage(content="Error: No nutrition data to save.")]}

    # 2. Call the Tool
    result = await log_food_entry.ainvoke(
        {
            "food_description": data["food_description"],
            "calories": data["calories"],
            "protein_g": data["protein_g"],
            "fat_g": data["fat_g"],
            "carbs_g": data["carbs_g"],
            "quantity": data["quantity"],
            "unit": data["unit"],
            "user_id": int(user_id),  # Ensure it's int for our BigInteger DB
            "fdc_id": data.get("fdc_id"),
            "source": data["source"],
        }
    )

    if "error" in result:
        return {
            "messages": [AIMessage(content="I couldn't save that to the database.")]
        }

    return {"current_daily_totals": result.get("daily_totals")}


# --- NODE 6: FORMAT RESPONSE ---
async def format_response(state: AgentState):
    """
    Generates the pretty message for the user.
    """
    model = get_model(temperature=0.7)
    intent = state.get("intent")

    if intent == "log_food":
        data = state.get("nutrition_data") or {}
        totals = state.get("current_daily_totals")

        food_desc = data.get("food_description", "food")
        calories = data.get("calories", 0)
        total_cal = totals.total_calories if totals else 0
        total_protein = totals.total_protein_g if totals else 0

        prompt = f"""
        User logged: {food_desc} ({calories} kcal).
        Daily Totals: {total_cal} kcal, {total_protein}g Protein.
        
        Write a short, encouraging message confirming this.
        """

    elif intent == "query_history":
        # (Assuming you implemented the history retrieval logic similarly to search)
        prompt = "Summarize the found food history..."

    else:
        # Chat
        messages = state.get("messages", [])
        last_content = messages[-1].content if messages else "Hello"
        prompt = f"Reply to: {last_content}"

    response = await model.ainvoke(prompt)
    return {"messages": [response]}
