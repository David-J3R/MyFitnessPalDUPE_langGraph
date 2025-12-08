from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from pachicoApp.config import config as app_config

from .config import Configuration
from .state import AgentState, RouteQuery


def get_model(config: Configuration):
    return ChatOpenAI(
        model=config.model_name,
        api_key=app_config.OPENROUNTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
    )


async def analyze_request(state: AgentState, config: RunnableConfig):
    """Analizes the user's input to decide the next step.
    Uses structured output schema defined in state.py.
    """

    configuration = Configuration.from_runnable_config(config)
    model = get_model(configuration)

    # Bind the Pydantic schema to the model
    structured_llm = model.with_structured_output(RouteQuery)

    system_prompt = """You are a nutrition expert assistant.
    Analize the user's input.
    - If the user mention eating food (e.g., "I had an apple", "I ate 2 slices of pizza"),
    route to 'search_usda'.
    - If the user is just greeting or asking general questions (e.g., "How are you?"),
    route to 'direct_answer'."""

    # We look at the last message
    last_message = state["messages"][-1]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Declarative Chain
    chain = prompt | structured_llm

    # Invoke the model with the prompt and last message
    # Note: It is not necessary an await since prompt and str
    decision = chain.invoke({"input": last_message.content})

    return {
        "search_query": decision.extracted_food,
        "nutrition_data": {"decision": decision},
    }
