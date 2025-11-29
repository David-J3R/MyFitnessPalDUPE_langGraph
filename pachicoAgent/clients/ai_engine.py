from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from pachicoAgent.config import config

template = """Question: {question}
Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])


def get_chat_model(model_name="x-ai/grok-4.1-fast:free"):
    """
    Returns a chat model instance connected to OpenRouter.
    You can change 'model_name' to any model on OpenRouter.
    """
    return ChatOpenAI(
        model=model_name,
        api_key=config.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.7,
    )


if __name__ == "__main__":
    # Test with a cheap model first
    llm = get_chat_model("x-ai/grok-4.1-fast:free")
    llm_chain = prompt | llm
    question = "Hello! Are you ready to track my calories?"

    print(llm_chain.invoke({"question": question}))
