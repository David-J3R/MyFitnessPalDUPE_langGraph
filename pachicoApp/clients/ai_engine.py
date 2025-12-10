from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from pachicoApp.config import config

template = """Question: {question}
Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])


def get_model(model_name: str = "amazon/nova-2-lite-v1:free", temperature: float = 0.7):
    """
    Returns a chat model instance connected to OpenRouter.
    You can change 'model_name' to any model on OpenRouter.
    """
    return ChatOpenAI(
        model=model_name,
        api_key=config.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
    )


if __name__ == "__main__":
    # Test with a cheap model first
    llm = get_model()
    llm_chain = prompt | llm
    question = "Hello! Are you ready to track my calories?"

    print(llm_chain.invoke({"question": question}))
