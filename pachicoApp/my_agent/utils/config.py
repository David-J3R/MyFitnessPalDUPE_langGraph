from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel


class Configuration(BaseModel):
    """Configuration for the agent, allowing dynamic model switching"""

    model_name: str = "amazon/nova-2-lite-v1:free"  # Default model

    # Classmethod for creating Configuration from RunnableConfig
    # it extracts the 'configurable' section from the RunnableConfig
    @classmethod
    def from_runnable_config(cls, config: RunnableConfig = None) -> "Configuration":
        config = config or {}
        configurable = config.get("configurable", {})
        return cls(**configurable)
