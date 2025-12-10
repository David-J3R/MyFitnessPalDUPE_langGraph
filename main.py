"""
Pachico - AI Nutrition Tracking Agent

Test script for the LangGraph agent
"""

import asyncio

from langchain_core.messages import HumanMessage

from pachicoApp.database.schema import database
from pachicoApp.my_agent.graph import agent_graph


async def main():
    """Test the nutrition tracking agent"""

    # Connect to database
    await database.connect()

    try:
        # Test 1: Log food (USDA search)
        print("\n=== Test 1: Logging food with USDA ===")
        result = await agent_graph.ainvoke(
            {
                "messages": [HumanMessage(content="I ate 2 large eggs")],
                "session_id": "test-session-001",
            },
            config={"configurable": {"user_id": 1}},
        )

        print(f"Final message: {result['messages'][-1].content}")
        print(f"Intent: {result.get('intent')}")
        print(f"Daily totals: {result.get('current_daily_totals')}")

        # Test 2: General chat
        print("\n=== Test 2: General chat ===")
        result2 = await agent_graph.ainvoke(
            {
                "messages": [HumanMessage(content="Hello! How are you?")],
                "session_id": "test-session-002",
            },
            config={"configurable": {"user_id": 1}},
        )

        print(f"Final message: {result2['messages'][-1].content}")
        print(f"Intent: {result2.get('intent')}")

    finally:
        # Disconnect from database
        await database.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
