from langchain_core.messages import HumanMessage
from adv_agent import graph

def run_research_query(query: str):

    initial_state = {
        "query": query,

        "messages": [
            HumanMessage(content=query)
        ],

        "proteins": [],
        "go_terms": [],
        "entities": []
    }

    result = graph.invoke(
        initial_state,
        config={"recursion_limit": 10}
    )

    print("\n========== TOOL EXECUTION ==========\n")

    for message in result["messages"]:

        if getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                print(f"Tool: {call['name']}")
                print(f"Arguments: {call['args']}")
                print()

    print("\n========== FINAL ANSWER ==========\n")

    print(result["messages"][-1].content)

    print("\n========== GRAPH STATE ==========\n")

    print("Proteins:", len(result.get("proteins", [])))
    print("GO terms:", len(result.get("go_terms", [])))
    print("Entities:", len(result.get("entities", [])))

    return result


if __name__ == "__main__":

    query = (
        "Find proteins associated with kinase activity"
    )

    result = run_research_query(query)
