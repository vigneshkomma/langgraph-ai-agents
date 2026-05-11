import requests
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool



# 1. Define the tool 
@tool
def searxng_search(query:str):

    """
    Search the web for current events, news, or real-time information.
    Use this whenever the user asks a question about the outside world.
    """

    url = "http://localhost:8080/search"
    params = {"q":query,"format":"json"}

    try:
        res = requests.get(url,params=params)
        data = res.json()
        results = []
        for r in data.get("results",[])[:5]:
            results.append(f"{r['title']} - {r['url']}")
        return "\n".join(results)
    except Exception as e:
        return f"Search failed:{str(e)}"
    

# 2. State Defination
class AgentState(TypedDict):
    messages: List[BaseMessage]
    next: str


# 3. Tool Node
def tool_node(state: AgentState) -> AgentState:
    last_msg = state["messages"][-1]

    if hasattr(last_msg,'tool_calls'):
        for tool_call in last_msg.tool_calls:
            content = searxng_search.invoke(tool_call["args"])

            result_msg = ToolMessage(content=content,tool_call_id=tool_call["id"])

            state["messages"].append(result_msg)
    
    return state


# 4. Agent Node
def agent_node(state:AgentState, llm) -> AgentState:

    system_prompt = SystemMessage(content="""
    You are a helpful assistant with web search capabilities
    """)

    messages = [system_prompt] + state["messages"]  
    response = llm.invoke(messages)
    state["messages"].append(response)


    if response.tool_calls:
        state["next"] = "tool"
    else:
        state["next"] = "end"

    return state


#router
def route(state:AgentState):
    return state["next"]

#build graph
def build_graph(llm):
    graph = StateGraph(AgentState)

    graph.add_node("agent",lambda s:agent_node(s,llm))
    graph.add_node("tool",tool_node)

    graph.add_edge(START,"agent")
    graph.add_conditional_edges(
        "agent",
        route,
        {
            "tool":"tool",
            "end": END
        }
    )

    graph.add_edge("tool","agent")

    return graph.compile()


def main():

    llm = ChatOllama(model='llama3.1:latest',temperature=0)

    llm_with_tools = llm.bind_tools([searxng_search])

    app = build_graph(llm_with_tools)
    state = {"messages":[]}

    print("Chat has begin, enter 'exit' to stop\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == 'exit':
            break

        state["messages"].append(HumanMessage(content=user_input))

        state = app.invoke(state)

        final_response = state["messages"][-1].content
        print(f"Agent: {final_response}\n")


if __name__ == "__main__":
    main()

