import requests
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage


#state graph
class AgentState(TypedDict):
    message: List[BaseMessage]
    next: str


#tool

def searxng_search(query:str):
    url = "http://localhost:8080/search"
    params = {"q":query,"format":"json"}

    res = requests.get(url, params=params)

    data = res.json()

    return data

#tool node
def tool_node(state: AgentState) -> AgentState:
    last_msg = state["message"][-1].content
    search_result = searxng_search(last_msg)

    state["message"].append(AIMessage(content=search_result))

    return state


#agent node
def agent_node(state:AgentState, llm) -> AgentState:
    response = llm.invoke(state["message"])
    state["message"].append(response)

    text = response.content.lower()

    if "search:" in text:
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
