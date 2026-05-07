import requests
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

#state graph
class AgentState(TypedDict):
    messages: List[BaseMessage]
    next: str


#tool

def searxng_search(query:str):
    url = "http://localhost:8080/search"
    params = {"q":query,"format":"json"}

    res = requests.get(url, params=params)

    data = res.json()

    results = []
    for r in data.get("results",[])[:5]:
        results.append(f"{r['title']} - {r['url']}")
    
    return "\n".join(results)

#tool node
def tool_node(state: AgentState) -> AgentState:
    last_msg = state["messages"][-1].content

    query = last_msg.replace("search:","").strip()

    search_result = searxng_search(query)

    state["messages"].append(AIMessage(content=f"search results:\n{search_result}"))

    return state


#agent node
def agent_node(state:AgentState, llm) -> AgentState:

    system_prompt = SystemMessage(content="""
    You are an AI agent with access to a web search tool.

    IMPORTANT RULES:
    - If the question requires current, recent, or real-time information, you MUST use the search tool.
    - Do NOT say you lack access to real-time data.
    - Instead, call the tool.

    To use the tool, respond EXACTLY like this:
    search: <your query>

    Examples:
    User: Who won the latest F1 race?
    You: search: latest F1 race winner

    If you already have enough information, answer normally.
    """)
    messages = [system_prompt] + state["messages"]  
    response = llm.invoke(messages)
    state["messages"].append(response)

    text = response.content.lower()

    if text.startswith("search:"):
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
    state = {"messages":[]}

    llm = ChatOllama(model='llama3:8b',temperature=0)

    app = build_graph(llm)

    print("Chat has begin, enter 'exit' to stop\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == 'exit':
            break

        state["messages"].append(HumanMessage(content=user_input))

        state = app.invoke(state)

        print('Agent: ',state["messages"][-1].content,"\n")


main()

