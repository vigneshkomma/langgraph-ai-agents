import requests
from typing import TypedDict, Annotated, List
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

# 1. State Defination
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage],add_messages]
    user_info: dict
    iteration_count: int
    is_authorised: bool

# 2. Define the tool 
@tool
def searxng_search(query: str):

    """
    Search the web for current events, news, or real-time information.
    Use this whenever the user asks a question about the outside world.
    """

    url = "http://localhost:8080/search"
    params = {"q":query,"format":"json"}

    try:
        res = requests.get(url,params = params, timeout = 10)
        res.raise_for_status()
        data = res.json()

        formatted_result = []

        #Grab the top 5 results
        raw_resluts = data.get("results",[])[:5]

        for r in raw_resluts:
            title = r.get("title","No title")
            link = r.get("url","No URL")

            snippet = r.get("content","No Description available.")

            clean_snippet = (snippet[:200]+'...') if len(snippet) > 200 else snippet

            formatted_result.append(f"Title: {title}\nURL: {link}\nSnippet: {clean_snippet}\n")

        #Add infobox data if available
        infoboxes = data.get("infoboxes",[])
        if infoboxes:
            info = infoboxes[0].get("content","")
            if info:
                formatted_result.insert(0,f"SUMMARY INFO: {info}\n---")

        if not formatted_result:
            return "No Relavent Search Results Found"

        return "\n".join(formatted_result)

    except Exception as e:
        return f"Search failed:{str(e)}"
    


# 4. Agent Node
def agent_node(state:AgentState, llm) -> AgentState:

    print(f"--- processing for user: {state['user_info'].get('name')}---")


    system_prompt = SystemMessage(content="""
    You are a helpful assistant with web search capabilities.
    CRITICAL RULES:
    1. BEFORE using a tool, check if the information is already in your chat history.
    2. DO NOT search for information regarding our current or previous conversation. You have access to the history directly.
    3. Use the search tool ONLY for factual, real-time, or external information that is not present in the chat history.
    """)

    response = llm.invoke([system_prompt] + state["messages"])

    return {
        "messages":[response],
        "iteration_count":state["iteration_count"] + 1
    }


#build graph
def build_graph(llm):
    builder = StateGraph(AgentState)

    builder.add_node("agent",lambda s: agent_node(s,llm))
    builder.add_node("tools",ToolNode([searxng_search]))

    builder.add_edge(START,"agent")
    builder.add_conditional_edges("agent",tools_condition)
    builder.add_edge("tools","agent")

    return builder.compile()

def main():

    llm = ChatOllama(model='llama3.1:latest',temperature=0).bind_tools([searxng_search])
    app = build_graph(llm)

    current_state = {
        "messages": [],
        "user_info": {"name":"carry","tier":"local"},
        "iteration_count": 0,
        "is_authorised": True
    }

    print("Chat has begin, enter 'exit' to stop\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == 'exit':
            break

        new_input = {"messages":[HumanMessage(content=user_input)]}

        current_state = app.invoke({**current_state,**new_input})

        print(f"Agent: {current_state['messages'][-1].content}")
        print(f"(Debug: Iteration #{current_state['iteration_count']})\n")

if __name__ == "__main__":
    main()

