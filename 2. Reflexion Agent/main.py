from typing import List
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import MessageGraph, END
from tool_executor import execute_tools
from chains import revisor, first_responder

MAX_ITERATIONS = 2
builder = MessageGraph()
builder.add_node("draft", first_responder)
builder.add_node("execute_tools",execute_tools)
builder.add_node("revisor", revisor)
builder.add_edge("draft","execute_tools")
builder.add_edge("execute_tools","revisor")

def event_loop(state: List[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visits
    if num_iterations>MAX_ITERATIONS:
        return END
    return "execute_tools"

builder.add_conditional_edges("revisor", event_loop)
builder.set_entry_point("draft")

graph = builder.compile()

res = graph.invoke("Write about AI-Powered SOC / autonomous soc  problem domain, list startups that do that and raised capital.")

print(res[-1].tool_calls[0]['args']['answer'])
print(res)