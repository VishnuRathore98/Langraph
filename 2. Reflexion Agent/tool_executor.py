from typing import List
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, HumanMessage
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
# from langgraph.prebuilt.tool_node import ToolExecutor, ToolInvocation
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from chains import parser
from collections import defaultdict
import json
from schemas import AnswerQuestion, Reflection
from dotenv import load_dotenv

load_dotenv()

# Using Tavily for making search requests
search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)
# Casting Tavily into a tool
tool_executor = ToolNode(tools=[tavily_tool])


# Tavily tool execution function

def execute_tools(state: List[BaseMessage]) -> List[ToolMessage]:
    """
    Executes tools based on the latest AIMessage in the message history.
    Assumes the last message includes tool_calls that match the ToolNode tools.
    """
    last_message = state[-1]

    if not isinstance(last_message, AIMessage):
        raise ValueError("Last message must be an AIMessage with tool_calls.")

    # Use ToolNode to automatically resolve tools and return ToolMessages
    result = tool_executor.invoke(state)

    # Ensure result is returned as a list
    return result if isinstance(result, list) else [result]


# def execute_tools(state: List[BaseMessage]) -> List[ToolMessage]:
#     # Getting the AIMessage from the state
#     tool_invocation: AIMessage = state[-1]

#     # Calling the JsonOutputToolsParser, and passing in the AI response which is in JSON, to convert it into python dictionary.
#     parsed_tool_calls = parser.invoke(tool_invocation)

#     # Storing ids
#     ids = []
#     # Storing tool invocation results
#     tool_invocations = []

#     # Gettin AI responses one by one
#     for parsed_calls in parsed_tool_calls:
#         # Getting search queries for Tavily to search for
#         for query in parsed_calls['args']['search_queries']:
#             tool_invocations.append({
#                 "name": "tavily_search_results_json",
#                 "args": query,
#             })
#             ids.append(parsed_calls['id'])

#     outputs = tool_executor.invoke(tool_invocations)

#     # Map each output to its corresponding ID and tool input
#     outputs_map = defaultdict(dict)
#     for id_, output, invocation in zip(ids, outputs, tool_invocations):
#         outputs_map[id_][invocation.tool_input] = output

#     # Converts the mapped outputs to ToolMessage objects
#     tool_messages=[]
#     for id_, mapped_output in outputs_map.items():
#         tool_messages.append(
#             ToolMessage(content=json.dumps(mapped_output), tool_call_id=id)
#         )
    
#     return tool_messages
# ----------------------------------------------------------------
# ------------------------------For testing-------------
# if __name__=="__main__":
#     print("Tool executor enter")

#     human_message=HumanMessage(
#         content="Write about AI-Powered SOC / autonomous soc  problem domain,"
#         " list startups that do that and raised capital."
#     )

#     answer = AnswerQuestion(
#         answer="",
#         reflection=Reflection(missing="", superfluous=""),
#         search_queries=[
#              "AI-powered SOC startups funding",
#             "AI SOC problem domain specifics",
#             "Technologies used by AI-powered SOC startups",
#         ],
#         id="call_KpYHichFFEmLitHFvFhKy1Ra"
#     )

#     raw_res = execute_tools(
#         state = [
#             HumanMessage(content="Write about AI-Powered SOC..."),
#             AIMessage(
#                 content="",
#                 tool_calls=[
#                     {
#                         "name": "tavily_search_results_json",
#                         "args": {"query": "AI-powered SOC startups funding"},
#                         "id": "call_xyz"
#                     }
#                 ]
#         )]

#         # state=[
#         #     human_message,
#         #     AIMessage(
#         #         content="",
#         #         tool_calls=[
#         #             {
#         #                 "name":AnswerQuestion.__name__,
#         #                 "args":answer.model_dump(),
#         #                 "id":"call_KpYHichFFEmLitHFvFhKy1Ra"
#         #             }
#         #         ]
#         #     )
#         # ]
#     )
#     print(raw_res)
