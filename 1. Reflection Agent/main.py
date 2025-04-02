from typing import List, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import START, END, StateGraph, MessageGraph
from chains import generate_chain, reflect_chain

load_dotenv()

REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: Sequence[BaseMessage]):
    return generate_chain.invoke({'messages': state})


# def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
#     response = reflect_chain.invoke({'messages': messages})
#     return [HumanMessage(content=response.content)]
def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    response = reflect_chain.invoke({'messages': messages})
    if not response.content.strip():
        return messages  # Preserve original messages if reflection is blank
    # Append instead of replacing
    return messages + [HumanMessage(content=response.content)]


builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    else:
        return REFLECT


builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()

if __name__ == "__main__":
    inputs = HumanMessage(content="""
                            Mondays are just mini New Years—fresh starts every week. Let’s make it count! 💪 #MondayMotivation #NewWeekNewGoals
                            """
                          )
    response = graph.invoke([inputs])
