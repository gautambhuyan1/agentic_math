from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

llm = init_chat_model(
    "anthropic:claude-3-5-haiku-latest"
)

@tool
def math_operations(numA: float, numB: float, op: str) ->float:
    '''Return the math operation of two numbers
    :param numA: first number, numB: second number, op: mathematical operation ADD, SUB or MUL
    :return: mathematical operation of the two numbers requested
    '''
    print("Inside math function ", numA, numB, op)
    return {
        "ADD": numA + numB,
        "SUB": numA - numB,
        "MUL": numA * numB
    }.get(op, 0.0)

tools = [math_operations]

llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_node("tools", ToolNode(tools))

graph_builder.add_edge(START, "chatbot")

graph_builder.add_conditional_edges("chatbot", tools_condition)

graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

user_input = input("Enter operation:")

state = graph.invoke({"messages":[{"role":"user", "content": user_input}]})

print(state["messages"])
print(state["messages"][-1].content)
