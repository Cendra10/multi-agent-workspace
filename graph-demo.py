from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State (TypedDict):
    task: str
    plan: str
    result: str

def planner(state: State) -> dict:
    return {"plan": f"Plan for: {state['task']}"}

def worker (state: State) -> dict:
    return {"result": f"Done using: {state['plan']}"}

builder = StateGraph(State)
builder.add_node("planner", planner)
builder.add_node("worker", worker)
builder.add_edge(START, "planner")
builder.add_edge("planner", "worker")
builder.add_edge("worker", END)

graph = builder.compile()

output = graph.invoke({"task": "write a sumamry"})
print(output)