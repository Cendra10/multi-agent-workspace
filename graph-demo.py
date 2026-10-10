from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from llm import ask

class State (TypedDict):
    task: str
    plan: str
    result: str

def planner(state: State) -> dict:
    system_prompt = "You are a planner. Create a short plan in 2-3 steps."
    user_prompt = f"Create a plan for this task: {state['task']}"

    plan = ask(system_prompt, user_prompt)
    return {"plan": plan}

def worker (state: State) -> dict:
    return {"result": f"Done using: {state['plan']}"}

builder = StateGraph(State)
builder.add_node("planner", planner)
builder.add_node("worker", worker)
builder.add_edge(START, "planner")
builder.add_edge("planner", "worker")
builder.add_edge("worker", END)

graph = builder.compile()

output = graph.invoke({"task": "write a summary"})
print(output)