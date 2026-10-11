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
    system_prompt = "You are a worker. Execute the task by following the provided plan."
    user_prompt = f"Task: {state['task']}\nPlan: {state['plan']}"

    result = ask(system_prompt, user_prompt)
    return {"result": result}

def supervisor (state: State) -> str:
    plan = state.get("plan")
    result = state.get("result")

    if not plan:
        return ("planner")
    elif not result:
        return ("worker")
    else:
        return END

builder = StateGraph(State)
builder.add_node("planner", planner)
builder.add_node("worker", worker)
builder.add_conditional_edges(START, supervisor)
builder.add_conditional_edges("planner", supervisor)
builder.add_conditional_edges("worker", supervisor)

graph = builder.compile()

output = graph.invoke({
    "task": "Explain what a REST API is in 3 sentences"
})
print(output)