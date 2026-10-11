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

def supervisor_llm(state:State) -> str:
    system_prompt = (
        "Choose exactly one: planner, worker or FINISH. "
        "If plan is missing, choose planner. "
        "If plan exists but result is missing, choose worker. "
        "If both exist, choose FINISH. "
        "Reply with one word only. "
    )
    user_prompt = (
        f"Task: {state.get('task', '')}\n"
        f"Plan: {state.get('plan', '')}\n"
        f"Result: {state.get('result', '')}"
    )
    source = ask(system_prompt, user_prompt)
    source_result = source.strip().lower()

    expected = supervisor(state)

    if source_result == "finish":
        source_result = END

    if source_result != expected:
        print(f"LLM ditolak: {source_result}, gunakan: {expected}")

        source_result = expected
    
    return source_result

builder = StateGraph(State)
builder.add_node("planner", planner)
builder.add_node("worker", worker)
builder.add_conditional_edges(START, supervisor_llm)
builder.add_conditional_edges("planner", supervisor_llm)
builder.add_conditional_edges("worker", supervisor_llm)

graph = builder.compile()

output = graph.invoke({
    "task": "Explain what a REST API is in 3 sentences"
})
print(output)