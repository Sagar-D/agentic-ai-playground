from langgraph.graph.state import StateGraph, END
from typing import TypedDict, Optional
import random
import string

class AgentState(TypedDict) :
    number:Optional[int]
    letter:Optional[str]

graph = StateGraph(state_schema=AgentState)

def incrementer(state:AgentState) -> AgentState :

    if "number" not in state :
        state["number"] = 0
    
    state['number'] = state['number'] + 1
    state['letter'] = random.choice(string.ascii_uppercase)
    
    return state

def state_printer(state:AgentState) -> AgentState :

    print(f"Current State - number : {state['number']}, letter : {state['letter']}")
    return state

def end_loop_decider(state:AgentState) -> string :

    if state['number'] == 10 :
        return "ABORT"
    return "CONTINUE"

graph.add_node("incrementer", incrementer)
graph.add_node("printer", state_printer)
graph.set_entry_point("incrementer")
graph.add_edge("incrementer", "printer")
graph.add_conditional_edges("printer", end_loop_decider, {
    "ABORT" : END,
    "CONTINUE": "incrementer"
})

app =graph.compile()

app.invoke({"number":0, "letter": "x"})