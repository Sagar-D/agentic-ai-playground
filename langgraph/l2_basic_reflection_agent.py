from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Union, Optional
from helpers.llm_manager import get_llm_instance

MAX_ITERATION_FOR_REFLECTION = 10
llm = get_llm_instance()

### Create a generator llm chain ###
GENERATOR_SYSTEM_MESSAGE = """\
You are an email generation assistant.

Your job:
- Produce a clear, concise, professional, and well-structured email based on the user’s context.
- If the user provides critique or feedback, refine your previous email accordingly.

Rules:
1. YOU MUST ONLY output the email text. 
2. DO NOT include explanations, notes, tips, or meta-comments.
3. DO NOT include greetings like “Here is your email:” or any instructional text.
4. Always rewrite the full revised email from scratch when refining.
5. Keep tone, length, and style aligned with the user’s request.

Your output = the final email only.
"""

generator_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", GENERATOR_SYSTEM_MESSAGE),
        MessagesPlaceholder("messages")
    ]
)

generator_chain = generator_prompt_template | llm | StrOutputParser()


### Create a reflector llm chain ###
REFLECTOR_SYSTEM_MESSAGE = """\
You are an expert email writing critic.

Your responsibilities:
1. Understand the user’s original request and the generated email.
2. Identify weaknesses, unclear phrasing, tone issues, structural problems, or missing elements.
3. Provide direct, actionable improvement instructions.

Rules:
- Do NOT give compliments or mention what is good.
- Focus ONLY on problems and how to improve them.
- Be specific, sharp, and concise.
- If the email is already fully appropriate and requires no changes, respond ONLY with: 'LOOKS GOOD'
- Your output must contain only critique or the phrase 'LOOKS GOOD'.

IMPORTANT NOTE 
- If the email is already fully appropriate and requires no changes, respond ONLY with: 'LOOKS GOOD'
"""

reflector_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", REFLECTOR_SYSTEM_MESSAGE),
        MessagesPlaceholder("messages")
    ]
)

reflector_chain = reflector_prompt_template | llm | StrOutputParser()


class ReflexState(TypedDict) :
    input : str
    messages : list[Union[AIMessage,HumanMessage]]
    iteration_counter : int

def initiate_flow(state: ReflexState) -> ReflexState :
    if "input" not in state :
        raise ValueError("Required field 'input' is not found in Agent State")
    if "messages" not in state :
        state['messages'] = []
    if "iteration_counter" not in state :
        state["iteration_counter"] = 0
    state['messages'].append(HumanMessage(content=state['input']))
    return state

def generate_email(state:ReflexState) -> ReflexState :
    generator_response = generator_chain.invoke(state['messages'])
    state["messages"].append(AIMessage(content=generator_response))
    state['iteration_counter'] += 1
    return state

def criticize_email(state:ReflexState) -> ReflexState :
    criticism = reflector_chain.invoke(state["messages"])
    state["messages"].append(HumanMessage(content=criticism))
    return state

def reflex_exit_condition(state: ReflexState) -> str :
    if state['iteration_counter'] == MAX_ITERATION_FOR_REFLECTION :
        return "ABORT"
    message_count = len(state["messages"])
    if message_count > 2 :
        curr_message = str(state["messages"][-2]).upper()
        if curr_message.strip() == "" or "LOOKS GOOD" in curr_message :
            return "ABORT"
    return "CONTINUE"

reflex_graph = StateGraph(ReflexState)
reflex_graph.add_node("init", initiate_flow)
reflex_graph.add_node("generator", generate_email)
reflex_graph.add_node("reflector", criticize_email)

reflex_graph.set_entry_point("init")
reflex_graph.add_edge("init", "generator")
reflex_graph.add_conditional_edges("generator", reflex_exit_condition, {
    "ABORT" : END,
    "CONTINUE" : "reflector"
})
reflex_graph.add_edge("reflector", "generator")

app =reflex_graph.compile()

app.get_graph().draw_mermaid_png(output_file_path="outputs/graph.png")
print(app.get_graph().draw_ascii())
print("--"*30)
print("For visual mermaid graph refer to file path 'langgraph/graph.png'")
print("--"*30+ "\n")


if __name__ == "__main__" :

    print("\n\nThis tool helps generate a email message for a given topic and context. Try out!!\n\n")

    user_prompt = input("User Prompt : ")
    for step in app.stream({"input":user_prompt}, stream_mode="values") :
        print("--"*30)
        if "iteration_counter" in step :
            print(f"Iteration {step['iteration_counter']}")
        if "messages" in step :
            print(f"Message : {step["messages"][-1].content}")

