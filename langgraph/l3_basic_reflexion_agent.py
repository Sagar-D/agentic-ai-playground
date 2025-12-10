from langgraph.graph import StateGraph, START, END
from langchain.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from pydantic import BaseModel, Field
from typing import TypedDict, Optional, Union
from helpers.llm_manager import get_llm_instance


import dotenv
from pprint import pprint
import ast

dotenv.load_dotenv()

MAX_REVISION_COUNT = 3

llm = get_llm_instance("ollama")

SPORTS_BOT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """\
You are a high-accuracy Sports Information Agent.  
Your responsibilities:

1. Produce a clear, concise, and factual answer to the user’s sports question using only your internal knowledge.  
2. After generating the answer, critically evaluate it:
   - Identify missing details that would make the answer more complete.
   - Identify any unnecessary or irrelevant information you added.
3. Generate a list of high-quality web search queries that would help fetch fresh, up-to-date, or more detailed knowledge relevant to the user’s question.
4. Always return your output using the structured tool format provided.

IMPORTANT NOTE : You always respond with tool call / function call
""",
        ),
        ("human", "{input}"),
    ]
)

SPORTS_EXPERT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """\
You are an advanced Sports Knowledge Reviewer and Revision Specialist.

Your role:
1. Review the previous draft answer, its criticism, and the retrieved web search results.
2. Revise the original answer by incorporating updated information, correcting inaccuracies, and improving clarity.
3. After producing the revised answer, critique it:
   - List missing information that would further improve accuracy or completeness.
   - List unnecessary or irrelevant content that should be removed in future revisions.
4. Suggest additional web search queries that could further improve the answer with more recent or authoritative data.
5. Always respond using the required structured tool format for revised answers.

IMPORTANT NOTE : You always respond with tool call / function call
""",
        ),
        MessagesPlaceholder("messages"),
    ]
)


class Criticism(BaseModel):
    missing: list[str] = Field(description="Things that are missing in the content")
    unnecessary: list[str] = Field(description="Things that are not necessary")


class AnswerSportsQuery(BaseModel):
    answer: str = Field(description="Answer to the users sports query")
    criticism: Criticism = Field(
        description="Critique addressing missing info and unneccsary extra info in the generated answer."
    )
    search_queries: list[str] = Field(
        description="List of we search queries that can be used to fetch latest knowlege related to the topic "
    )


class ReviseSportsAnswer(AnswerSportsQuery):
    revision_needed: bool = Field(
        "Boolean flag representing whether the answer should be revised. Set it to True, if answer needs some revision. If answe looks good and no more revision is needed, set it to False"
    )
    pass


sports_bot_chain = SPORTS_BOT_PROMPT | llm.bind_tools([AnswerSportsQuery])
sports_expert_chain = SPORTS_EXPERT_PROMPT | llm.bind_tools([ReviseSportsAnswer])


class AgentState(TypedDict):
    input: str
    messages: Optional[list[Union[HumanMessage, AIMessage, ToolMessage]]]
    output: Optional[str]
    exit_loop: bool = False


def generate_response(state: AgentState) -> AgentState:

    if not (state.get("input") and len(state.get("input").strip()) > 0):
        raise ValueError("Required field 'input' is missing in Agent State")
    if "messages" not in state:
        state["messages"] = []
    if "exit_loop" not in state:
        state["exit_loop"] = False

    state["messages"].append(HumanMessage(content=state["input"]))
    response = sports_bot_chain.invoke({"input": state["input"]})

    if response.content.strip() == "" and len(response.tool_calls) != 0:
        response.content = response.tool_calls[0]["args"]["answer"]
    state["messages"].append(response)
    return state


def fetch_latest_knowledge(state: AgentState) -> AgentState:

    if (
        type(state["messages"][-1]) != AIMessage
        or len(state["messages"][-1].tool_calls) == 0
    ):
        state["exit_loop"] = True
        return state

    tool_call = state["messages"][-1].tool_calls[0]

    previous_answer = tool_call["args"]["answer"]
    previous_criticism = tool_call["args"]["criticism"]
    search_queris = tool_call["args"]["search_queries"]
    knowledge_base = []

    serach_tool = TavilySearchResults(
        api_wrapper=TavilySearchAPIWrapper(), max_results=3
    )
    for query in search_queris:
        search_results = serach_tool.invoke(query)
        knowledge = {
            "query": query,
            "results": [
                {result["title"], result["url"], result["content"]}
                for result in search_results
            ],
        }
        knowledge_base.append(knowledge)

    tool_message_content = str(
        {
            "previous_answer": previous_answer,
            "criticism": previous_criticism,
            "latest_knowledge": knowledge_base,
            "instruction": "Revise the previous answer using the criticism and fresh knowledge.",
        }
    )

    state["messages"].append(
        ToolMessage(content=tool_message_content, tool_call_id=tool_call["id"])
    )
    return state


def revise_results(state: AgentState) -> AgentState:

    response = sports_expert_chain.invoke({"messages": state["messages"]})

    if len(response.tool_calls) == 0:
        state["exit_loop"] = True
    elif not response.tool_calls[0]["args"]["revision_needed"]:
        state["exit_loop"] = True
    else:
        for tool_call in response.tool_calls:
            if tool_call["name"] in [
                ReviseSportsAnswer.__name__,
                AnswerSportsQuery.__name__,
            ]:
                response.content = tool_call["args"]["answer"]

    state["messages"].append(response)
    return state


def conditinal_abort(state: AgentState) -> AgentState:

    revision_count = len([msg for msg in state["messages"] if type(msg) == ToolMessage])
    if state["exit_loop"] or revision_count >= MAX_REVISION_COUNT:
        return "EXIT"
    return "REVISE"


def populate_output_string(state: AgentState) -> AgentState:

    response_message = state["messages"][-1]
    for message in reversed(state["messages"]):
        if type(response_message) == AIMessage:
            response_message = message
            break

    if response_message.content.strip() != "":
        state["output"] = response_message.content
    elif len(response_message.tool_calls) != 0:
        state["output"] = response_message.tool_calls[0]["args"]["answer"]
    else:
        state["output"] = (
            "Sorry, Something went wrong!\n I am not able to process your request for now. Please try again later!"
        )
    return state


graph = StateGraph(AgentState)

graph.add_node("generator", generate_response)
graph.add_node("search_tool", fetch_latest_knowledge)
graph.add_node("revisor", revise_results)
graph.add_node("result_generator", populate_output_string)

graph.add_edge(START, "generator")
graph.add_edge("generator", "search_tool")
graph.add_edge("search_tool", "revisor")
graph.add_conditional_edges(
    "revisor", conditinal_abort, {"REVISE": "search_tool", "EXIT": "result_generator"}
)
graph.add_edge("result_generator", END)

app = graph.compile()

print(app.get_graph().draw_ascii())
app.get_graph().draw_mermaid_png(output_file_path="outputs/l3_graph.png")

final_response = None
for step in app.stream(
    {"input": "What is the latest score on recent India vs South Africa ODI series?"},
    stream_mode="values",
):
    if "messages" in step:
        cur_msg = step["messages"][-1]
        if type(cur_msg) == AIMessage:
            print(f"AI : {cur_msg.content}", end="")
            if len(cur_msg.tool_calls) > 0:
                pprint(cur_msg.tool_calls[0]["args"])
        elif type(cur_msg) == ToolMessage:
            print("TOOL : ", end="")
            pprint(ast.literal_eval(cur_msg.content))
        else:
            print("HUMAN : ", end="")
            print(cur_msg.content)
    final_response = step

print("\n\n")
print("--" * 30)
print(f"Final Response : {final_response['output']}")
print("--" * 30)
