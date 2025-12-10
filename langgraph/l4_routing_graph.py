from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import Optional, Literal, TypedDict, Union
from helpers.llm_manager import get_llm_instance
import dotenv

dotenv.load_dotenv()

llm = get_llm_instance()

comedian_propmt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a funny comedian who can generate one line comedy on any given topic.",
        ),
        ("human", "Tell me a joke on {topic}"),
    ]
)

comedian_llm = comedian_propmt | llm | StrOutputParser()


class TopicCategoryAssigner(BaseModel):
    topic_category: Literal["Sports", "Economics", "Science", "Other"] = Field(
        description="Category to which current topic under discussion falls to. One among Sports, Economics, Science or Unknown"
    )


class AgentState(TypedDict):
    input: str
    output: Optional[str]
    messages: list[Union[AIMessage, HumanMessage, ToolMessage]]
    topic_category: str


def topic_category_extractor(state: AgentState) -> AgentState:

    topic_extraction_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """\
You are an helpful assistant who can anlyse a message and catagorize into one of the available catogories.
Use the available tool to assign the topic category.
""",
            ),
            ("human", "{input}"),
        ]
    )
    topic_extractor_chain = topic_extraction_prompt | llm.bind_tools(
        [TopicCategoryAssigner]
    )
    response = topic_extractor_chain.invoke({"input": state["input"]})

    if len(response.tool_calls) == 0:
        state["output"] = (
            "Failed to figure of out category of topic under discussion. Please try with different prompt!"
        )
    else:
        state["topic_category"] = response.tool_calls[0]["args"]["topic_category"]

    return state


def sports_expert(state: AgentState) -> AgentState:

    state["output"] = comedian_llm.invoke({"topic": "Sports"})
    return state


def economics_expert(state: AgentState) -> AgentState:

    state["output"] = comedian_llm.invoke({"topic": "Economics"})
    return state


def science_expert(state: AgentState) -> AgentState:

    state["output"] = comedian_llm.invoke({"topic": "Science"})
    return state


def topic_catogory_router(state: AgentState) -> AgentState:

    if ("topic_category" not in state) or (not state["topic_category"]):
        state["output"] = (
            "Failed to figure of out category of topic under discussion. Please try with different prompt!"
        )
        return "OTHER"

    if state["topic_category"].strip().upper() == "OTHER":
        state["output"] = (
            "Only Sports, Science and Economics experts are available right now. Please provide one of topics related to these fields!"
        )
        return "OTHER"

    return state["topic_category"].strip().upper()


graph = StateGraph(AgentState)

graph.add_node("topic_category_extractor", topic_category_extractor)
graph.add_node("sports_expert", sports_expert)
graph.add_node("economics_expert", economics_expert)
graph.add_node("science_expert", science_expert)

graph.set_entry_point("topic_category_extractor")
graph.add_conditional_edges(
    "topic_category_extractor",
    topic_catogory_router,
    {
        "SPORTS": "sports_expert",
        "SCIENCE": "science_expert",
        "ECONOMICS": "economics_expert",
        "OTHER": END,
    },
)

app = graph.compile()
app.get_graph().draw_mermaid_png(output_file_path="outputs/l4_graph.png")


print(app.invoke({"input": "How is GDP of the country is calculated"}))
