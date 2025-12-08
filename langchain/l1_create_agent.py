from langchain.agents.factory import create_agent
from langchain_core.prompts import ChatPromptTemplate
from tools.math_tools import MATH_TOOLS
from helpers.llm_manager import get_llm_instance

llm = get_llm_instance()

llm_with_tools = llm.bind_tools(MATH_TOOLS)

agent = create_agent(
    model=llm_with_tools,
    tools=MATH_TOOLS,
    system_prompt="You are a helpful math assistant. You help solve the problems given by user. Use the tools when neccessary.",
)

while True:
    print("--" * 30)
    user_prompt = input("User Query : ")
    response = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})

    print("--" * 30)
    print(f"Agent Response : {response['messages'][-1].content}")
