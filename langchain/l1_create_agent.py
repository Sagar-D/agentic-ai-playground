from langchain.agents.
from langchain_core.prompts import ChatPromptTemplate
from tools.math_tools import MATH_TOOLS
from helpers.llm_manager import get_llm_instance

llm = get_llm_instance("ollama")

llm_with_tools = llm.bind_tools(MATH_TOOLS)

agent = create_agent(
    model=llm_with_tools,
    tools=MATH_TOOLS,
    system_prompt="You are a helpful math assistant. You help solve the problems given by user. Use the tools when neccessary."
)

while True:
    user_prompt = input("User Query : ")
    response = agent.invoke({"input": user_prompt})

    print("--"*30)
    print(f"\n\nAgent Response : {response}")

