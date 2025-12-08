from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain.agents.factory import create_agent
from helpers.llm_manager import get_llm_instance

wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

llm = get_llm_instance("ollama")

SYSTEM_PROMPT = """
You are an agent whose only purpose is to use the Wikipedia search tool to retrieve information about a user-requested topic.
Follow these rules:

Always call the Wikipedia tool. Never answer directly from your own knowledge.

After receiving the tool result, extract exactly 5 interesting and distinct facts about the topic.

Return the final answer as 5 bullet points only.

Do not add explanations, notes, or extra text outside the bullet points.

If the tool returns no results, reply with:
"I'm unable to find information on that topic."
"""

wiki_agent = create_agent(
    model=llm,
    tools=[wiki_tool],
    system_prompt=SYSTEM_PROMPT
)

if __name__ == "__main__" :

    print("Ask about any topic. Our agent will fetch 5 interseting facts about that from Wikipidea")

    user_prompt = input("Topic : ")
    agent_resp = wiki_agent.invoke({"messages" : user_prompt})

    print(agent_resp['messages'][-1].content)