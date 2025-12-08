from langchain.agents.factory import create_agent
from langchain.tools import tool
from langchain.messages import AIMessage, ToolMessage, HumanMessage
from helpers.llm_manager import get_llm_instance
from l4_sql_agent import sql_agent
from l5_wiki_facts_agent import wiki_agent

llm = get_llm_instance("ollama")

@tool
def music_database_agent(prompt:str) -> str:
    """
    Tool to answer any queries related to english music from a music store database.
    This tool has access to a huge database of English Tracks, Albums and Artists. 
    It can answer any stats related query regarding music artists, tracks and their albums.

    Available tables in database: ['Album', 'Artist', 'Customer', 'Employee', 'Genre', 'Invoice', 'InvoiceLine', 'MediaType', 'Playlist', 'PlaylistTrack', 'Track']

    Arguments :
    prompt (str) - A natural language prompt related to music (Eg : "Most sold album in 2000")
    
    Return :
    (str) - A natural language response to the user prompt with appropriate data (Eg : "")
    """

    agent_resp = sql_agent.invoke({"messages":prompt})

    return agent_resp["messages"][-1].content

@tool
def interesting_facts_generator(topic:str) -> str :
    """
    This tool generates a 5 inetresting facts about any given topic
    
    Arguments :
    - topic (str) : A string representing topic name.

    Return :
    - (str) : A String with five interesting facts.
    """
    
    wiki_resp = wiki_agent.invoke({"messages":topic})

    return wiki_resp["messages"][-1].content

SUPERVISER_SYSTEM_MESSAGE = """You are a supervisor agent that coordinates two tools:

1. music_database_agent — Use this tool to understand the user’s music-related query and obtain factual database results.
2. interesting_facts_generator — Use this tool to generate 5 interesting facts about a topic.

Your workflow MUST follow these steps:

STEP 1 — Use music_database_agent  
- When the user asks any question related to music, artists, albums, tracks, or statistics, you MUST first call music_database_agent with the user’s full query.  
- Do NOT infer or guess. Only use data returned by the tool.

STEP 2 — Derive a topic for interesting facts  
- From the database result, determine the correct **topic** for generating interesting facts.  
- Usually this will be the name of an artist, album, track, or other music-related entity mentioned in the database response.

STEP 3 — Use interesting_facts_generator  
- Call interesting_facts_generator with the derived topic.  
- Wait for the tool’s result and present it to the user.

General Rules:
- NEVER answer directly from your own knowledge.
- NEVER skip a tool. Always follow the 3-step sequence above. DO NOT SKIP ANY STEP.
- If the database tool returns no meaningful result, inform the user instead of generating facts.
- Output combination of reponses from both music_database_agent response and final interesting facts in the below format.

Response Format :
`
<response from music_database_agent>.

Here are 5 Interesting facts about <topic> :
<response from interesting_facts_generator>
`
"""

superviser_agent = create_agent(
    model=llm,
    tools=[interesting_facts_generator, music_database_agent],
    system_prompt=SUPERVISER_SYSTEM_MESSAGE
)

if __name__ == "__main__" :

    while True:

        print("Ask any question related to chinook database and get few interseting facts regarding the question.")

        user_prompt = input("User Query : ")

        for step in superviser_agent.stream({"messages": user_prompt}, stream_mode="values") :

            print("--"*15)
            if "messages" in step :
                current_message = step["messages"][-1]
                if type(current_message) == AIMessage :
                    print(f"== AI == : {current_message.content}")
                    if len(current_message.tool_calls) > 0 :
                        for tool in current_message.tool_calls :
                            print(f"Tool Call Request || Name : {tool["name"]} || Args : {tool["args"]}")
                elif type(current_message) == ToolMessage :
                    print(f"== Tool == : {current_message.content}")
                elif type(current_message) == HumanMessage :
                    print(f"== Human == : {current_message.content}")
