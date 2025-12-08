from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain.agents.factory import create_agent
from helpers.llm_manager import get_llm_instance
from helpers.sql_db_setup import setup_sqlite_db, CHINOOK_DB_URL
from langchain.messages import AIMessage, HumanMessage, ToolMessage

setup_sqlite_db()
db = SQLDatabase.from_uri(CHINOOK_DB_URL)
llm = get_llm_instance("ollama")
sql_tool_kit = SQLDatabaseToolkit(llm=llm, db=db)

print(f"Available tables: {db.get_usable_table_names()}")

SYSTEM_PROMPT = """\
You are a database query assistant.

Your responsibilities:
- You ONLY interact with the database through the provided tools.
- You perform READ-ONLY queries. Never modify data.
- You must translate the user’s request into the appropriate tool call.
- After receiving the tool response, answer the user’s question clearly and concisely.
- Do NOT invent information that is not returned by the tool.
- If the requested data does not exist, say so.

Always use the tools for factual answers."""

sql_agent = create_agent(
    model=llm,
    tools=sql_tool_kit.get_tools(),
    system_prompt=SYSTEM_PROMPT
)

if __name__ == "__main__" :

    while True :

        print("--"*30)
        user_prompt = input("USER QUERY : ")
        print("--"*30)
        for step in sql_agent.stream({"messages":user_prompt}, stream_mode='values') :
            
            if 'messages' in step and len(step['messages']) > 0 :
                current_message = step['messages'][-1]
                
                if type(current_message) == AIMessage :
                    print(f"AI Message : {current_message.content}")
                    if len(current_message.tool_calls) > 0 :
                        print(f"Request for Tool call : ")
                        for tool in current_message.tool_calls :
                            print(f"=> Tool Name : {tool['name']} || Args : {tool['args']}")
                elif type(current_message) == ToolMessage:
                    print(f"Tool Message : {current_message.content}")
                else :
                    print(f"Human Message : {current_message.content}")



