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

sql_agent = create_agent(
    model=llm,
    tools=sql_tool_kit.get_tools(),
    system_prompt="""You are a helpful database query assistant.
You perform read only operations on datbase using the tools provided and answer user queries."""
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



