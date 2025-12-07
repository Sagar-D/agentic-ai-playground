import kagglehub
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from helpers.llm_manager import get_llm_instance
import os
import dotenv

dotenv.load_dotenv()

# Download latest version of dataset
kaggle_dataset_id = "lamskdna/video-games-sales"
dataset_path = kagglehub.dataset_download(kaggle_dataset_id, path="VideoGames_Sales.xlsx")

df = pd.read_excel(dataset_path)

llm = get_llm_instance("ollama")

pd_agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_code=os.getenv("ALLOW_REPL_EXECUTION")
)

if __name__ == "__main__" :

    print("--"*30)
    df_summary_resp = pd_agent.invoke("""\n\nSummarize the dataframe that has been loaded including 
- a brief on dataset
- list of columns (with datatype)""")
    
    print("""You are interacting with a Data Analysis bot. Below is the summary of the data file the bot has access to.
Ask any queries related to this data so that bot can help you out.
          
Example queries : 
- How many entries ar there in the dataset?
- Which is the most sold game in the data set?
- Show a graph plot of highest revenue collected by any game for every year.

Below is the summary of the dataset : \n\n""")


    print(df_summary_resp['output'])
    print("--"*30)

    while True :

        user_query = input("USER QUERY : ")
        agent_response = pd_agent.invoke(user_query)
        print(f"\nAGENT RESPONSE : {agent_response['output']}")