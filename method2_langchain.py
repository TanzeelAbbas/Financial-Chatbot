from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import InfoSQLDatabaseTool
from langchain_openai import ChatOpenAI
import os
from config import OPENAI_API_KEY, DB_HOST, DB_NAME, DB_USER, DB_PASS, DB_PORT

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

def run_sql_agent():
    # Connect to the database
    db = SQLDatabase.from_uri(f'postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

    llm = ChatOpenAI(model_name="gpt-4", temperature=0)

    # Create the SQL toolkit
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # Create the SQL agent
    tool_names = [tool.name for tool in toolkit.get_tools()]
    agent_executor = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=False,
            handle_parsing_errors=True,
            tool_names=tool_names,
            prefix= """  """
        )
    
    while True:
        query = input("Enter your query (or 'q' to quit): ")
        if query.lower() == 'q':
            break
        if not query.strip():
            print("Please enter a valid query.")
            continue
        
        try:
            response = agent_executor(query)
            
            print("\nResponse:")
            print(response['output'])
            
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try rephrasing your query or check if the required data is available in the database.")

if __name__ == "__main__":
    run_sql_agent()