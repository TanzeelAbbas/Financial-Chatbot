from database import get_db_tools
from config import OPENAI_API_KEY
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
import os


def main():
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

    llm = OpenAI(model="gpt-4")
    
    db_tools = get_db_tools()
    
    agent = OpenAIAgent.from_tools(db_tools.to_tool_list(), llm=llm, verbose=False)

    while True:
        query = input("Enter your query (or 'q' to quit): ")
        if query.lower() == 'q':
            break
        answer = agent.chat(query)
        print("Answer:", answer.response)

if __name__ == "__main__":
    main()
