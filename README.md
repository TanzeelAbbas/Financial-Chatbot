# BrokerBotics Chatbot

## Requirements

-  Python 3.8+
- `llama_index` library
- `langchain_community` library
- `langchain_openai` library
- `psycopg2-binary` library
- 'crewai'  library
- 'crewai_tools' library 
- OpenAI API key
- Database credentials
## Installation

1. Clone the repository:

    ```
    github repo url
    ```

2. Install the required libraries:

    ``` terminal
    cd backend
    pip install -r requirements.txt
    ```

## Usage

1. Run the main scripts:

    ``` terminal
    python method1_llamaindex.py
    python method2_langchain.py
    multi_agents_approach.py
    
    ```

2. Enter your query in the console. Type 'q' to quit the application.

## File Structure

- `method1_llamaindex.py`: The main script to run the chatbot with `llama_index`.
- `method2_langchain.py`: The main script to run the chatbot with `langchain_community`.
- 'multi_agents_approach.py'
- `database.py`: Contains the `get_db_tools` function for `llama_index` postrgres connection .
- `config.py`: Contains the `OPENAI_API_KEY` and database credentials.








