import os
import sys
import io
from textwrap import dedent
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
from crewai_tools import tool
from langchain_openai import ChatOpenAI
from crewai import Agent, Crew, Process, Task
from config import OPENAI_API_KEY, DB_HOST, DB_NAME, DB_USER, DB_PASS, DB_PORT


# Database and LLM setup
def setup_environment():
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    db = SQLDatabase.from_uri(f'postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
    llm = ChatOpenAI(model='gpt-4', temperature=0)
    return db, llm

# Agent creation
def create_agents(llm, db):
    @tool("list_tables")
    def list_tables() -> str:
        """List the available tables in the database"""
        return ListSQLDatabaseTool(db=db).invoke("")

    @tool("tables_schema")
    def tables_schema(tables: str) -> str:
        """
        Input is a comma-separated list of tables, output is the schema and sample rows
        for those tables. Be sure that the tables actually exist by calling `list_tables` first!
        Example Input: table1, table2, table3
        """
        tool = InfoSQLDatabaseTool(db=db)
        return tool.invoke(tables)

    @tool("execute_sql")
    def execute_sql(sql_query: str) -> str:
        """Execute a SQL query against the database. Returns the result"""
        return QuerySQLDataBaseTool(db=db).invoke(sql_query)

    @tool("check_sql")
    def check_sql(sql_query: str) -> str:
        """
        Use this tool to double check if your query is correct before executing it. Always use this
        tool before executing a query with `execute_sql`.
        """
        return QuerySQLCheckerTool(db=db, llm=llm).invoke({"query": sql_query})

    sql_dev = Agent(
        role="Senior Database Developer",
        goal="Construct and execute SQL queries based on a request",
        backstory=dedent(
            """
            You are an experienced database engineer who is master at creating efficient and complex SQL queries.
            You have a deep understanding of how different databases work and how to optimize queries.
            Use the `list_tables` to find available tables.
            Use the `tables_schema` to understand the metadata for the tables.
            Use the `execute_sql` to execute queries against the database.
            Use the `check_sql` to check your queries for correctness before execution.
        """
        ),
        llm=llm,
        tools=[list_tables, tables_schema, execute_sql, check_sql],
        allow_delegation=False,
    )

    data_analyst = Agent(
        role="Senior Financial Analyst/Data Analyst",
        goal="You receive data from the database developer and analyze it",
        backstory=dedent(
            """
            You have deep experience with analyzing datasets using Python.
            Your work is always based on the provided data and is clear,
            easy-to-understand and to the point. You have attention
            to detail and always produce very detailed work (as long as you need).
        """
        ),
        llm=llm,
        allow_delegation=False,
    )

    report_writer = Agent(
        role="Senior Report Editor",
        goal="Write an executive summary type of report based on the work of the analyst",
        backstory=dedent(
            """
            Your writing style is well known for clear and effective communication.
            You always summarize long texts into bullet points that contain the most
            important details.
            """
        ),
        llm=llm,
        allow_delegation=False,
    )
    
    
    data_analyst = Agent(
        role="Senior Financial Analyst/Data Analyst",
        goal="You receive data from the database developer and analyze it",
        backstory=dedent(
            """
            You have deep experience with analyzing datasets using Python.
            Your work is always based on the provided data and is clear,
            easy-to-understand and to the point. You have attention
            to detail and always produce very detailed work (as long as you need).
        """
        ),
        llm=llm,
        allow_delegation=False,
    )

    report_writer = Agent(
        role="Senior Report Editor",
        goal="Write an executive summary type of report based on the work of the analyst",
        backstory=dedent(
            """
            Your writing still is well known for clear and effective communication.
            You always summarize long texts into bullet points that contain the most
            important details.
            """
        ),
        llm=llm,
        allow_delegation=False,
    )

    return sql_dev, data_analyst, report_writer

# Task creation
def create_tasks(sql_dev, data_analyst, report_writer):
    extract_data = Task(
        description="Extract data that is required for the query {query}.",
        expected_output="Database result for the query",
        agent=sql_dev,
    )

    analyze_data = Task(
        description="Analyze the data from the database and write an analysis for {query}.",
        expected_output="Detailed analysis text",
        agent=data_analyst,
        context=[extract_data],
    )

    write_report = Task(
        description=dedent(
            """
            Write an executive summary of the report from the analysis. The report
            must be less than 100 words.
        """
        ),
        expected_output="Markdown report",
        agent=report_writer,
        context=[analyze_data],
    )

    return extract_data, analyze_data, write_report

# Crew creation and execution
def run_crew(agents, tasks, query):
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=0,
        memory=False,
        output_log_file="crew.log",
    )

    inputs = {"query": query}

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        result = crew.kickoff(inputs=inputs)
    finally:
        # Restore stdout
        sys.stdout = old_stdout

    return result

def main():
    db, llm = setup_environment()
    sql_dev, data_analyst, report_writer = create_agents(llm, db)
    extract_data, analyze_data, write_report = create_tasks(sql_dev, data_analyst, report_writer)
    
    agents = [sql_dev, data_analyst, report_writer]
    tasks = [extract_data, analyze_data, write_report]
    
    while True:
        query = input("Enter your query (or 'q' to quit): ")
        if query.lower() == 'q':
            print("Exiting the program. Goodbye!")
            break
        
        result = run_crew(agents, tasks, query)
        print("\nResult:")
        print(result)
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()