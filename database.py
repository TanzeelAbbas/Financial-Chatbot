from llama_index.tools.database import DatabaseToolSpec
from config import DB_HOST, DB_NAME, DB_USER, DB_PASS, DB_PORT

def get_db_tools():
    db_tools = DatabaseToolSpec(
    		scheme="postgresql",  # Database Scheme
    		host=DB_HOST,  # Database Host
    		port=DB_PORT,  # Database Port
    		user=DB_USER,  # Database User
    		password=DB_PASS,  # Database Password
    		dbname=DB_NAME,  # Database Name
    		)
    
    return db_tools

