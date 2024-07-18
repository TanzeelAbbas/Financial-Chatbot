import os

DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT= os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'BrokerBotics')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASS = os.getenv('DB_PASS', 'password')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', "OPENAI_API_KEY...")