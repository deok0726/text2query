from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType
import sqlite3
import logging

logging.basicConfig(level=logging.DEBUG)

def find_table_with_max_rows(conn):
    """가장 많은 행을 가진 테이블을 찾습니다."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    table_names = [table[0] for table in tables]

    max_rows = 0
    max_table = None

    for table_name in table_names:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        row_count = cursor.fetchone()[0]
        print(f"Table {table_name} has {row_count} rows.")
        if row_count > max_rows:
            max_rows = row_count
            max_table = table_name

    return max_table, max_rows

if __name__ == "__main__":
    db = input("Input DB name: ")
    llm = Ollama(model="llama3.1:70b")
    conn = sqlite3.connect(f'{db}.db')
    max_table, max_rows = find_table_with_max_rows(conn)

    db = SQLDatabase.from_uri(f"sqlite:///{db}.db", sample_rows_in_table_info = 10)
    print(db.get_usable_table_names())
    print(db.table_info)

    agent_executor = create_sql_agent(llm, db=db, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True, verbose=True)
    question = input("DB 질문을 입력하세요: ")
    
    try:
        result = agent_executor.invoke(question)
        print("LLM Output:", result)
    except ValueError as e:
        print(f"An error occurred: {e}")
        logging.error("LLM output parsing error", exc_info=True)