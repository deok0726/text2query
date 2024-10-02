from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType
import sqlite3
import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    db = input("Input DB name: ")
    llm = Ollama(model="llama3.1:70b")
    # llm = Ollama(model="codellama:70b")
    
    db = SQLDatabase.from_uri(f"sqlite:///{db}.db")
    print(db.dialect)
    print(db.get_usable_table_names())
    print(db.table_info)
    print("-"*200)

    agent_executor = create_sql_agent(llm, db=db, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True, verbose=True)
    question = input("DB 질문을 입력하세요: ")
    
    try:
        result = agent_executor.invoke(question)
        print("LLM Output:", result)
    except ValueError as e:
        print(f"An error occurred: {e}")
        logging.error("LLM output parsing error", exc_info=True)