import os
import sqlite3
import logging
from openai import OpenAI
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.chains import create_sql_query_chain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

logging.basicConfig(level=logging.DEBUG)
dotenv_path = os.path.join(os.path.dirname(__file__), '../config', '.env')
print(dotenv_path)
load_dotenv(dotenv_path)
api_key = os.getenv("OPENAI_API_KEY")

def generate_natural_language_answer(llm, question, sql_query, sql_result):
    answer_prompt_template = '''
    Given the following user question, corresponding SQL query, and SQL result, answer the user question in natural language in Korean.

    Question: {question}
    SQL Query: {sql_query}
    SQL Result: {sql_result}

    Answer: '''
    
    answer_prompt = answer_prompt_template.format(question=question, sql_query=sql_query, sql_result=sql_result)
    response = llm.invoke(answer_prompt)
        
    return response

def generate_sql_with_synonyms(llm, db, db_path, question):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%META%';")
    tables = cursor.fetchall()

    # table_and_column_info = "\n".join([
    #     f"{col[1]} (Code: {col[0]}), Synonyms: {col[2] if col[2] else 'None'}"
    #     for col in tables
    # ])

    table_and_column_info = []
    for table in tables:
        table_name = table[0]
        # print(f"Table: {table_name}")
        
        # SQL 쿼리 생성 및 실행
        query = f"SELECT * FROM {table_name}"
        try:
            cursor.execute(query)
            rows = cursor.fetchall()
            
            for row in rows:
                if row[2]:
                    table_and_column_info.append(f"{table_name} {row[1]} (Code: {row[0]}), Synonyms: {row[2]}")
                    # print(f"{row[1]} (Code: {row[0]}), Synonyms: {row[2]}")
                # else:
                #     table_and_column_info.append(f"{row[1]} (Code: {row[0]})")
                    # print(f"{row[1]} (Code: {row[0]})")
        except sqlite3.OperationalError as e:
            print(f"Error reading from table {table_name}: {e}")
        
        # print("-" * 50)

    table_and_column_info_txt = "\n".join(table_and_column_info)
    print(table_and_column_info_txt)
    few_shot_examples = '''
        Example 1)
        Question: 24.06.04~24.06.30 기간 이벤트 응모
        SQLQuery: "STDT BETWEEN '20240604' AND '20240630'"

        Example 2)
        Question: 10000원 이상 사용
        SQLQuery: "SL_AM > 10000"

    '''

    template = '''Given an input question, create a syntactically correct top {top_k} {dialect} query to run which should end with a semicolon.
        Use the following format:

        Question: "Question here"
        SQLQuery: "SQL Query to run"
        SQLResult: "Result of the SQLQuery"
        Answer: "Final answer here"

        Only use the following tables:

        {table_info}.

        The following columns have the following synonyms:
        
        {table_and_column_info}.

        Question: {input}'''
    
    prompt = PromptTemplate.from_template(template)
    query_chain = create_sql_query_chain(llm, db, prompt=prompt)
    generated_sql_query = query_chain.invoke({"table_info": db.get_table_info(), "table_and_column_info": table_and_column_info_txt, "input": question, "dialect": db.dialect, "top_k": 1})

    return generated_sql_query

def sql_result_with_synonyms(llm, db, db_path, question):
    generated_sql_query = generate_sql_with_synonyms(llm, db, db_path, question)
    
    print("Generated SQL Query:", generated_sql_query)

    # SQL 쿼리 실행 및 결과 확인
    execute = QuerySQLDataBaseTool(db=db)
    try:
        result = execute.invoke({"query": generated_sql_query})
        print("-" * 200)
        print(result)

        # 자연어로 결과를 반환
        answer = generate_natural_language_answer(llm, question, generated_sql_query, result)
        print(answer)

    except Exception as e:
        print("-" * 200)
        print(f"Error: {e}")
        print(generated_sql_query.__repr__())


if __name__ == "__main__":
    db_name = input("Input DB name: ")
    llm = Ollama(model="llama3.1:70b", temperature=0)

    db = SQLDatabase.from_uri(f"sqlite:///{db_name}.db")
    db_path = f"{db_name}.db"
    print(db.dialect)
    print(db.get_usable_table_names())
    print("-"*200)
    question = input("DB 질문을 입력하세요: ")

    sql_result_with_synonyms(llm, db, db_path, question)
