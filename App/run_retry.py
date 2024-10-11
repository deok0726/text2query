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

# logging.basicConfig(level=logging.DEBUG)
dotenv_path = os.path.join(os.path.dirname(__file__), '../config', '.env')
load_dotenv(dotenv_path)
api_key = os.getenv("OPENAI_API_KEY")

def generate_natural_language_answer(llm, question, sql_query, sql_result):
    # 프롬프트 템플릿 정의
    answer_prompt_template = '''
    Given the following user question, corresponding SQL query, and SQL result, answer the user question in natural language in Korean.

    Question: {question}
    SQL Query: {sql_query}
    SQL Result: {sql_result}

    Answer: '''
    
    answer_prompt = answer_prompt_template.format(question=question, sql_query=sql_query, sql_result=sql_result)
    response = llm.invoke(answer_prompt)
    
    return response

def sql_result(llm, db, question):
    few_shot_examples = '''
        Example 1)
        Question: LG전자에서 9월 동안 받은 혜택 금액을 알려줘
        SQLQuery: SELECT SUM(T1.SL_AM)-SUM(BIL_PRN) AS BENEFIT FROM WBM_V_BLL_SPEC_IZ AS T1 INNER JOIN LM_TB_MC AS T2 ON T1.DAF_MCNO = T2.MCNO WHERE T2.MC_NM = 'LG전자' AND SUBSTR(T1.SL_DT, 5, 2) = '09';
        SQLResult: [(2301000,)]
        Answer: 9월 한 달 동안 LG전자에서 받은 혜택 금액은 2,301,000 원입니다.
    '''

    template = '''Given an input question, create a syntactically correct top {top_k} {dialect} query to run which should end with a semicolon.
        Use the following format:

        Question: "Question here"
        SQLQuery: "SQL Query to run"
        SQLResult: "Result of the SQLQuery"
        Answer: "Final answer here"

        Only use the following tables:

        {table_info}.

        Here are some examples of previous questions and their corresponding SQL queries to help you understand the expected format:

        {few_shot_examples}

        Please generate a new SQL query based on the following question without referencing the previous examples directly:

        Question: {input}'''
    
    prompt = PromptTemplate.from_template(template)
    query_chain = create_sql_query_chain(llm, db, prompt=prompt)
    # generated_sql_query = query_chain.invoke({"table_info": db.get_table_info(), "input": question, "dialect": db.dialect, "top_k": 1})
    generated_sql_query = query_chain.invoke({
        "table_info": db.get_table_info(), 
        "input": question, 
        "dialect": db.dialect, 
        "top_k": 1, 
        "few_shot_examples": few_shot_examples})
    print("SQL query: ", generated_sql_query.__repr__())
    
    execute = QuerySQLDataBaseTool(db=db)
    
    fix = 1
    retry = 4
    final = None
    result = execute.invoke({"query": generated_sql_query})
    while (fix < retry) :
        if "OperationalError" in result:
            print(f"Try {fix}...")
            print(f"### Error executing SQL query ### \n{result}")
            error_message = str(result)

            fixed_template = '''
                The following SQL query resulted in an error: {sql_query}

                Error: {error_message}

                Based on the provided table and column information, please correct the SQL query.

                {table_info}

                Question: {input}

                Only reproduce corrected top {top_k} SQL query. Generate an SQL query without any additional formatting. The query should start directly with SELECT. 
                
                SQLQuery: '''

            prompt = PromptTemplate.from_template(fixed_template)
            query_chain = create_sql_query_chain(llm, db, prompt=prompt)
            corrected_sql_query = query_chain.invoke({
                "sql_query": result, 
                "table_info": db.get_table_info(), 
                "input": question, 
                "top_k": 1, 
                "error_message": error_message})
            
            result = execute.invoke({"query": corrected_sql_query})
            # print("-"*200)
            # print(result)

            # answer = generate_natural_language_answer(llm, question, corrected_sql_query, result)
            # print(answer)
        else:
            final = result
            print("-"*200)
            print("SQL query: ", corrected_sql_query)
            print("SQL result: ", result)

            answer = generate_natural_language_answer(llm, question, generated_sql_query, result)
            print(answer)
            break
        fix += 1
    
    if final is None:
        print("##### SQL Execution Error No Result #####")

if __name__ == "__main__":
    # db_name = input("Input DB name: ")
    # llm = Ollama(model="llama3.1:latest", temperature=0)
    # llm = Ollama(model="llama3.1:70b", temperature=0)
    # llm = Ollama(model="codellama:70b", temperature=0)
    llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=None, openai_api_key=api_key)

    # db = SQLDatabase.from_uri(f"sqlite:///{db_name}.db")
    db = SQLDatabase.from_uri("sqlite:///app.db")
    print(db.dialect)

    print(db.get_usable_table_names())
    # print(db.get_table_info())
    print("-"*200)
    # question = input("DB 질문을 입력하세요: ")
    # question = "최근 두 달 동안 LG전자에서 받은 혜택 금액을 알려줘"
    # question = "지난 달 가장 돈을 많이 쓴 매장을 알려줘"
    question = "10월에 스타벅스를 몇 번 방문했는지 알려줘"

    sql_result(llm, db, question)
