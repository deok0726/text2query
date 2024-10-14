import os, sys
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
        Question: 24. 7~9월까지 '70018819695' 고객이 스타벅스에서 얼마를 썼고, 보유카드 중에 할인을 받을 수 있는 상품이 있다면 얼마를 할인 받았고, 할인 한도가 얼마나 남았는지 알려줘
        SQLQuery: WITH monthly_discounts AS (
                    SELECT 
                        strftime('%Y-%m', SL_DT) AS month,
                        SUM(BLL_SV_AM) AS total_discount
                    FROM 
                        WBM_T_BLL_SPEC_IZ
                    WHERE 
                        ACCTNO = '70018819695'
                        AND SL_DT BETWEEN '20240701' AND '20240930'
                        AND BLL_MC_NM LIKE '%스타벅스%'
                        AND BLL_SV_DC IN (SELECT SV_C FROM WPD_T_SV_SNGL_PRP_INF)
                    GROUP BY 
                        strftime('%Y-%m', SL_DT)
                ),
                total_spent AS (
                    SELECT 
                        SUM(COALESCE(BIL_PRN, SL_AM)) AS total_spent
                    FROM 
                        WBM_T_BLL_SPEC_IZ
                    WHERE 
                        ACCTNO = '70018819695'
                        AND SL_DT BETWEEN '20240701' AND '20240930'
                        AND BLL_MC_NM LIKE '%스타벅스%'
                ),
                total_discounted AS (
                    SELECT 
                        SUM(BLL_SV_AM) AS total_discounted
                    FROM 
                        WBM_T_BLL_SPEC_IZ
                    WHERE 
                        ACCTNO = '70018819695'
                        AND SL_DT BETWEEN '20240701' AND '20240930'
                        AND BLL_MC_NM LIKE '%스타벅스%'
                        AND BLL_SV_DC IN (SELECT SV_C FROM WPD_T_SV_SNGL_PRP_INF)
                )
                SELECT 
                    (SELECT total_spent FROM total_spent) AS total_spent,
                    (SELECT total_discounted FROM total_discounted) AS total_discounted,
                    (SELECT MLIM_AM FROM WPD_T_SV_SNGL_PRP_INF WHERE SV_C = 'SP03608') - COALESCE((SELECT total_discount FROM monthly_discounts WHERE month = strftime('%Y-%m', 'now')), 0) AS remaining_discount_limit;

        SQLResult: [(101660, 16040, 10000)]
        Answer: 고객 '70018819695'는 7월부터 9월까지 스타벅스에서 총 101,660원을 사용했습니다. 보유한 카드 중 할인 가능한 상품을 통해 16,040원의 할인을 받았으며, 현재 할인 한도는 10,000원이 남았습니다.
    '''

    few_shots = '''
        Example 1)
        Question: 24. 7~9월까지 'A' 고객이 OOO에서 얼마를 썼고, 보유카드 중에 할인을 받을 수 있는 상품이 있다면 얼마를 할인 받았고, 이번 달 할인 한도가 얼마나 남았는지 알려줘
        SQLQuery: WITH monthly_discounts AS (
                        SELECT strftime(
                                '%Y-%m',
                                substr(SL_DT, 1, 4) || '-' || substr(SL_DT, 5, 2) || '-' || substr(SL_DT, 7, 2)
                            ) AS month,
                            SUM(BLL_SV_AM) AS total_discount
                        FROM WBM_T_BLL_SPEC_IZ
                        WHERE ACCTNO = 'A'
                            AND SL_DT BETWEEN strftime('%Y%m%d', date('now', 'start of month')) AND strftime('%Y%m%d', date('now', 'start of month', '+1 month', '-1 day'))
                            AND BLL_MC_NM LIKE '%OOO%'
                            AND BLL_SV_DC IN (
                                SELECT SV_C
                                FROM WPD_T_SV_SNGL_PRP_INF
                            )
                        GROUP BY strftime(
                                '%Y-%m',
                                substr(SL_DT, 1, 4) || '-' || substr(SL_DT, 5, 2) || '-' || substr(SL_DT, 7, 2)
                            )
                    ),
                    total_spent AS (
                        SELECT SUM(COALESCE(BIL_PRN, SL_AM)) AS total_spent
                        FROM WBM_T_BLL_SPEC_IZ
                        WHERE ACCTNO = 'A'
                            AND SL_DT BETWEEN '20240701' AND '20240930'
                            AND BLL_MC_NM LIKE '%OOO%'
                    ),
                    total_discounted AS (
                        SELECT SUM(BLL_SV_AM) AS total_discounted
                        FROM WBM_T_BLL_SPEC_IZ
                        WHERE ACCTNO = 'A'
                            AND SL_DT BETWEEN '20240701' AND '20240930'
                            AND BLL_MC_NM LIKE '%OOO%'
                            AND BLL_SV_DC IN (
                                SELECT SV_C
                                FROM WPD_T_SV_SNGL_PRP_INF
                            )
                    )
                    SELECT (
                            SELECT total_spent
                            FROM total_spent
                        ) AS total_spent,
                        (
                            SELECT total_discounted
                            FROM total_discounted
                        ) AS total_discounted,
                        (
                            SELECT MLIM_AM
                            FROM WPD_T_SV_SNGL_PRP_INF
                            WHERE SV_C = 'SP03608'
                        ) - COALESCE(
                            (
                                SELECT total_discount
                                FROM monthly_discounts
                                WHERE month = strftime('%Y-%m', 'now')
                            ),
                            0
                        ) AS remaining_discount_limit;

        SQLResult: [(101660, 16040, 4540)]
        Answer: 고객 'A'는 7월부터 9월까지 OOO에서 총 101,660원을 사용했습니다. 보유한 카드 중 할인 가능한 상품을 통해 16,040원의 할인을 받았으며, 이번 달 할인 한도는 4,540원이 남았습니다.
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
        "few_shot_examples": few_shots})
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
    db = SQLDatabase.from_uri("sqlite:///data_v2.db")
    print(db.dialect)

    print(db.get_usable_table_names())
    # print(db.get_table_info())
    print("-"*200)
    # question = input("DB 질문을 입력하세요: ")
    # question = "24. 7~9월까지 '70018819695' 고객이 스타벅스에서 얼마를 썼고, 보유카드 중에 할인을 받을 수 있는 상품이 있다면 얼마를 할인 받았고, 이번 달 할인 한도가 얼마나 남았는지 알려줘"
    question = "24. 7~9월까지 '70018819695' 고객이 기간 동안 스타벅스에서 사용한 총액과 결제일 별 결제금액을 알려줘. 그리고 보유카드 중에 할인을 받을 수 있는 상품이 있다면 얼마를 할인 받았고, 이번 달 할인 한도가 얼마나 남았는지 알려줘"

    sql_result(llm, db, question)