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
    few_shots = '''
        Example 1)
        Question: 'A' 고객이 2024년 1월부터 3월까지 OOO에서 사용한 총액을 알려줘
        SQLQuery: 
            SELECT SUM(COALESCE(BIL_PRN, SL_AM)) AS total_spent
                FROM WBM_T_BLL_SPEC_IZ
                WHERE ACCTNO = 'A'
                AND SL_DT BETWEEN '20240101' AND '20240331'
                AND BLL_MC_NM LIKE '%OOO%';
        SQL result: 101660
        Answer: '2024년 1월부터 3월까지 'A' 고객이 OOO에서 사용한 총액은 101,660원 입니다.'

        Example 2)
        Question: 'A' 고객이 2024년 1월부터 3월까지 사용한 월 별 결제금액을 알려줘
        SQLQuery: 
            SELECT strftime('%Y-%m', substr(SL_DT, 1, 4) || '-' || substr(SL_DT, 5, 2) || '-' || substr(SL_DT, 7, 2)) AS month,
                SUM(COALESCE(BIL_PRN, SL_AM)) AS total_spent
            FROM WBM_T_BLL_SPEC_IZ
            WHERE ACCTNO = 'A'
            AND SL_DT BETWEEN '20240101' AND '20240331'
            AND BLL_MC_NM LIKE '%OOO%'
            GROUP BY month;
        SQL result: [{"month":"2024-01","total_spent":14000},{"month":"2024-02","total_spent":43200},{"month":"2024-03","total_spent":44460}]
        Answer: '2024년 1월부터 3월까지 'A' 고객이 OOO에서 사용한 금액은 다음과 같습니다: 7월에는 14,000원, 8월에는 43,200원, 9월에는 44,460원입니다.

        Example 3)
        Question: 'A' 고객의 이번 달 남은 할인 한도를 알려줘
        SQLQuery: 
            SELECT 
                (SELECT MLIM_AM
                FROM WPD_T_SV_SNGL_PRP_INF
                WHERE SV_C = 'SP03608') - 
                COALESCE(
                    (SELECT SUM(BLL_SV_AM) AS total_discount
                    FROM WBM_T_BLL_SPEC_IZ
                    WHERE ACCTNO = 'A'
                    AND SL_DT BETWEEN strftime('%Y%m%d', date('now', 'start of month')) AND strftime('%Y%m%d', date('now', 'start of month', '+1 month', '-1 day'))
                    AND BLL_MC_NM LIKE '%OOO%'
                    AND BLL_SV_DC IN (
                        SELECT SV_C
                        FROM WPD_T_SV_SNGL_PRP_INF
                    )
                    ), 0
                ) AS remaining_discount_limit;
        SQL result: 4540
        Answer: 'A' 고객의 이번 달 남은 할인 한도는 4,540원 입니다.'

        Example 4)
        Question: OOO와 관련해서 추천해줄만한 이벤트를 알려줘
        SQLQuery: 
            SELECT EVN_BULT_TIT_NM
                FROM WLP_T_EVN_INF
                WHERE EVN_BULT_TIT_NM LIKE '%OOO%'
                AND EVN_SDT <= strftime('%Y%m%d', 'now')
                AND EVN_EDT >= strftime('%Y%m%d', 'now');
        SQL result: ["안심쇼핑 신청하면 OOO 쿠폰 증정","5월 국세·지방세 납부하고 OOO 커피쿠폰 받자"]
        Answer: 'OOO와 관련된 추천 이벤트로는 "안심쇼핑 신청하면 OOO 쿠폰 증정"과 "5월 국세·지방세 납부하고 OOO 커피쿠폰 받자"가 있습니다.'
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
            print("-"*100)
            print("SQL query: ", corrected_sql_query)
            print("SQL result: ", result)

            answer = generate_natural_language_answer(llm, question, generated_sql_query, result)
            print(answer)
            break
        fix += 1
    
    if final is None:
        print("##### SQL Execution Error No Result #####")

if __name__ == "__main__":
    # llm = Ollama(model="llama3.1:70b", temperature=0)
    llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=None, openai_api_key=api_key)
    db = SQLDatabase.from_uri("sqlite:///data_v2_1.db")
    print(db.dialect)

    print(db.get_usable_table_names())
    print("-"*200)
    # question = input("DB 질문을 입력하세요: ")
    # question = "24. 7~9월까지 '70018819695' 고객이 스타벅스에서 얼마를 썼고, 보유카드 중에 할인을 받을 수 있는 상품이 있다면 얼마를 할인 받았고, 이번 달 할인 한도가 얼마나 남았는지 알려줘"
    # question = "24. 7~9월까지 '70018819695' 고객이 기간 동안 스타벅스에서 사용한 총액과 결제일 별 결제금액을 알려줘. 그리고 보유카드 중에 할인 받은 상품이 있다면 얼마를 할인 받았는지, 이번 달 할인 한도가 얼마나 남았는지도 알려줘"
    # question = "24. 7~10월까지 '70018819695' 고객이 사용한 금액을 카드 별로 알려줘"
    # question = "24. 7~9월까지 '70018819695' 고객이 통신 요금으로 납부한 금액과 할인 받은 금액을 알려줘"
    # question = "'70018819695' 고객이 결제한 전체 카드 별로 각각 9월 달에 받은 혜택 금액과 이번 달 잔여 한도를 알려줘"
    
    # question = "24. 7~9월까지 '70018819695' 고객이 기간 동안 스타벅스에서 사용한 총액과 결제월 별 결제금액을 알려줘."
    question = "24. 7~9월까지 '70018819695' 고객이 스타벅스 할인을 받았다면 할인 금액을 월 별로 알려주고, 이번 달 할인 한도가 얼마나 남았는지도 알려줘."
    # question = "24. 7~9월까지 '70018819695' 고객에게 스타벅스와 관련해서 추천해줄만한 이벤트가 있다면 알려줘."

    sql_result(llm, db, question)