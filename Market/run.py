import os
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
        Question: LG전자에서 사용한 총 금액을 알려줘
        SQLQuery: "SELECT SUM(APR_DE_AM) FROM LP_TB_TBAUTH AS T1 INNER JOIN LM_TB_MC AS T2 ON T1.MCNO = T2.MCNO WHERE T2.MC_NM = 'LG전자';"
        SQLResult: 51000000
        Answer: LG전자에서 사용한 총 금액은 51000000 원 입니다.

        Example 2)
        Question: 지금까지 치킨을 몇 번 시켜먹었는지 알려줘
        SQLQuery: "SELECT COUNT(*) FROM LP_TB_TBAUTH AS T1 INNER JOIN LM_TB_MC AS T2 ON T1.MCNO = T2.MCNO WHERE T2.MC_NM LIKE '%치킨%';"
        SQLResult: 3
        Answer: 지금까지 치킨을 5 번 시켜먹었습니다.
    '''

    template = '''Given an input question, create a syntactically correct top {top_k} {dialect} query to run which should end with a semicolon.
        Use the following format:

        Question: "Question here"
        SQLQuery: "SQL Query to run"
        SQLResult: "Result of the SQLQuery"
        Answer: "Final answer here"

        Only use the following tables:

        {table_info}.

        Question: {input}'''
    prompt = PromptTemplate.from_template(template)

    query_chain = create_sql_query_chain(llm, db, prompt=prompt)
    generated_sql_query = query_chain.invoke({"table_info": db.get_table_info(), "input": question, "dialect": db.dialect, "top_k": 1})

    print(generated_sql_query.__repr__())
    execute = QuerySQLDataBaseTool(db=db)
    try:
        result = execute.invoke({"query": generated_sql_query})
        print("-"*200)
        print(result)

        answer = generate_natural_language_answer(llm, question, generated_sql_query, result)
        print(answer)

        # result = execute.invoke({"query": "SELECT T1.MC_NM, T2.APR_DTTI, T2.CDNO, T2.DE_DT, T2.CNO, T2.APR_DE_AM FROM LM_TB_MC AS T1 INNER JOIN LP_TB_TBAUTH AS T2 ON T1.MCNO = T2.MCNO WHERE T1.MC_NM = '스타벅스'"})
    except Exception as e:
       print("-"*200)
       print(f"Error: {e}")
       print(generated_sql_query.__repr__())


if __name__ == "__main__":
    db_name = input("Input DB name: ")
    llm = Ollama(model="llama3.1:70b", temperature=0)
    # llm = Ollama(model="codellama:70b", temperature=0)
    # llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=None, openai_api_key=api_key)

    db = SQLDatabase.from_uri(f"sqlite:///{db_name}.db")
    print(db.dialect)
    print(db.get_usable_table_names())
    # print(db.get_table_info())
    print("-"*200)
    question = input("DB 질문을 입력하세요: ")
    # question = "스타벅스에서 결제한 전체 내역을 보여줘"

    sql_result(llm, db, question)
