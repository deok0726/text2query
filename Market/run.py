import os, torch
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
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from dotenv import load_dotenv

# logging.basicConfig(level=logging.DEBUG)
dotenv_path = os.path.join(os.path.dirname(__file__), '../config', '.env')
load_dotenv(dotenv_path)
api_key = os.getenv("OPENAI_API_KEY")

def load_llama_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Hugging Face 파이프라인 생성
    hf_pipeline = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        # max_length=8192, 
        temperature=0.1, 
        top_p=0.1,
        # truncation=True
        return_full_text=False
    )

    # Langchain에서 사용할 수 있는 Hugging Face Pipeline LLM 객체 생성
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    
    return llm

def generate_natural_language_answer(llm, question, sql_query, sql_result):
    answer_prompt_template = '''
    Given the following user question, corresponding SQL query, and SQL result, answer the user question in natural language in Korean.

    Question: {question}
    SQL Query: {sql_query}
    SQL Result: {sql_result}

    Answer: '''
    
    answer_prompt = answer_prompt_template.format(question=question, sql_query=sql_query, sql_result=sql_result)
    max_position_embeddings = llm.pipeline.model.config.max_position_embeddings
    max_input_length = len(llm.pipeline.tokenizer.encode(answer_prompt))

    max_new_tokens = max_position_embeddings - max_input_length
    if max_new_tokens < 0:
        logging.error("입력 데이터가 너무 큽니다. 입력 텍스트의 길이를 줄이거나 모델의 설정을 확인하세요.")
        return None
    
    try:
        response = llm.invoke(answer_prompt, max_new_tokens=max_new_tokens)
    except ValueError as e:
        logging.error(f"Error generating answer: {str(e)}")
        response = None
    
    return response

def sql_result(llm, db, question):
    few_shot_examples = '''
        Example 1)
        Question: 24.06.01~24.07.31 기간 동안 이벤트에 응모하고, 앱으로 단기카드대출을 10000원 이상 이용한 고객 리스트를 뽑아줘.
        SQLQuery: "SELECT DISTINCT T1.CNO FROM WMK_T_CMP_APL_OJP AS T1 INNER JOIN WSC_V_UMS_FW_HIST AS T2 ON T1.CMP_ID = T2.CMP_ID INNER JOIN WMG_T_D_SL_OUT AS T3 ON T1.CNO = T3.PSS_CNO WHERE T2.UMS_MSG_DTL_CN LIKE '%단기카드대출%' AND T3.STDT BETWEEN '20240604' AND '20240630' AND T3.SL_PD_DC = 3 AND T3.SL_AM > 10000;"
        SQLResult: [('123456789010000', '345678901230000')]
        Answer: 24.06.04~24.06.30 기간 내에 앱으로 단기카드대출을 10000원 이상 이용한 이벤트 응모 고객은 '123456789010000', '345678901230000'로 2 명 입니다.
        
        Example 2)
        Question: 24.06.04~24.06.30 기간 내에 이벤트에 응모하고, 앱을 통해 단기카드대출을 10000원 이상 이용한 고객을 뽑아줘. 추출된 고객 수를 UMS메시지의 이벤트 제공 혜택 포인트와 곱해서 최종적으로 산출된 오퍼 금액을 알려줘.
        SQLQuery: "SELECT COUNT(DISTINCT T1.CNO) * 10000 AS TotalOfferAmount FROM WMK_T_CMP_APL_OJP AS T1 INNER JOIN WSC_V_UMS_FW_HIST AS T2 ON T1.CMP_ID = T2.CMP_ID INNER JOIN WMG_T_D_SL_OUT AS T3 ON T1.CNO = T3.PSS_CNO WHERE T2.UMS_MSG_DTL_CN LIKE '%단기카드대출%' AND T3.STDT BETWEEN '20240604' AND '20240630' AND T3.SL_AM > 10000 AND T3.SL_PD_DC = 3 AND T2.UMS_MSG_DTL_CN LIKE '%띵코인 1만 포인트 적립%';"
        SQLResult: [(20000,)]
        Answer: 24.06.04~24.06.30 기간 내에 단기카드대출을 10000원 이상 이용한 고객 대상 제공되는 이벤트 총 금액은 띵코인 20000 포인트 입니다.
        
        Example 3)
        Question: 24.06.01~24.07.31 기간 내에 이벤트에 응모하고, 일시불로 1000원 이상 결제한 고객을 뽑아줘. 추출된 고객 수를 UMS메시지의 이벤트 제공 혜택 금액과 곱해서 최종적으로 산출된 오퍼 금액을 알려줘.
        SQLQuery: "SELECT COUNT(DISTINCT T1.CNO) * 5000 AS TotalOfferAmount FROM WMK_T_CMP_APL_OJP AS T1 INNER JOIN WSC_V_UMS_FW_HIST AS T2 ON T1.CMP_ID = T2.CMP_ID INNER JOIN WMG_T_D_SL_OUT AS T3 ON T1.CNO = T3.PSS_CNO WHERE T2.UMS_MSG_DTL_CN LIKE '%일시불%' AND T3.STDT BETWEEN '20240601' AND '20240731' AND T3.SL_AM > 1000 AND T3.SL_PD_DC = 1 AND T2.UMS_MSG_DTL_CN LIKE '%현금 5천 원%';"
        SQLResult: [(10000,)]
        Answer: 24.06.01~24.07.31 기간 내에 일시불을 1000원 이상 이용한 고객 대상 제공되는 이벤트 총 금액은 10000 원 입니다.
    '''

    # template = '''Given an input question, create a syntactically correct top {top_k} {dialect} query to run which should end with a semicolon.
    #     Use the following format:

    #     Question: "Question here"
    #     SQLQuery: "SQL Query to run"
    #     SQLResult: "Result of the SQLQuery"
    #     Answer: "Final answer here"

    #     Only use the following tables:

    #     {table_info}.

    #     DO NOT use non-existent tables or columns.

    #     Question: {input}'''

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
            print("-"*100)
            if fix == 1:
                print("SQL query: ", generated_sql_query)
                answer = generate_natural_language_answer(llm, question, generated_sql_query, result)
            else:
                print("SQL query: ", corrected_sql_query)
                answer = generate_natural_language_answer(llm, question, corrected_sql_query, result)

            print("SQL result: ", result)
            print(answer)
            break
        fix += 1
    
    if final is None:
        print("##### SQL Execution Error No Result #####")

    # try:
    #     result = execute.invoke({"query": generated_sql_query})
    #     print("-"*200)
    #     print("SQL result: ", result)

    #     answer = generate_natural_language_answer(llm, question, generated_sql_query, result)
    #     print(answer)
    # except sqlite3.OperationalError as e:
    #     # SQL 쿼리 실행 중 발생한 OperationalError 예외 처리
    #     print("-" * 200)
    #     print(f"Error executing SQL query: {e}")
    #     error_message = str(e)

    #     new_template = '''
    #         The following SQL query resulted in an error: {sql_query}

    #         Error: {error_message}

    #         Based on the provided table and column information, please correct the SQL query.

    #         {table_info}

    #         Question: {input}

    #         SQLQuery: '''
        
    #     tables = db.get_usable_table_names()
    #     table_info = {}

    #     for table in tables:
    #         table_columns = db.get_table_info(table)
    #         column_names = [col["name"] for col in table_columns]
    #         table_info[table] = column_names

    #     table_info_str = "\n".join([f"Table: {table}, Columns: {', '.join(columns)}" for table, columns in table_info.items()])

    #     prompt = PromptTemplate.from_template(new_template)
    #     query_chain = create_sql_query_chain(llm, db, prompt=prompt)
    #     corrected_sql_query = query_chain.invoke({"sql_query": result, "table_info": table_info_str, "input": question, "error_message": error_message})
        
    #     result = execute.invoke({"query": corrected_sql_query})
    #     print("-"*200)
    #     print(result)

    #     answer = generate_natural_language_answer(llm, question, corrected_sql_query, result)
    #     print(answer)

    # except Exception as e:
    #     print("-" * 200)
    #     print(f"Error: {e}")

if __name__ == "__main__":
    model_id = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'
    llm = load_llama_model(model_id)

    # db_name = input("Input DB name: ")
    # llm = Ollama(model="llama3.1:latest", temperature=0)
    # llm = Ollama(model="llama3.1:70b", temperature=0)
    # llm = Ollama(model="codellama:70b", temperature=0)
    # llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=None, openai_api_key=api_key)

    # db = SQLDatabase.from_uri(f"sqlite:///{db_name}.db")
    db = SQLDatabase.from_uri(f"sqlite:///market.db")
    # print(db.dialect)

    print(db.get_usable_table_names())
    # print(db.get_table_info())
    print("-"*200)
    # question = input("DB 질문을 입력하세요: ")0
    question = "24.06.04~24.06.30 기간 내에 이벤트 응모하고, 롯데카드로 앱에서 단기카드대출 10000원 이상 이용한 고객 리스트를 뽑아주세요. 고객들한테 나가야 하는 최종 혜택 금액도 계산해주세요."

    sql_result(llm, db, question)
