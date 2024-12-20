import os, torch
import logging
from openai import OpenAI
import google.generativeai as genai

from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_anthropic import ChatAnthropic
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.chains import create_sql_query_chain
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from dotenv import load_dotenv
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("text2query_market")
mlflow.langchain.autolog(log_models=True, log_input_examples=True)

# logging.basicConfig(level=logging.DEBUG)
dotenv_path = os.path.join(os.path.dirname(__file__), '../config', '.env')
load_dotenv(dotenv_path)
openai_api_key = os.getenv("OPENAI_API_KEY")
nvidia_api_key = os.getenv("NVIDIA_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

def load_google_model():
    genai.configure(api_key=google_api_key)

    # generation_config = {
    #     "temperature": 0.1,
    #     "top_p": 0.5,
    #     # "top_k": 64,
    #     "max_output_tokens": 8192,
    #     "response_mime_type": "text/plain",
    # }

    # model = genai.GenerativeModel(
    #     model_name="gemini-1.5-pro",
    #     generation_config=generation_config,
    # )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.1,
        top_p=0.5,
        max_tokens=8192,
        timeout=None,
    )

    return llm

def load_anthropic_model():
    client = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=anthropic_api_key,
        temperature=0,
        max_tokens=1024,
    )
    
    return client 

def load_nvidia_model():
    client = ChatNVIDIA(
        model="meta/llama-3.1-405b-instruct",
        api_key=nvidia_api_key, 
        temperature=0.1,
        top_p=0.5,
        max_tokens=1024,
    )
    
    # for chunk in client.stream([{"role":"user","content":"client = OpenAI(\n  base_url = \"https://integrate.api.nvidia.com/v1\",\n  api_key = \"$API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC\"\n) 에서 openai를 호출하는 이유는 무엇입니까?"}]): 
    #   print(chunk.content, end="")

    return client

def generate_natural_language_answer_nvidia(llm, question, sql_query, sql_result):
    answer_prompt_template = '''
        Given the following user question, corresponding SQL query, and SQL result, answer the user question in natural language in Korean.

        Question: {question}
        SQL Query: {sql_query}
        SQL Result: {sql_result}

        Answer: '''
    
    answer_prompt = answer_prompt_template.format(question=question, sql_query=sql_query, sql_result=sql_result)

    try:
        final_answer = ""
        print("Starting streaming response...")
        for chunk in llm.stream([{"role": "user", "content": answer_prompt}]):
            if hasattr(chunk, 'content') and chunk.content.strip():
                final_answer += chunk.content

        # if final_answer.strip() == "":
        #     logging.error("최종 답변이 비어 있습니다.")
        # else:
        #     print(f"Final answer: {final_answer}")

        return final_answer.strip()
    
    except Exception as e:
        logging.error(f"Error generating answer: {str(e)}")
        return None

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

def sql_result(llm, db, question):
    few_shot_examples = '''        
        Example 1)
        Question: 24.03.02~24.03.31 기간 내에 이벤트에 응모하고, 앱을 통해 단기카드대출을 A원 이상 이용한 고객을 뽑아줘. 이벤트로 나가는 최종 오퍼 금액을 알려줘.
        SQLQuery: "SELECT DISTINCT T1.CNO, COUNT(DISTINCT T1.CNO) * '혜택금액' AS TotalOfferAmount
                    FROM WMK_T_CMP_APL_OJP AS T1
                        INNER JOIN WSC_V_UMS_FW_HIST AS T2 ON T1.CMP_ID = T2.CMP_ID
                        INNER JOIN WMG_T_D_SL_OUT AS T3 ON T1.CNO = T3.PSS_CNO
                    WHERE T2.UMS_MSG_DTL_CN LIKE '%단기카드대출%'
                        AND T3.STDT BETWEEN '20240302' AND '20240331'
                        AND T3.SL_AM > A
                        AND T3.SL_PD_DC = 3
                        AND T2.UMS_MSG_DTL_CN LIKE '%1만 포인트 적립%';"
        
        Example 2)
        Question: 24.03.02~24.03.31 기간 내에 이벤트에 응모하고, 일시불로 B원 이상 결제한 고객을 뽑아줘. 이벤트로 나가는 최종 오퍼 금액을 알려줘.
        SQLQuery: "SELECT DISTINCT T1.CNO, COUNT(DISTINCT T1.CNO) * '혜택금액' AS TotalOfferAmount
                    FROM WMK_T_CMP_APL_OJP AS T1
                        INNER JOIN WSC_V_UMS_FW_HIST AS T2 ON T1.CMP_ID = T2.CMP_ID
                        INNER JOIN WMG_T_D_SL_OUT AS T3 ON T1.CNO = T3.PSS_CNO
                    WHERE T2.UMS_MSG_DTL_CN LIKE '%일시불%'
                        AND T3.STDT BETWEEN '20240302' AND '20240331'
                        AND T3.SL_AM > B
                        AND T3.SL_PD_DC = 1
                        AND T2.UMS_MSG_DTL_CN LIKE '%현금 5천 원%';"
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
    generated_sql_query = query_chain.invoke({
        "table_info": db.get_table_info(), 
        "input": question, 
        "dialect": db.dialect, 
        "top_k": 1, 
        "few_shot_examples": few_shot_examples})
    # print("SQL query: ", generated_sql_query.__repr__())
    
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
        else:
            print("*"*100)
            final = result
            if fix == 1:
                print("generated SQL query: ", generated_sql_query)
                answer = generate_natural_language_answer(llm, question, generated_sql_query, result)
                # answer = generate_natural_language_answer_nvidia(llm, question, generated_sql_query, result)
            else:
                print("corrected SQL query: ", corrected_sql_query)
                answer = generate_natural_language_answer(llm, question, corrected_sql_query, result)
                # answer = generate_natural_language_answer_nvidia(llm, question, corrected_sql_query, result)

            print("SQL result: ", result)
            print(answer)
            break
        fix += 1
    
    if final is None:
        print("##### SQL Execution Error No Result #####")

if __name__ == "__main__":
    # llm = Ollama(model="llama3.1:latest", temperature=0)
    # llm = Ollama(model="llama3.1:70b", temperature=0)
    # llm = Ollama(model="codellama:70b", temperature=0)
    # llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=None, openai_api_key=openai_api_key)
    # llm = load_nvidia_model()
    # llm = load_anthropic_model()

    llm = load_google_model()

    db = SQLDatabase.from_uri(f"sqlite:///database/market_vf.db")
    print(db.get_usable_table_names())
    print("-"*200)

    # question = input("DB 질문을 입력하세요: ")
    question = "24.06.04~24.06.30 기간 내에 이벤트 응모하고, 롯데카드 앱에서 단기카드대출 10000원 이상 이용한 고객 리스트를 뽑아주세요. 고객 별 지급 오퍼 금액과 고객 전체로 나가는 최종 오퍼 금액을 계산해주세요."

    sql_result(llm, db, question)
