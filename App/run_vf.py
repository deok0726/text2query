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

def load_hf_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    hf_pipeline = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_length=16384, 
        temperature=0.1, 
        top_p=0.5,
        truncation=True,
        return_full_text=False
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    
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
    few_shots = '''
        Example 1)
        Question: 24. 1~3월까지 'A' 고객이 OOO에서 사용한 결제일 별 결제금액을 알려줘, 보유카드 중에 할인을 받을 수 있는 상품이 있다면 얼마를 할인 받았고, 이번 달 할인 한도가 얼마나 남았는지 알려줘. 마지막으로, OOO와 관련해서 추천해줄만한 이벤트나 UMS 내용이 있다면 알려줘.
        SQLQuery: 
            SELECT 
                (SELECT json_group_array(json_object('month', month, 'total_spent', total_spent))
                FROM (SELECT strftime('%Y-%m', substr(SL_DT, 1, 4) || '-' || substr(SL_DT, 5, 2) || '-' || substr(SL_DT, 7, 2)) AS month, SUM(COALESCE(BIL_PRN, SL_AM)) AS total_spent
                    FROM WBM_T_BLL_SPEC_IZ WHERE ACCTNO = 'A' AND SL_DT BETWEEN '20240101' AND '20240331' AND BLL_MC_NM LIKE '%OOO%' GROUP BY month)) AS monthly_spending,
                (SELECT json_group_array(json_object('month', month, 'total_discount', total_discount))
                FROM (SELECT strftime('%Y-%m', substr(SL_DT, 1, 4) || '-' || substr(SL_DT, 5, 2) || '-' || substr(SL_DT, 7, 2)) AS month, SUM(BLL_SV_AM) AS total_discount
                    FROM WBM_T_BLL_SPEC_IZ WHERE ACCTNO = 'A' AND SL_DT BETWEEN '20240101' AND '20240331' AND BLL_MC_NM LIKE '%OOO%' AND BLL_SV_DC IN (SELECT SV_C FROM WPD_T_SV_SNGL_PRP_INF) GROUP BY month)) AS monthly_discounts,
                ((SELECT MLIM_AM FROM WPD_T_SV_SNGL_PRP_INF WHERE SV_C = 'SP*****') - COALESCE((SELECT SUM(BLL_SV_AM) AS total_discount
                FROM WBM_T_BLL_SPEC_IZ WHERE ACCTNO = 'A' AND SL_DT BETWEEN strftime('%Y%m%d', date('now', 'start of month')) AND strftime('%Y%m%d', date('now', 'start of month', '+1 month', '-1 day')) AND BLL_MC_NM LIKE '%OOO%' AND BLL_SV_DC IN (SELECT SV_C FROM WPD_T_SV_SNGL_PRP_INF)), 0)) AS remaining_discount_limit,
                (SELECT json_group_array(EVN_BULT_TIT_NM) FROM WLP_T_EVN_INF WHERE EVN_BULT_TIT_NM LIKE '%OOO%' AND EVN_SDT <= strftime('%Y%m%d', 'now') AND EVN_EDT >= strftime('%Y%m%d', 'now')) AS recommended_events,
                (SELECT json_group_array(UMS_MSG_CN) FROM WSC_T_UMS_FW_HIST WHERE UMS_MSG_CN LIKE '%OOO%') AS recommended_ums_messages;
                '''
    
    template = '''Given an input question, create a syntactically correct top {top_k} {dialect} query to run which should end with a semicolon.
        Use the following format:

        Question: "Question here"
        SQLQuery: "SQL Query to run"
        SQLResult: "Result of the SQLQuery"
        Answer: "Final answer here"

        Only use the following tables:

        {table_info}.

        Here are some examples of previous questions and their corresponding SQL queries to help you understand the table and column structure:

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
        "few_shot_examples": few_shots})
    
    execute = QuerySQLDataBaseTool(db=db)
    
    fix = 1
    retry = 6
    final = None
    result = execute.invoke({"query": generated_sql_query})
    while (fix < retry) :
        if "OperationalError" in result:
            print(f"##### Error executing SQL query ##### \n{result}")
            print(f"Try {fix}...")
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
                print("Generated SQL query: ", generated_sql_query)
                answer = generate_natural_language_answer(llm, question, generated_sql_query, result)
                # answer = generate_natural_language_answer_nvidia(llm, question, generated_sql_query, result)
            else:
                print("Corrected SQL query: ", corrected_sql_query)
                answer = generate_natural_language_answer(llm, question, corrected_sql_query, result)
                # answer = generate_natural_language_answer_nvidia(llm, question, corrected_sql_query, result)

            print("SQL result: ", result)
            print(answer)
            break
        fix += 1
    
    if final is None:
        print("########## SQL Execution Error: No Result ##########")
        return

if __name__ == "__main__":
    # llm = Ollama(model="llama3.1:latest", temperature=0)
    # llm = Ollama(model="llama3.1:70b", temperature=0)
    # llm = Ollama(model="codellama:70b", temperature=0)
    # llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=None, openai_api_key=openai_api_key)
    # llm = load_nvidia_model()
    # llm = load_anthropic_model()
    # llm = load_hf_model('MLP-KTLim/llama-3-Korean-Bllossom-8B')

    llm = load_google_model()

    db = SQLDatabase.from_uri("sqlite:///app_vf.db")
    print(db.get_usable_table_names())
    print("-"*200)
    # print(db.get_table_info())

    # question = input("DB 질문을 입력하세요: ")
    a = "24. 7~10월까지 '70018819695' 고객이 사용한 금액을 카드 별로 알려줘."
    b = "24. 7~9월까지 '70018819695' 고객이 기간 동안 스타벅스에서 사용한 총액과 월 별 결제금액을 알려줘."
    c = "24. 7~9월까지 '70018819695' 고객이 기간 동안 스타벅스에서 할인을 받았다면 할인 금액을 월 별로 알려주고, 이번 달 할인 한도가 얼마나 남았는지도 알려줘."
    d = "'70018819695' 고객에게 스타벅스와 관련해서 추천해줄만한 이벤트와 UMS 메시지를 알려줘."
    e = "24. 7~9월까지 '70018819695' 고객이 기간 동안 스타벅스에서 사용한 총액과 결제일 별 결제금액을 알려줘. 그리고 보유카드 중에 할인 받은 상품이 있다면 얼마를 할인 받았는지, 이번 달 할인 한도가 얼마나 남았는지도 알려줘. 그리고 스타벅스와 관련해서 추천해 줄만한 이벤트들이 있다면 그 내용도 알려줘"
    
    questions = [a, b, c, d, e]
    for question in questions:
        print("question:", question)
        sql_result(llm, db, question)