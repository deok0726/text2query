import os, sqlite3, logging, json
from openai import OpenAI
from langchain_community.utilities import SQLDatabase
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

dotenv_path = os.path.join(os.path.dirname(__file__), '../config', '.env')
load_dotenv(dotenv_path)
api_key = os.getenv("NVIDIA_API_KEY")

def load_llama_model():
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )
    return client

def generate_sql_query(llm, db, question):
    # SQL 생성 프롬프트 템플릿
    query_prompt_template = '''
    Given an input question, create a syntactically correct SQL query in {dialect} dialect to run, which should end with a semicolon.

    Question: {question}
    SQLQuery: '''
    
    prompt = query_prompt_template.format(question=question, dialect=db.dialect)
    
    # Nvidia API를 호출하여 SQL 쿼리 생성
    try:
        completion = llm.chat.completions.create(
            model="meta/llama-3.1-405b-instruct",
            messages=[{"role":"user", "content": prompt}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=512
        )
        
        sql_query = ""
        # 응답이 빈 문자열인지 먼저 확인
        for chunk in completion:
            if isinstance(chunk, tuple):
                chunk = chunk[0]  # 첫 번째 요소로 접근

            if not chunk:  # 응답이 비어있는지 확인
                logging.error("Nvidia API로부터 빈 응답을 받았습니다.")
                return None

            # 문자열일 경우 JSON 변환 시도
            try:
                if isinstance(chunk, str):
                    chunk = json.loads(chunk)  # JSON 파싱
            except json.JSONDecodeError as e:
                logging.error(f"응답을 JSON으로 변환하는 중 오류 발생: {str(e)}")
                continue  # 다음 chunk로 넘어감

            # 'choices'와 'delta'가 있는지 확인
            if 'choices' in chunk and 'delta' in chunk['choices'][0]:
                sql_query += chunk['choices'][0]['delta'].get('content', '')

        if not sql_query:  # 만약 SQL 쿼리가 생성되지 않았다면
            logging.error("SQL 쿼리 생성에 실패했습니다.")
            return None
        
        return sql_query.strip()

    except Exception as e:
        logging.error(f"SQL 쿼리 생성 중 오류 발생: {str(e)}")
        return None

def execute_sql_and_fix_errors(llm, db, question, sql_query):
    execute = QuerySQLDataBaseTool(db=db)
    result = execute.invoke({"query": sql_query})

    # If there's an SQL execution error, attempt to fix it
    fix = 1
    retry = 4
    final_result = None
    
    while fix < retry:
        if "OperationalError" in result:
            error_message = str(result)
            logging.error(f"### SQL Error on try {fix}: {error_message}")
            
            # Prompt template for fixing SQL errors
            fix_prompt_template = '''
            The following SQL query resulted in an error: {sql_query}

            Error: {error_message}

            Based on the provided table and column information, please correct the SQL query.

            Corrected SQLQuery: '''
            
            fix_prompt = fix_prompt_template.format(sql_query=sql_query, error_message=error_message)
            
            # Call Nvidia API to generate corrected SQL query
            completion = llm.chat.completions.create(
                model="meta/llama-3.1-405b-instruct",
                messages=[{"role": "user", "content": fix_prompt}],
                temperature=0.2,
                top_p=0.7,
                max_tokens=512
            )
            
            corrected_sql_query = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    corrected_sql_query += chunk.choices[0].delta.content
            
            sql_query = corrected_sql_query.strip()
            result = execute.invoke({"query": sql_query})
            
        else:
            final_result = result
            break
        fix += 1

    return final_result, sql_query

def generate_natural_language_answer(llm, question, sql_query, sql_result):
    # Prompt template for answer generation
    answer_prompt_template = '''
    Given the following user question, corresponding SQL query, and SQL result, answer the user question in natural language in Korean.

    Question: {question}
    SQL Query: {sql_query}
    SQL Result: {sql_result}

    Answer: '''
    
    answer_prompt = answer_prompt_template.format(question=question, sql_query=sql_query, sql_result=sql_result)

    # Call Nvidia API to generate the natural language answer
    try:
        completion = llm.chat.completions.create(
            model="meta/llama-3.1-405b-instruct",
            messages=[{"role":"user", "content": answer_prompt}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024
        )

        response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content

        return response.strip()
    
    except ValueError as e:
        logging.error(f"Error generating answer: {str(e)}")
        return None

def sql_result(llm, db, question):
    # Generate SQL query using Nvidia API
    sql_query = generate_sql_query(llm, db, question)
    if not sql_query:
        logging.error("SQL query generation failed.")
        return

    # Execute SQL query and attempt to fix errors if necessary
    sql_result, final_sql_query = execute_sql_and_fix_errors(llm, db, question, sql_query)
    if sql_result is None:
        logging.error("SQL execution failed after retries.")
        return

    # Generate natural language answer
    answer = generate_natural_language_answer(llm, question, final_sql_query, sql_result)
    if answer:
        print(f"Answer: {answer}")
    else:
        logging.error("Answer generation failed.")

if __name__ == "__main__":
    llm = load_llama_model()
    db = SQLDatabase.from_uri("sqlite:///app_vf.db")

    questions = [
        "24. 7~10월까지 '70018819695' 고객이 사용한 금액을 카드 별로 알려줘.",
        "24. 7~9월까지 '70018819695' 고객이 기간 동안 스타벅스에서 사용한 총액과 월 별 결제금액을 알려줘.",
        "24. 7~9월까지 '70018819695' 고객이 스타벅스에서 할인을 받았다면 할인 금액을 월 별로 알려주고, 이번 달 할인 한도가 얼마나 남았는지도 알려줘."
    ]

    for question in questions:
        sql_result(llm, db, question)
