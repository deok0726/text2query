import sqlite3
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.chains import create_sql_query_chain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import sqlite3
import logging
logging.basicConfig(level=logging.DEBUG)

def sql_result(db, question):
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

        You can check some examples of results:

        {few_shot_examples}

        Question: {input}'''
    prompt = PromptTemplate.from_template(template)

    query_chain = create_sql_query_chain(llm, db, prompt=prompt)
    generated_sql_query = query_chain.invoke({"table_info": db.get_table_info(), "input": question, "dialect": db.dialect, "top_k": 1, "few_shot_examples": few_shot_examples})

    print(generated_sql_query.__repr__())
    execute = QuerySQLDataBaseTool(db=db)
    try:
        result = execute.invoke({"query": generated_sql_query})
        print("-"*200)
        print(result)
        # result = execute.invoke({"query": "SELECT T1.MC_NM, T2.APR_DTTI, T2.CDNO, T2.DE_DT, T2.CNO, T2.APR_DE_AM FROM LM_TB_MC AS T1 INNER JOIN LP_TB_TBAUTH AS T2 ON T1.MCNO = T2.MCNO WHERE T1.MC_NM = '스타벅스'"})
    except:
       print("-"*200)
       print(generated_sql_query.__repr__())

def sql_answer(db, question):
    execute_query = QuerySQLDataBaseTool(db=db)
    write_query = create_sql_query_chain(llm, db)

    answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

            Question: {question}
            SQL Query: {query}
            SQL Result: {result}
            Answer: """
    )

    chain = (
        RunnablePassthrough.assign(query=write_query).assign(
            result=itemgetter("query") | execute_query
        )
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    chain.invoke({f"question": {question}})

if __name__ == "__main__":
    db_name = input("Input DB name: ")
    llm = Ollama(model="llama3.1:70b", temperature=0)
    # llm = Ollama(model="codellama:70b", temperature=0)
    
    db = SQLDatabase.from_uri(f"sqlite:///{db_name}.db")
    print(db.dialect)
    print(db.get_usable_table_names())
    print("-"*200)
    question = input("DB 질문을 입력하세요: ")
    # question = "스타벅스에서 결제한 전체 내역을 보여줘"

    sql_result(db, question)
    sql_answer(db, question)
