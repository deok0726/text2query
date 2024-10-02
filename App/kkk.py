import sqlite3
from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.chains import create_sql_query_chain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import sqlite3
import logging
logging.basicConfig(level=logging.DEBUG)


if __name__ == "__main__":
    db_name = input("Input DB name: ")
    llm = Ollama(model="llama3.1:70b")
    # llm = Ollama(model="codellama:70b")
    
    db = SQLDatabase.from_uri(f"sqlite:///{db_name}.db")
    print(db.dialect)
    print(db.get_usable_table_names())
    print("-"*200)

    conn = sqlite3.connect(f'{db_name}.db')
    cursor = conn.cursor()

    # 데이터베이스에 존재하는 모든 메타데이터 테이블 이름을 검색
    cursor.execute('''
        SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_mapping';
    ''')
    meta_tables = cursor.fetchall()

    # 모든 메타데이터 테이블의 값을 읽어오기
    table_metadata = []
    column_metadata = []
    for table in meta_tables:
        table_name = table[0]
        cursor.execute(f'''
            SELECT * FROM {table_name}
        ''')
        if table_name == 'table_mapping':
            table_metadata.extend(cursor.fetchall())
        elif table_name == 'column_mapping':
            column_metadata.extend(cursor.fetchall())
    conn.close()

    # print(metadata)
    # 메타데이터를 문자열로 변환
    metadata_str = "\n".join([f"{col_id}: {col_name}" for col_id, col_name in metadata])
    print("메타데이터 테이블:")
    print(metadata_str)
    print("-"*200)

    question = input("DB 질문을 입력하세요: ")
    prompt = PromptTemplate(template="""
        아래의 예시를 참고해서 컬럼명을 포함하는 자연어 쿼리를 기반으로 메타데이터 테이블을 참조하여 컬럼명과 가장 일치하는 컬럼ID로 변환한 자연어 쿼리를 생성해줘.
        컬럼명과 일치하는 컬럼ID가 없다면 입력된 자연어 쿼리를 그대로 반환해.
                            
            ### 메타데이터 테이블 예시:
            | 컬럼명       | 컬럼ID  |
            |--------------|---------|
            | 고객명       | col_01  |
            | 청구서번호   | col_02  |
            | 사용금액     | col_03  |
            | 청구일자     | col_04  |
            | 카드번호     | col_05  |
            
            예시 1:
            #### 사용자 입력 자연어 쿼리:
            "지난달 사용금액이 100만원 이상인 고객명을 조회해줘"
            #### 변환된 자연어 쿼리:
            "지난달 col_03이 100만원 이상인 col_01을 조회해줘"
            
            예시 2:
            #### 사용자 입력 자연어 쿼리:
            "청구일자가 2024년 8월인 청구서번호를 알려줘"
            #### 변환된 자연어 쿼리:
            "col_04가 2024년 8월인 col_02를 알려줘"

            예시 3:
            #### 사용자 입력 자연어 쿼리:
            "카드번호가 1234-5678-9876-5432인 고객의 사용금액을 확인해줘"
            #### 변환된 자연어 쿼리:
            "col_05가 1234-5678-9876-5432인 고객의 col_03을 확인해줘"
        
        -----------------------------------------------------------------------------------------------------------------------------------------------------------
        다음은 입력으로 받는 자연어 쿼리와 메타데이터 테이블의 정보야. 변환된 자연어 쿼리만 출력으로 반환해줘.
                            
        사용자 입력 자연어 쿼리
        {query}

        메타데이터 테이블:
        {metadata}

        변환된 자연어 쿼리:
    """)

    llm_chain = prompt | llm | StrOutputParser()
    converted_query = llm_chain.invoke({"query": question, "metadata": metadata_str}).strip()
    print(f"변환된 자연어 쿼리: {converted_query}")

    query_chain = create_sql_query_chain(llm, db)

    execute = QuerySQLDataBaseTool(db=db)
    execute_chain = query_chain | execute
    result = execute_chain.invoke({"question": converted_query})