import sys
from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.chains import create_sql_query_chain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import sqlite3
import logging
from sentence_transformers import SentenceTransformer, util
logging.basicConfig(level=logging.DEBUG)


# model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

def get_best_column_match(question, column_names):
    # 질문과 컬럼명을 임베딩하여 유사도 비교
    question_embedding = model.encode(question, convert_to_tensor=True)
    column_embeddings = model.encode(column_names, convert_to_tensor=True)
    
    # 각 컬럼명과 질문의 유사도를 계산
    similarities = util.pytorch_cos_sim(question_embedding, column_embeddings)
    
    # 유사도가 가장 높은 컬럼 선택
    best_match_index = similarities.argmax()
    return column_names[best_match_index]

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
        SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_META';
    ''')
    meta_tables = cursor.fetchall()

    # 모든 메타데이터 테이블의 값을 읽어오기
    metadata_dict = {}
    metadata_list = []
    for table in meta_tables:
        table_name = table[0]
        cursor.execute(f'''
            SELECT 컬럼명, 컬럼ID FROM {table_name}
        ''')
        metadata = cursor.fetchall()
        metadata_list.append(metadata)
        for col_name, col_id in metadata:
            metadata_dict[col_name] = (table_name, col_id)

    conn.close()
    # print("메타데이터 테이블:")
    print(metadata_dict)
    # metadata_str = "\n".join([f"{col_name}: {col_id}" for col_id, col_name in metadata])
  
    # print(metadata_list[0][0][0])
    # print(metadata_list[1][0][0])
    
    metadata_str = ""

    for idx, metadata in enumerate(metadata_list):
        table = metadata[0]
        print(table)
        # metadata_str = "\n".join([f"{table[1]} : {col_name}" for col_name, col_id in metadata[1:]])
        for col_name, col_id in metadata[1:]:
            metadata_str += f"{table[1]} : ({col_name}, {col_id})\n"

    

    print("-"*200)
    print(metadata_str)
    print("-"*200)

    sys.exit()

    natural_language_query = input("DB 질문을 입력하세요: ")
    prompt = PromptTemplate(template="""
        너는 한글 자연어 쿼리와 메타데이터 테이블을 입력받아 질문의 문맥을 파악하여 데이터베이스 검색에 필요한 테이블과 컬럼ID를 찾는 SQL agent야.
        메타데이터 테이블은 테이블 이름, 그 테이블에 속하는 컬럼의 이름과 이름에 매칭되는 ID 값이 ###테이블 : (컬럼명, 컬럼ID)### 형태로 저장돼 있어. 
        문맥을 통해 파악한 컬럼명을 기반으로 가장 적절한 ###테이블 : (컬럼명, 컬럼ID)###를 메타데이터 테이블에서 추출해줘.
                                   
        사용자 입력 자연어 쿼리: {query}

        메타데이터 테이블: {metadata}

        response: (테이블 : 컬럼ID)
    """)

    prompt_step = PromptTemplate(template="""
        Step 1: 자연어 질문을 해석하여 아래 메타데이터 테이블을 참고하여 어떤 테이블과 컬럼이 필요한지 알아내세요.
            자연어 질문: {query}
            메타데이터 테이블: {metadata}
        
        Step 2: 필요한 테이블과 컬럼의 ID를 사용하여 SQL 쿼리를 생성하세요.
            SQL query: 
    """)

    llm_chain = prompt_step | llm | StrOutputParser()
    response = llm_chain.invoke({"query": natural_language_query, "metadata": metadata_str}).strip()
    print(f"{response}")

    # selected_column_id = metadata_dict.get(response, "컬럼명을 찾을 수 없습니다.")


    # query_chain = create_sql_query_chain(llm, db)
    # execute = QuerySQLDataBaseTool(db=db)

    # execute_chain = query_chain | execute
    # result = execute_chain.invoke({"question": converted_query})