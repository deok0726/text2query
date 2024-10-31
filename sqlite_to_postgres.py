import pandas as pd
from sqlalchemy import create_engine, text

# SQLite 연결
sqlite_engine = create_engine("sqlite:///App/database/app_vf.db")

# PostgreSQL 연결
postgresql_engine = create_engine("postgresql://loca:digiloca@localhost/app")

# 테이블 목록 가져오기
with sqlite_engine.connect() as conn:
    # 쿼리를 text()로 래핑하여 실행
    tables = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';")).fetchall()
    table_names = [table[0] for table in tables]

print("SQLite 테이블 목록:", table_names)

# 각 테이블을 PostgreSQL로 복사
for table_name in table_names:
    # SQLite에서 테이블 데이터 가져오기
    df = pd.read_sql_table(table_name, sqlite_engine)
    
    # PostgreSQL로 데이터 적재
    df.to_sql(table_name, postgresql_engine, if_exists="replace", index=False)
    print(f"Table {table_name} copied successfully.")