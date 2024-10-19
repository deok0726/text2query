import pandas as pd
from sqlalchemy import create_engine

df = pd.read_excel('1_WBP_T_CLN_PAY_IZ.xlsx', sheet_name='Sheet1')

engine = create_engine('sqlite:///app_vf2.db')

df.to_sql('WBP_T_CLN_PAY_IZ', con=engine, if_exists='replace', index=False)

print("엑셀 파일의 데이터를 데이터베이스에 성공적으로 저장했습니다!")