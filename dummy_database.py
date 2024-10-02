import sqlite3

def create_database():
    conn = sqlite3.connect('example.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS companys (
        id INTEGER PRIMARY KEY,
        name TEXT UNIQUE NOT NULL,
        established INTEGER NOT NULL,
        homepage TEXT UNIQUE NOT NULL,
        phone TEXT UNIQUE NOT NULL,
        location TEXT,
        industry TEXT
    )
    ''')

    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT UNIQUE NOT NULL,
        age INTEGER NOT NULL,
        email TEXT UNIQUE NOT NULL,
        phone TEXT UNIQUE NOT NULL,
        location TEXT,
        gender TEXT
    )
    ''')
    conn.commit()
    conn.close()

def insert_data():
    conn = sqlite3.connect('example.db')
    c = conn.cursor()

    # 예시 데이터
    companys = [
        (1, 'Tech Innovators', 2001, 'techinnovators.com', '123-456-7890', 'Seoul', 'Technology'),
        (2, 'Health Solutions', 1998, 'healthsolutions.org', '234-567-8901', 'Busan', 'Healthcare'),
        (3, 'Green Energy', 2010, 'greenenergy.net', '345-678-9012', 'Incheon', 'Energy'),
        (4, 'Fast Delivery', 2005, 'fastdelivery.co.kr', '456-789-0123', 'Daegu', 'Logistics'),
        (5, 'Foodies Delight', 2015, 'foodiesdelight.com', '567-890-1234', 'Daejeon', 'Food'),
        (6, 'EduTech', 2000, 'edutech.edu', '678-901-2345', 'Gwangju', 'Education'),
        (7, 'AutoMakers', 1995, 'automakers.com', '789-012-3456', 'Ulsan', 'Automotive'),
        (8, 'Fashion Forward', 2008, 'fashionforward.com', '890-123-4567', 'Suwon', 'Fashion'),
        (9, 'FinTech Pioneers', 2012, 'fintechpioneers.io', '901-234-5678', 'Yongin', 'Finance'),
        (10, 'Travel Experts', 2003, 'travelexperts.com', '012-345-6789', 'Jeju', 'Travel'),
        (11, 'Smart Solutions', 2002, 'smartsolutions.com', '123-456-7891', 'Seoul', 'Technology'),
        (12, 'Healthy Living', 1999, 'healthyliving.org', '234-567-8902', 'Busan', 'Healthcare'),
        (13, 'Solar Power', 2011, 'solarpower.net', '345-678-9013', 'Incheon', 'Energy'),
        (14, 'Quick Transport', 2006, 'quicktransport.co.kr', '456-789-0124', 'Daegu', 'Logistics'),
        (15, 'Gourmet Foods', 2016, 'gourmetfoods.com', '567-890-1235', 'Daejeon', 'Food'),
        (16, 'LearnTech', 2001, 'learntech.edu', '678-901-2346', 'Gwangju', 'Education'),
        (17, 'Car Innovators', 1996, 'carinnovators.com', '789-012-3457', 'Ulsan', 'Automotive'),
        (18, 'Style Makers', 2009, 'stylemakers.com', '890-123-4568', 'Suwon', 'Fashion'),
        (19, 'Crypto Finances', 2013, 'cryptofinances.io', '901-234-5679', 'Yongin', 'Finance'),
        (20, 'Adventure Travel', 2004, 'adventuretravel.com', '012-345-6790', 'Jeju', 'Travel'),
        (21, 'Tech Wizards', 2003, 'techwizards.com', '123-456-7892', 'Seoul', 'Technology'),
        (22, 'Wellness Solutions', 2000, 'wellnesssolutions.org', '234-567-8903', 'Busan', 'Healthcare'),
        (23, 'Wind Energy', 2012, 'windenergy.net', '345-678-9014', 'Incheon', 'Energy'),
        (24, 'Fast Logistics', 2007, 'fastlogistics.co.kr', '456-789-0125', 'Daegu', 'Logistics'),
        (25, 'Delightful Eats', 2017, 'delightfuleats.com', '567-890-1236', 'Daejeon', 'Food'),
        (26, 'Edu Innovators', 2002, 'eduinnovators.edu', '678-901-2347', 'Gwangju', 'Education'),
        (27, 'Auto Experts', 1997, 'autoexperts.com', '789-012-3458', 'Ulsan', 'Automotive'),
        (28, 'Fashion Trends', 2010, 'fashiontrends.com', '890-123-4569', 'Suwon', 'Fashion'),
        (29, 'FinTech Solutions', 2014, 'fintechsolutions.io', '901-234-5680', 'Yongin', 'Finance'),
        (30, 'Global Travels', 2005, 'globaltravels.com', '012-345-6791', 'Jeju', 'Travel')
    ]

    # 여러 개의 데이터를 한 번에 삽입
    c.executemany('INSERT INTO companys (id, name, established, homepage, phone, location, industry) VALUES (?, ?, ?, ?, ?, ?, ?)', companys)

    users = [
        ('Alice', 30, 'alice@example.com', '123-456-7890', 'Seoul', 'F'),
        ('Bob', 25, 'bob@example.com', '234-567-8901', 'Busan', 'M'),
        ('Charlie', 35, 'charlie@example.com', '345-678-9012', 'Incheon', 'M'),
        ('David', 40, 'david@example.com', '456-789-0123', 'Daegu', 'M'),
        ('Eve', 28, 'eve@example.com', '567-890-1234', 'Daejeon', 'F'),
        ('Frank', 33, 'frank@example.com', '678-901-2345', 'Gwangju', 'M'),
        ('Grace', 29, 'grace@example.com', '789-012-3456', 'Ulsan', 'F'),
        ('Hank', 32, 'hank@example.com', '890-123-4567', 'Suwon', 'M'),
        ('Ivy', 27, 'ivy@example.com', '901-234-5678', 'Yongin', 'F'),
        ('Jack', 31, 'jack@example.com', '012-345-6789', 'Jeju', 'M'),
        ('Kelly', 26, 'kelly@example.com', '111-222-3333', 'Seoul', 'F'),
        ('Leo', 34, 'leo@example.com', '222-333-4444', 'Busan', 'M'),
        ('Mia', 30, 'mia@example.com', '333-444-5555', 'Incheon', 'F'),
        ('Noah', 28, 'noah@example.com', '444-555-6666', 'Daegu', 'M'),
        ('Olivia', 32, 'olivia@example.com', '555-666-7777', 'Daejeon', 'F'),
        ('Peter', 31, 'peter@example.com', '666-777-8888', 'Gwangju', 'M'),
        ('Quinn', 29, 'quinn@example.com', '777-888-9999', 'Ulsan', 'F'),
        ('Ryan', 27, 'ryan@example.com', '888-999-0000', 'Suwon', 'M'),
        ('Sophia', 33, 'sophia@example.com', '999-000-1111', 'Yongin', 'F'),
        ('Tom', 35, 'tom@example.com', '000-111-2222', 'Jeju', 'M'),
        ('Uma', 30, 'uma@example.com', '111-333-5555', 'Seoul', 'F'),
        ('Victor', 28, 'victor@example.com', '222-444-6666', 'Busan', 'M'),
        ('Wendy', 32, 'wendy@example.com', '333-555-7777', 'Incheon', 'F'),
        ('Xavier', 31, 'xavier@example.com', '444-666-8888', 'Daegu', 'M'),
        ('Yara', 29, 'yara@example.com', '555-777-9999', 'Daejeon', 'F'),
        ('Zach', 27, 'zach@example.com', '666-888-0000', 'Gwangju', 'M'),
        ('Abby', 33, 'abby@example.com', '777-999-1111', 'Ulsan', 'F'),
        ('Ben', 35, 'ben@example.com', '888-000-2222', 'Suwon', 'M'),
        ('Chloe', 30, 'chloe@example.com', '999-111-3333', 'Yongin', 'F'),
        ('Danny', 28, 'danny@example.com', '000-222-4444', 'Jeju', 'M')
    ]

    # 여러 개의 데이터를 한 번에 삽입
    c.executemany('INSERT INTO users (name, age, email, phone, location, gender) VALUES (?, ?, ?, ?, ?, ?)', users)
    conn.commit()
    conn.close()

def query_data():
    conn = sqlite3.connect('example.db')
    c = conn.cursor()
    c.execute('SELECT * FROM companys')
    rows = c.fetchall()
    for row in rows:
        print(row)
    conn.close()

def get_table_names(conn):
    """데이터베이스에서 모든 테이블 이름을 가져옵니다."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    return [table[0] for table in tables]

def get_row_count(conn, table_name):
    """특정 테이블의 행 수를 계산합니다."""
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    return cursor.fetchone()[0]

def find_table_with_max_rows(conn):
    """가장 많은 행을 가진 테이블을 찾습니다."""
    table_names = get_table_names(conn)
    max_rows = 0
    max_table = None

    for table_name in table_names:
        row_count = get_row_count(conn, table_name)
        print(f"Table {table_name} has {row_count} rows.")
        if row_count > max_rows:
            max_rows = row_count
            max_table = table_name

    return max_table, max_rows


if __name__ == "__main__":
    # create_database()
    # insert_data()
    # query_data()
    conn = sqlite3.connect('example.db')
    table, rows = find_table_with_max_rows(conn)
    print(table)
    print(rows)