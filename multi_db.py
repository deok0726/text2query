from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType
import sqlite3
import logging

db = SQLDatabase.from_uri("mssql+pymssql://<some server>/<some db>",
                          include_tables=['Some table'], view_support=True)
                        
db1 = SQLDatabase.from_uri("mssql+pymssql://<some other server>/<some other db>",
                        include_tables=['Some other table'], view_support=True)

toolkit = SQLDatabaseToolkit(db=db, llm=llm, reduce_k_below_max_tokens=True)

sql_agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

toolkit1 = SQLDatabaseToolkit(db=db1, llm=llm, reduce_k_below_max_tokens=True)

sql_agent_executor1 = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

tools = [
    Tool(
        name = "Object or Product to Classification Association",
        func=sql_agent_executor.run,
        description="""
        Useful for when you need to Query on database to find the object or product to classification association.
        
        <user>: Get me top 3 records with Object number and description for approved classification KKK
        <assistant>: I need to check Object or Product to Classification Association details.
        <assistant>: Action: SQL Object or Product to Classification Association
        <assistant>: Action Input: Check The Object or Product to Classification Association Table

        """
    ),
    Tool(
        name = "Authorization or Authority or License Database",
        func=sql_agent_executor1.run,
        description="""
        Useful for when you need to Query on some thing else .
        
        <user>: Get me top 2 Authority Records with LicenseNumber
        <assistant>: I need to check Authorization or Authority or License Database details.
        <assistant>: Action: SQL Authorization or Authority or License Database
        <assistant>: Action Input: Check The Authorization or Authority or License Database Table

        """
    )
]
