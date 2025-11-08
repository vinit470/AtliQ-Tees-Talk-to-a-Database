from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX
from langchain.prompts.prompt import PromptTemplate
from langchain.sql_database import SQLDatabase

from urllib.parse import quote_plus
from dotenv import load_dotenv
import os

from few_shots import few_shots   # Ensure few_shots.py returns a list of examples

# Load environment variables from .env
load_dotenv()

def get_few_shot_db_chain():
    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2
    )

    # Database Credentials
    db_user = "root"
    db_password = quote_plus("vinit@07062004")  # URL-encodes special characters
    db_host = "localhost"
    db_name = "atliq_tshirts"

    # DB Connection
    db = SQLDatabase.from_uri(
        f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
        sample_rows_in_table_info=3
    )

    # Embeddings + Vector Store for example selection
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    to_vectorize = [" ".join(example.values()) for example in few_shots]
    vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)

    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2
    )

    # Custom SQL Prompt
    mysql_prompt = """
You are a MySQL expert. Given an input question, first create a correct MySQL query, then return the answer.
Never select all columns. Use only necessary columns, wrap column names in backticks (`), and respect table structure.
Use LIMIT appropriately and consider dates using CURDATE() if needed.

Use this exact format:

Question: question
SQLQuery: SQL query
SQLResult: result of query
Answer: final answer
"""

    # Example Format
    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}"
    )

    # Few-shot prompt template
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"]
    )

    # ✅ Build SQL chain using the few-shot prompt
    db_chain = SQLDatabaseChain.from_llm(
        llm=llm,
        db=db,
        prompt=few_shot_prompt,
        verbose=False
    )

    return db_chain


if __name__ == "__main__":
    chain = get_few_shot_db_chain()
    result = chain.run("How many total t shirts are left in total in stock?")
    print(f"The answer is {result}")
