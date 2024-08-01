import argparse
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv
from typing import List, Tuple
import openai 
import os

# Load environment variables
load_dotenv()

# Set OpenAI API key 
openai.api_key = os.getenv('OPENAI_API_KEY')

# Set the path to the DB
CONNECTION_STRING = os.getenv('PGVECTOR_CONNECTION_STRING')


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    print(rag_response())
    

def rag_response():
    query_text = retrieve_query_text()
    results = query_pgvector(query_text=query_text)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return
    prompt = generate_rag(results, query_text)
    return generate_response(prompt, results)


def retrieve_query_text():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    return args.query_text


def query_pgvector(query_text: str):
    embeddings = OpenAIEmbeddings()
    pgvector_store = PGVector(
        embeddings=embeddings, 
        connection=CONNECTION_STRING,
        collection_name='langchain') 
    
    results = pgvector_store.similarity_search_with_relevance_scores(query=query_text, k=3)
    return results


def generate_rag(results:List[Tuple[Document, float]], query_text:str):
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    return prompt_template.format(context=context_text, question=query_text)


def generate_response(prompt:str, results:List[Tuple[Document, float]]):
    model = ChatOpenAI()
    response_text = model.invoke(prompt)
    response_text = response_text.__dict__["content"]
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response

if __name__ == "__main__":
    main()