# Create vector embeddings and store them in a postgres vector database using openai

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from typing import List
from langchain_postgres.vectorstores import PGVector
import openai 
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set OpenAI API key 
openai.api_key = os.getenv('OPENAI_API_KEY')

# Set the path to the docs
DATA_PATH = "ezza_docs/ameen_docs/"
CONNECTION_STRING = os.getenv('PGVECTOR_CONNECTION_STRING')
COLLECTION_NAME = "state_of_union_vectors" 


def main():
    generate_data_store()


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf")
    documents = loader.load()
    return documents


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_pgvector(chunks)


def split_text(documents: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=80,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # document = chunks[10]
    # print(document.page_content)
    # print(document.metadata)
    return chunks


def save_to_pgvector(chunks: List[Document]):   
    vector_store = PGVector.from_documents(
                embedding=OpenAIEmbeddings(),
                documents=chunks,
                connection=CONNECTION_STRING,
                #collection_name=COLLECTION_NAME,
                use_jsonb=True,
                async_mode=False,
            )

    print(f"Saved {len(chunks)} chunks to the PostgreSQL database.")
    return vector_store

if __name__ == "__main__":
    main()