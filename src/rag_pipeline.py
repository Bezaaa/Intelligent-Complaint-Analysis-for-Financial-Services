
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document
from src.prompts import BASE_PROMPT_TEMPLATE
from src.llm import generate_answer  

import os

def get_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("vector_store/", embeddings, allow_dangerous_deserialization=True)

def retrieve_documents(question: str, k=5):
    vectorstore = get_vector_store()
    docs = vectorstore.similarity_search(question, k=k)
    return docs

def build_prompt(question: str, docs: list[Document]):
    context = "\n---\n".join([doc.page_content for doc in docs])
    return BASE_PROMPT_TEMPLATE.format(context=context, question=question)

def answer_question(question: str):
    docs = retrieve_documents(question)
    final_prompt = build_prompt(question, docs)
    answer = generate_answer(final_prompt)
    return answer, docs
