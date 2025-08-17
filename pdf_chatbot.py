import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import io
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv



load_dotenv()

genai.configure(api_key=os.getenv("API_KEY"))


def get_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
       
        pdf_reader = PdfReader(io.BytesIO(pdf.read()))
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector(text_chunk):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunk, embedding=embeddings)
    vector_store.save_local('faiss_index')


def get_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the context, just say "Answer is not available in the context."

    Context:\n {context}\n
    Question:\n{question}\n

    Answer:
    """
    #model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    #new_db = FAISS.load_local("faiss_index", embeddings)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)
    chain = get_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply:", response["output_text"])

def main():
    st.set_page_config("Chat with Multiple PDFs")
    st.header("Chat with Multiple PDFs ")

    user_question = st.text_input("Ask a question")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("MENU")
       
        pdf_docs = st.file_uploader("Upload PDF files and click submit", type=["pdf"], accept_multiple_files=True)
        if st.button("SUBMIT"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_text(pdf_docs)
                    text_chunks = get_chunks(raw_text)
                    get_vector(text_chunks)
                    st.success("Done!")

if __name__ == "__main__":
    main()
