import streamlit as st

import os

#helps to read all the pdf data
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter

import os

#provides embedding technique (vector )

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

#FAISS is for vector embedding

from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

#it helps us to do any kind of chat 
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()

#to load the api keys 
genai.configure(os.env(api_key="GOOGLE_API_KEY"))

#when pdf is uploaded we should be able to read the pdf and whatever data 


# we read pdf and we go through each and every pages and extract text 
# def pdf_text(pdf_docs):
#     text=""
#     for pdf in pdf_docs:
#         #pdf_file=io.BytesIO(pdf_docs)
#         pdf_reader=PdfReader(pdf)
#               # all pdf reader read in form of list 
#       #as soons as pdf_reader read it will be able to get the details of all the pages
#         for page in pdf_reader.pages:
#             text+=page.extract_text()
#     return text  


def pdf_text(pdf_bytes):
    text=""
    # Wrap the bytes object in a BytesIO object
    for pdf in pdf_bytes:
        pdf_reader = PdfReader(pdf)
        
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text
       

#dividing into particular size of 10,000 tokens
def get_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks

#converting chunks into vectors 
def get_vector(text_chunks):
    #in langchain different embedding  and in openai also different embedding 
    embeddings=GoogleGenerativeAIEmbeddings(model='models/embedding-004',google_api_key='GOOGLE_API_KEY')
    
    #takes all the texture and embed according to this embedding that i have initialize
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)

    #this can be save in database or in local environment
    vector_store.save_local("faiss_index")

    #faiss_index will be the folder that will be created and inside this can see my vectors 

    #got pdf--->converted into chunks--->divided into vectors

def conversation_chain():
    prompt_template="""
Answer the question as details as possible .If Answer is not 
availabel just say answer is not availabel in the context .

Context:\n {context}?\n
Question:\n {question}\n


Answer:"""
    #we use the gemini pro so initialize my model chat generative chat 
    model=ChatGoogleGenerativeAI(model='gemini-pro',temperature=0.3)
    
    prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain


def user_input(user_question):
   

    #local faiss_index from local 
    #pdf is already converted into vectors so already store in the faiss index
    embeddings=GoogleGenerativeAIEmbeddings(model='models/embedding-004',google_api_key='GOOGLE_API_KEY')

    new_db=FAISS.load_local('faiss_index',embeddings)

#similarity search based on the user question 
    docs=new_db.similarity_search(user_question)

    chain=conversation_chain()

    response=chain(
        {"input_documents":docs,"question":user_question},
         return_only_outputs=True)
    print(response)
    st.write("Reply: ",response["output_text"])


def main():
    st.set_page_config("Chat with Multiple PDF")

    st.header=("Chat with PDF using Gemini")

    user_question=st.text_input("Ask a Question from the pdf files")

    #After i gave question below function should be executed automatically

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs=st.file_uploader("UPload your PDF Files and Click on the ",accept_multiple_files=True,type=["pdf"])

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text=pdf_text(pdf_docs)
                text_chunks=get_chunks(raw_text)
                get_vector(text_chunks)
                st.success("Done")

if __name__=="__main__":
    main()
# main()