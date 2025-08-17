## Overview
Chat with multiple PDFs using semantic search. Ask questions, and get answers based only on the uploaded PDFs.

## Features:
Upload multiple PDF files
Extract and chunk text
Generate embeddings and store in FAISS
Answer questions using Google Gemini LLM

## Setups:
git clone <repo_url>
cd <repo_folder>
pip install -r requirements.txt
( Add your API key to .env)

## To run 
streamlit run app.py

## Workflow:
Upload PDFs → extract text → split into chunks
Generate embeddings → store in FAISS
Ask question → similarity search → QA model returns answer

### Notes:
If answer is not in PDFs, bot responds: "Answer is not available in the context."
FAISS index can be rebuilt anytime by re-uploading PDFs.
