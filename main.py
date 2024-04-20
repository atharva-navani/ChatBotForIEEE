from fastapi import FastAPI, HTTPException
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langserve import add_routes
import uvicorn
import os
from dotenv import load_dotenv
import bs4
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI

# Load environment variables
# OPENAI_API_KEY = os.environ['MY_KEY']
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# Initialize FastAPI application
app = FastAPI(title="Chatbot Server",
              version="1.0",
              description="Simple API Server for Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # This allows requests from any origin
    allow_methods=["POST"],  # This allows POST requests
    allow_headers=["*"],
)

# Add routes for OpenAI
add_routes(app, ChatOpenAI(), path="/openai")

# Define function to load, chunk, and index the content of the HTML page
# def load_html_content(url):
#     loader = WebBaseLoader(web_paths=(url,),
#                            bs_kwargs=dict(parse_only=bs4.SoupStrainer(
#                                class_=("jsx-3247734209 jsx-3126643613 jsx-3477660809 row")
#                            )))
#     docs = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     documents = text_splitter.split_documents(docs)
#     return documents

# # Load and index the content of the HTML page
# documents = load_html_content("https://collegedunia.com/pune-colleges")
# db = Chroma.from_documents(documents, OpenAIEmbeddings())
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('IEEE.pdf')
docs = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=200)
documents = text_splitter.split_documents(docs)

# Initialize the chatbot model
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Define prompt for the chatbot
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
Think step by step before providing a detailed answer.
I will tip you $1000 if the user finds the answer helpful.
＜context＞
{context}
</context>
Question: {input}""")

# Create document chain
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain

document_chain = create_stuff_documents_chain(llm, prompt)
db = Chroma.from_documents(documents, OpenAIEmbeddings())

# Create retriever
retriever = db.as_retriever()

# Create retrieval chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)


# Define API endpoint for chatbot invocation
@app.post("/chatbot/invoke")
async def invoke_chatbot(input_data: dict):
  response = retrieval_chain.invoke(input_data)
  return {"output": response}


# Endpoint to generate the script tag
@app.get('/script-tag')
async def generate_script_tag():
  script_code = """
        <!-- Chatbot script -->
        <script src="http://localhost:3000/script.js"></script> <!-- Updated script source -->
        <!-- End of Chatbot script -->
    """
  return script_code


# Run the FastAPI application
if __name__ == "__main__":
  uvicorn.run(app, host="127.0.0.1", port=8000)
