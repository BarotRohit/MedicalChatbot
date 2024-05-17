from flask import Flask, render_template, jsonify, request
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
import chromadb
import chromadb.config
from langchain_community.vectorstores import Chroma

app = Flask(__name__)



extracted_data = load_pdf("D:\MobileFirst\Test-OpenAI\MedicalChatbot\data")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()
vectorstore = Chroma("langchain_store", embeddings)
persist_directory = "db"
vectorstore.from_documents(documents=text_chunks,embedding=embeddings,persist_directory=persist_directory)
vectorstore.persist()
vectorstore = None
# Now we can load the persisted database from disk, and use it as normal.
vectorstore = Chroma(persist_directory=persist_directory,embedding_function=embeddings)
docsearch = vectorstore.as_retriever()

# query = input(f"Input Prompt:")
# docs = docsearch.get_relevant_documents(query)

# print(docs[0].page_content)


# PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# chain_type_kwargs={"prompt": PROMPT}

# llm=CTransformers(model="D:\MobileFirst\Test-OpenAI\MedicalChatbot\model\llama-2-7b-chat.ggmlv3.q4_0.bin",
#                   model_type="llama",
#                   config={'max_new_tokens':512,
#                           'temperature':0.8})

# qa=RetrievalQA.from_chain_type(
#     llm=llm, 
#     chain_type="stuff", 
#     retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
#     return_source_documents=True, 
#     chain_type_kwargs=chain_type_kwargs)

@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    docs = docsearch.get_relevant_documents(input)
    print("Response : ", docs[0].page_content["result"])
    return str(input["result"])



if __name__ == '__main__':
    app.run(debug= True)