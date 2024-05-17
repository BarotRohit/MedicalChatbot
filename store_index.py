from src.helper import load_pdf, text_split, download_hugging_face_embeddings
import chromadb
import chromadb.config
from langchain_community.vectorstores import Chroma


extracted_data = load_pdf("D:\MobileFirst\Test-OpenAI\MedicalChatbot\data")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()
vectorstore = Chroma("langchain_store", embeddings)
persist_directory = "db"
vectorstore.from_documents(documents=text_chunks,embedding=embeddings,persist_directory=persist_directory)