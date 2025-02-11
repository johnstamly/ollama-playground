import streamlit as st
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

pdfs_directory = "pdfs/"

embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
vector_store = InMemoryVectorStore(embeddings)

model = OllamaLLM(model="deepseek-r1:7b", num_ctx=16384, num_predict=-2)

def upload_pdf(file):
    # Ensure the target directory exists
    if not os.path.exists(pdfs_directory):
        os.makedirs(pdfs_directory)
    
    # Build the full path using os.path.join
    file_path = os.path.join(pdfs_directory, file.name)
    
    # Write the file's content from the in-memory buffer to disk
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    
    # Return the file path so you can use it later
    return file_path

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()

    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        add_start_index=True
    )

    return text_splitter.split_documents(documents)

def index_docs(documents):
    vector_store.add_documents(documents)

def retrieve_docs(query):
    return vector_store.similarity_search(query)

def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    return chain.stream({"question": question, "context": context})

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)

if uploaded_file:
    # Save the file and retrieve its full path
    file_path = upload_pdf(uploaded_file)
    
    # Now load the PDF from the saved file location
    documents = load_pdf(file_path)
    chunked_documents = split_text(documents)
    index_docs(chunked_documents)
    
    question = st.chat_input()
    if question:
        st.chat_message("user").write(question)
        related_documents = retrieve_docs(question)
        answer = answer_question(question, related_documents)
        st.chat_message("assistant").write(answer)

        # Streaming response
        message_placeholder = st.chat_message("assistant").empty()
        response = ""
        for chunk in answer_question(question, related_documents):
            response += chunk
            message_placeholder.write(response)  # Updates dynamically
