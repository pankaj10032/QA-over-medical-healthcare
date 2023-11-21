from flask import Flask, render_template, request
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
import os

app = Flask(__name__)
url="https://www.cranberry.fit/post/ovulation-pain-unmasking-the-mystery"
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/answer', methods=['POST'])
def answer():
    os.environ['OPENAI_API_KEY'] = request.form['openai_api_key']
    
    if not os.environ['OPENAI_API_KEY']:
        return 'Please insert OpenAI API Key. Instructions [here]'

    llm = load_LLM()

    loader = WebBaseLoader(url)
    text = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=50,
    )

    splitted_text = text_splitter.split_documents(text)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splitted_text, embeddings)

    chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)

    chat_history = []

    text_input = request.form['text_input']
    result = chain({"question": text_input, "chat_history": chat_history})
    return f"Answer: {result['answer']}"

def load_LLM():
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.7)
    return llm

if __name__ == '__main__':
    app.run(debug=True)
