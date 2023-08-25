

import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import pickle
from dotenv import load_dotenv
import os

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

with st.sidebar:
    st.title('🤗💬 Mooiatti Chat App')
    st.markdown('''
    ## About
    This app is an AI chatbot for Mooiatti:
    - [Mooiatti](https://mooiatti.com/)
    - [Mooiatti Portal](https://mooiatti.me/)

    ## Example
    #### 한글
    - 1970년대 80년대 빈티지 의자를 추천해줘
    - 빨간색 의자, 심플한 디자인을 찾아줘
    - arne jacobsen 제품을 링크와 사진도 함께 보여줘
    #### 日本語
    - 木の椅子を探していますが、シンプルなデザインで3つおすすめしてください。 リンクも一緒
 
    ''')
    add_vertical_space(15)
    st.write('Made with ❤️ by Mooiatti')

def main():
    st.header('Chat with Mooiatti AI')
    st.write("모이아띠 AI 에게 물어보세요")

    load_dotenv()

    st.write("GLoading ...")
    loader = GoogleDriveLoader(
        folder_id="1xTSGtI0XdFfJeHqS2CBJ6gBH8-JjlP2j",
        recursive=False
    )
    docs = loader.load()
    st.write("Loading Done ...")

    st.write("RecursiveCharacterTextSplitter ...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, chunk_overlap=0, separators=[" ", ",", "\n"]
        )

    st.write("split_documents ...")
    texts = text_splitter.split_documents(docs)
    st.write("OpenAIEmbeddings ...")
    embeddings = OpenAIEmbeddings()
    st.write("vector db from_documents ...")
    db = Chroma.from_documents(texts, embeddings)
    st.write("as_retriever ...")
    retriever = db.as_retriever()

    st.write("ChatOpenAI ...")

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    st.write("RetrievalQA ...")
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    st.write("Ready ...")

    # Accept user questions/query
    query = st.text_input("Ask Mooiatti:")
    st.write(query)

    if query:
        response = qa.run(query)
        st.write(response)

if __name__ == '__main__':
    main()
