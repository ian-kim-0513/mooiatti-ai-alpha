# streamlit_app.py

import streamlit as st
from google.oauth2 import service_account
from gsheetsdb import connect

# Create a connection object.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ],
)
conn = connect(credentials=credentials)


import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import pickle

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import GoogleDriveLoader
from langchain.document_loaders import UnstructuredFileIOLoader

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
    
    st.write("GLoading 2 ...")

    
    spreadsheet_id = "1UHH6ZJjA3BvYAP8Zx7BQw01FBJOzTbdAW3yg_o8uifs"
    loader = GoogleDriveLoader(file_ids=[spreadsheet_id])
    docs = loader._load_sheet_from_id(spreadsheet_id)
    st.write(loader)
    st.write(docs)


#    loader = GoogleDriveLoader(
#        folder_id="1x_Ze95L2lBfoojCA8tj6o56lnw0_-Hiy",
#        recursive=False
#    )
#    docs = loader.load()
    st.write("Loading Done2 ...")



if __name__ == '__main__':
    main()
