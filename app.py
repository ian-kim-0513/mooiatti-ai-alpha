import streamlit as st
# from streamlit_extras.add_vertical_space import add_vertical_space
# from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle
from dotenv import load_dotenv
import os
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

with st.sidebar:
    st.title('Mooiatti Chat App (alpha)')
    st.markdown('''
    ## About
    This app is an AI chatbot for Mooiatti:
    - [Mooiatti](https://mooiatti.com/)
    - [Mooiatti Dealer's Portal](https://mooiatti.me/)
 
    ''')
    # add_vertical_space(15)
    st.write('Made with ❤️ by Mooiatti')

def main():
    st.header('Chat with Mooiatti AI')
    st.write("hello")

    load_dotenv()

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    
    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # st.write(text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)

        # st.write(chunks)

        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)
 
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            st.write('Embeddings Loaded from the Disk')
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
            st.write('Embeddings Computation Completed')


        # Accept user questions/query
        query = st.text_input("Ask Mooiatti:")
        st.write(query)
  
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            # llm = OpenAI()
            llm = OpenAI(model_name='gpt-3.5-turbo')
            # llm = OpenAI(model_name='gpt-3.5-turbo-16k')
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)

if __name__ == '__main__':
    main()
