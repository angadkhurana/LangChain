import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import  ChatOpenAI
import tiktoken

from dotenv import load_dotenv, find_dotenv
import chromadb

load_dotenv(find_dotenv())

def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=500,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructE  mbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=False
    )
    return conversation_chain

def main():
    st.set_page_config(page_title="Q&A with pdfs",page_icon="📚")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Q&A with pdfs :books:")
    # prompt = st.chat_input("Enter your question")

    with st.chat_message("Assistant"):
        st.write("Hello 👋, Please upload the pdfs")

    prompt = st.chat_input("Ask a question")

    with st.sidebar:
        st.subheader("Your Docs")
        pdf_docs = st.file_uploader("Upload your pdf",accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Wait for it..."):
                raw_text = get_pdf_text(pdf_docs)

                text_chunks = get_text_chunks(raw_text)

                vectorstore = get_vectorstore(text_chunks)

                st.session_state.conversation = get_conversation_chain(vectorstore)


    if prompt:
        with st.chat_message("User"):
            st.write(prompt)

        with st.chat_message("Assistant"):
            result = st.session_state.conversation({"question":prompt})
            st.write(result["answer"])
    # st.write("qwertyui")
            

if __name__ == "__main__":
    main()
