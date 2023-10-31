import streamlit as st
from dotenv import load_dotenv
from PyPDF import PdfReader

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text




def main():
    st.set_page_config(page_title="Q&A with pdfs",page_icon="📚")
    st.header("Q&A with pdfs :books:")
    st.text_input("Enter your question")
    with st.sidebar:
        st.subheader("Your Docs")
        pdf_docs = st.file_uploader("Upload your pdf",accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Wait for it..."):
                raw_text = get_pdf_text(pdf_docs)

                text_chunks = get_pdf_text(raw_text)



if __name__ == "__main__":
    main()
