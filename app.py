import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import os, re

st.set_page_config(page_title="Exam bot", page_icon="🧠")


# Caching heavy setup (loading PDFs, embeddings and FAISS index) to avoid re-running every time
@st.cache_resource 

def set_exambot():

    # function for text processing
    def fix_text_formatting(text):
        text = re.sub(r'-\s*\n\s*', '', text)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        text = re.sub(r'[ ]{2,}', ' ', text)
        text = re.sub(r'\n{2,}', '\n\n', text)
        return text.strip()

    # Step - 1 : INDEXING  

    # 1-A : Document loading
    loader = DirectoryLoader(
        path='data',
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    clean_documents = [
        Document(page_content=fix_text_formatting(doc.page_content), metadata=doc.metadata)
        for doc in documents
    ]

    # print(documents)

    # 1-B : Text-splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )


    chunks = splitter.split_documents(clean_documents)

    # 1-C & 1-D : Embedding generation and Storing in vector stores
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    DB_FAISS_PATH = "vectorstore/db_faiss"

    if not os.path.exists(os.path.join(DB_FAISS_PATH, "index.faiss")):
        vector_store = FAISS.from_documents(chunks, embedding_model)
        vector_store.save_local(DB_FAISS_PATH)
    else:
        vector_store = FAISS.load_local(
            DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True
        )

    # Step - 2 : RETRIEVAL
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 7})

    llm = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",                 
    temperature=0.5,
    api_key=st.secrets["GROQ_API_KEY"]
    )

    return retriever, llm

retriever, llm = set_exambot()

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Exam bot", page_icon="🧠")
st.title("🧠 Exam bot - Your AI Exam Preparation Assistant")

# Display previous Chat Messages 
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        st.markdown(entry["answer"])

user_input = st.chat_input("Ask question...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream real-time answer
    with st.chat_message("assistant"):

        # Step - 3 : Augmentation
        context_docs = retriever.invoke(user_input)
        context = "\n\n".join(doc.page_content for doc in context_docs)

        prompt = dedent(f"""
            You are ExamBot, a helpful AI exam-preparation assistant.

            Your role:
            - Help the student study and understand concepts that may appear in exams.
            - Use the provided syllabus/content from the uploaded PDFs as the primary source when it is relevant.
            - If the answer is not clearly covered in the PDFs, you may use your own knowledge, but stay within the same subject/domain.

            Here is the relevant context from the uploaded syllabus/notes:
            {context}

            Below is the student's question:

            Question: {user_input}

            Instructions:
            - Give clear, structured, and exam-focused answers.
            - Explain concepts in simple terms first, then go deeper if needed.
            - When useful, include short examples, key points, or formulas.
            - If the question is clearly unrelated to studying or the subjects implied by the PDFs, respond:
            "I'm designed to help with exam preparation based on your syllabus and related subjects. Please ask a study-related question."

            Answer:
        """)


        # Step - 4 : GENERATION (Streaming Geneation)
        full_response = ""
        response_container = st.empty()
        for chunk in llm.stream(prompt):
            full_response += chunk.content
            response_container.markdown(full_response)

    st.session_state.chat_history.append({
        "question": user_input,
        "answer": full_response
    })