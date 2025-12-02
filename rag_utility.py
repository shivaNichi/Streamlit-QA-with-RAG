import os
from dotenv import load_dotenv

# SAFE PDF loader for Streamlit Cloud (no libGL)
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# LangChain Core (LCEL)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Embeddings
embedding = HuggingFaceEmbeddings()

# Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


def process_document_to_chroma_db(file_name):

    # Use PyPDFLoader (safe, pure Python)
    loader = PyPDFLoader(f"{working_dir}/{file_name}")
    docs = loader.load()

    # Splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    # Save vectors to Chroma DB
    Chroma.from_documents(
        chunks,
        embedding,
        persist_directory=f"{working_dir}/doc_vectorstore"
    )

    return 0


def answer_question(user_question):

    # Load existing vector DB
    vectorstore = Chroma(
        persist_directory=f"{working_dir}/doc_vectorstore",
        embedding_function=embedding
    )

    retriever = vectorstore.as_retriever()

    # Prompt template
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use ONLY the context to answer the question.
If answer is not in the document, reply:
"I could not find that information in the document."

Context:
{context}

Question: {question}
""")

    # LCEL chain
    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(user_question)
