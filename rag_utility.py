import os
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# NEW imports for LangChain 1.1+
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

working_dir = os.path.dirname(os.path.abspath(__file__))

embedding = HuggingFaceEmbeddings()

# Load the Llama-3.3-70B model from Groq
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


def process_document_to_chroma_db(file_name):

    loader = UnstructuredPDFLoader(f"{working_dir}/{file_name}")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    Chroma.from_documents(
        chunks,
        embedding,
        persist_directory=f"{working_dir}/doc_vectorstore"
    )

    return 0


def answer_question(user_question):
    # Load the persistent Chroma vector database
    vectorstore = Chroma(
        persist_directory=f"{working_dir}/doc_vectorstore",
        embedding_function=embedding
    )
    # Create a retriever for document search
    retriever = vectorstore.as_retriever()

    # We do retrieval using LCEL (LangChain Expression Language)
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use ONLY the context to answer the question.
If the answer is not found in the context, say:
"I could not find that information in the document."

Context:
{context}

Question: {question}
""")

    # LCEL pipeline (official new API)
    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
#  # Create a RetrievalQA chain to answer user questions using Llama-3.3-70B
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#     )
    answer = chain.invoke(user_question)
    return answer
