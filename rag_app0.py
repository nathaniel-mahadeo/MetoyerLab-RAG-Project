# rag_app.py
# This script executes the Retrieval-Augmented Generation (RAG) process
# using LangChain, Google Gemini, ChromaDB, and your PDF document.

# --- 1. Imports (All stable core imports) ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
# Ensure RunnableParallel is imported for the final, correct LCEL structure
from langchain_core.runnables import RunnablePassthrough, RunnableParallel 
import os 


# --- 2. Configuration Variables and Key Assignment ---
DOCUMENT_PATH = "The-Design-of-Everyday-Things-Revised-and-Expanded-Edition.pdf"
CHROMA_DB_PATH = "./chroma_db_rag"
GENERATION_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/embedding-001"
USER_QUERY = "What is the complexity of modern devices?"

# 🔑 HARDCODED API KEY (FOR IMMEDIATE TESTING ONLY)
# This bypasses all environment variable issues and uses your confirmed-working key.
API_KEY = "Your_API_key_here"

if not API_KEY:
    raise ValueError("API Key is missing from the script.")


# --- 3. Load and Chunk Your Data (Retrieval Setup) ---

print(f"1. Loading document: {DOCUMENT_PATH}")
try:
    loader = PyPDFLoader(DOCUMENT_PATH)
    documents = loader.load()
except FileNotFoundError:
    print(f"\nERROR: Could not find the file at {DOCUMENT_PATH}. Make sure the PDF is in this folder.")
    exit()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200, 
    length_function=len
)
chunks = text_splitter.split_documents(documents)
print(f"Document loaded and split into {len(chunks)} chunks.")


# --- 4. Embed and Store Data (The Vector Database) ---

print(f"2. Generating embeddings and storing in {CHROMA_DB_PATH}...")
# FIX: Pass api_key via client_options to force authentication
client_config = {'api_key': API_KEY}
embeddings_model = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL, 
    api_key=API_KEY, 
    client_options=client_config  # <-- FORCES AUTHENTICATION
)

# This step converts text chunks to vectors and stores them in the Chroma database
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory=CHROMA_DB_PATH
)
print("Vector store creation complete.")


# --- 5. Define the RAG Chain (Pure LCEL Structure) ---

print("\n3. Defining RAG retrieval and generation pipeline (Pure LCEL)...")

# FIX: Pass api_key via client_options to force authentication
llm = ChatGoogleGenerativeAI(
    model=GENERATION_MODEL, 
    api_key=API_KEY, 
    client_options=client_config # <-- FORCES AUTHENTICATION
)

# Convert the Vector Store into a Retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Define the Prompt Template
PROMPT_TEMPLATE = """
You are an expert Q&A system. Your goal is to answer the question based ONLY on the provided CONTEXT.
If the answer cannot be found in the context, state that you cannot find the answer in the document.

CONTEXT:
{context}

QUESTION: {input}
"""
prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# Define the 'Stuffing' (Document Formatting) logic as a Python function
def format_docs(docs):
    """Formats the list of document objects into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# --- FINAL CORRECTED RAG CHAIN DEFINITION ---
# This structure fixes the TypeError by isolating the query string.
setup_and_retrieval = RunnableParallel(
    # 'input' key gets the original query string (from the invoke call)
    input=RunnablePassthrough(), 
    # 'context' key extracts the query string (x["input"]), passes it to retriever, and formats docs
    context=(lambda x: x["input"]) | retriever | format_docs,
)

# Combine the parallel steps with the Prompt and LLM
rag_chain = setup_and_retrieval | prompt | llm

print("RAG chain defined using pure LCEL.")


# --- 6. Run the Query and Get the Final Answer ---

print(f"4. Running query: '{USER_QUERY}'")
# Note: The query is passed as a dictionary with the key "input" to match the chain structure.
response = rag_chain.invoke({"input": USER_QUERY})

print("\n--- Model Response (Grounded in your PDF) ---")
print(response.content) 

# Optional: Cleanup
del vector_store
print("\n--- RAG Process Complete ---")