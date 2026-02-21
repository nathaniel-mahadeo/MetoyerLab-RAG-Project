# rag_app.py
# This script executes the Retrieval-Augmented Generation (RAG) process
# combining PDFMinerLoader, REGEX cleaning, and aggressive filtering.

# --- 1. Imports (All stable core imports) ---
# 🔄 Using PDFMinerLoader (Pure Python, better text extraction)
from langchain_community.document_loaders import PDFMinerLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel 
from langchain_core.documents import Document # Added for type hinting and clarity
from langchain_core.retrievers import BaseRetriever # Added to properly type the retriever
import os 
import re 
from typing import List, Tuple # Added for type hinting

# --- 2. Configuration Variables and Key Assignment ---
DOCUMENT_PATH = "The-Design-of-EveryDAY-Things-Revised-and-Expanded-Edition.pdf"
CHROMA_DB_PATH = "./chroma_db_rag"
GENERATION_MODEL = "gemini-2.5-flash"
# 💡 FIX HERE: Changing the model to one that outputs 1536 dimensions
EMBEDDING_MODEL = "text-embedding-004" 
USER_QUERY = "What is the complexity of modern devices?"

# 🔑 HARDCODED API KEY (FOR IMMEDIATE TESTING ONLY)
API_KEY = "Your_API_key_here"

if not API_KEY:
    raise ValueError("API Key is missing from the script.")


# --- 3. Load and Chunk Your Data (Retrieval Setup) ---

print(f"1. Loading document: {DOCUMENT_PATH}")
try:
    # 🎯 Using PDFMinerLoader for a robust, purely Python-based parsing
    loader = PDFMinerLoader(DOCUMENT_PATH)
    documents = loader.load()
except FileNotFoundError:
    print(f"\nERROR: Could not find the file at {DOCUMENT_PATH}. Make sure the PDF is in this folder.")
    exit()

# 🧼 ULTIMATE CLEANING FUNCTION (FINAL, AGGRESSIVE VERSION)
def clean_document_text(documents: List[Document]) -> List[Document]:
    """
    Performs aggressive cleaning by keeping only alphanumeric characters and
    common punctuation, then collapsing all resulting excessive whitespace.
    """
    
    cleaned_documents = []
    
    for doc in documents:
        cleaned_content = doc.page_content
        
        # 1. Aggressively remove known junk strings (Case-insensitive)
        JUNK_PATTERNS = [
            r"9780465050659-text\.indd",
            r"THE DESIGN OF EVERYDAY THINGS",
            r"The Design of Everyday Things",
            r"8/19/13 5:22 PM",
            r"\d+\s+The Design of Everyday Things" # Handles page numbers like "266 The Design..."
        ]
        
        for pattern in JUNK_PATTERNS:
            cleaned_content = re.sub(pattern, " ", cleaned_content, flags=re.IGNORECASE).strip()

        # 2. COLLAPSE ALL EXCESSIVE WHITESPACE (spaces, tabs, newlines) into a single space
        cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
        
        # 3. Restore some minimal structure (add a newline after a period followed by a space)
        cleaned_content = cleaned_content.replace('. ', '.\n\n')
        
        doc.page_content = cleaned_content.strip()
        cleaned_documents.append(doc)
        
    return cleaned_documents

print("Applying FINAL text cleaning (Regex-based pattern removal and aggressive whitespace collapse)...")

documents = clean_document_text(documents) 


# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200, 
    length_function=len
)
chunks = text_splitter.split_documents(documents)

# 🧹 FINAL STEP: Filter out short/empty chunks
MIN_CHUNK_LENGTH = 500 
filtered_chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) > MIN_CHUNK_LENGTH]

print(f"Initial chunks: {len(chunks)}. Filtered chunks: {len(filtered_chunks)}.")
chunks = filtered_chunks 
print(f"Document loaded and split into {len(chunks)} meaningful chunks.")


# --- 4. Embed and Store Data (The Vector Database) ---

print(f"2. Generating embeddings and storing in {CHROMA_DB_PATH}...")
client_config = {'api_key': API_KEY}
embeddings_model = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL, 
    api_key=API_KEY, 
    client_options=client_config 
)

# 🆕 NEW DEBUG BLOCK: Display the embedded query vector
print(f"\n--- DEBUG: Embedded Query Vector for '{USER_QUERY}' ---")
query_vector = embeddings_model.embed_query(USER_QUERY)
# The Dimension (Length) here should now be 1536
print(f"Vector Dimension (Length): {len(query_vector)}")
print(f"Vector (First 10 elements): {[f'{x:.4f}' for x in query_vector[:10]]}") 
print("\nFull Query Vector (Limited Print):")
# Printing the full query vector (limited to 50 elements) to prevent console crash
MAX_QUERY_PRINT_ELEMENTS = 50
print(f"[{', '.join(f'{x:.4f}' for x in query_vector[:MAX_QUERY_PRINT_ELEMENTS])}, ... ({len(query_vector) - MAX_QUERY_PRINT_ELEMENTS} more elements)]")
print("--------------------------------------------------------------------------------")
# END NEW DEBUG BLOCK

# This step converts text chunks to vectors and stores them in the Chroma database
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory=CHROMA_DB_PATH
)
print("Vector store creation complete.")


# --- 5. Define the RAG Chain (Pure LCEL Structure) ---

print("\n3. Defining RAG retrieval and generation pipeline (Pure LCEL)...")

llm = ChatGoogleGenerativeAI(
    model=GENERATION_MODEL, 
    api_key=API_KEY, 
    client_options=client_config
)

# Using Max Marginal Relevance (MMR) for diverse results
retriever = vector_store.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 8, "fetch_k": 50} 
) 

PROMPT_TEMPLATE = """
Answer the question based on the following context. If you can't find the answer, just say that you don't know.

CONTEXT:
{context}

QUESTION: {input}
"""
prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

def format_docs(docs: List[Document]) -> str:
    """Formats the list of document objects into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

setup_and_retrieval = RunnableParallel(
    input=RunnablePassthrough(), 
    context=(lambda x: x["input"]) | retriever | format_docs,
)

rag_chain = setup_and_retrieval | prompt | llm

print("RAG chain defined using pure LCEL.")


# --- 6. Run the Query and Get the Final Answer ---

# Changed type hint to BaseRetriever for broader compatibility
def debug_retrieved_vectors(query: str, retriever: BaseRetriever, embeddings_model):
    """
    Retrieves documents using the retriever (MMR) and then calculates and prints 
    the vector for each retrieved document, limiting the printed output.
    
    This function uses the public `invoke` method to correctly handle internal 
    argument requirements like 'run_manager'.
    """
    print(f"\n--- DEBUG: Vectors for {retriever.search_kwargs['k']} Retrieved Chunks (MMR Search) ---")
    print("--------------------------------------------------------------------------------")
    
    # 1. Retrieve the relevant documents using the public .invoke() method (the robust fix)
    retrieved_documents: List[Document] = retriever.invoke(query)
    
    # 2. Extract the page content from the documents
    document_contents = [doc.page_content for doc in retrieved_documents]
    
    # 3. Embed the retrieved documents to get their vectors
    retrieved_vectors = embeddings_model.embed_documents(document_contents)
    
    # 4. Print the content snippet and the corresponding vector
    context_text = ""
    MAX_PRINT_ELEMENTS = 50 # Limiting the printed output to 50 elements
    
    for i, (doc, vector) in enumerate(zip(retrieved_documents, retrieved_vectors)):
        print(f"Chunk {i+1} Content (First 100 chars): {doc.page_content[:100]}...")
        # The Vector (First 10 elements) and Dimension will reflect the 1536 size here
        print(f"Vector (First 10 elements): {[f'{x:.4f}' for x in vector[:10]]}")
        
        # ⚠️ Fixed: Limiting the print of the full vector to prevent console overflow
        print("Full Vector (Limited Print):")
        print(f"[{', '.join(f'{x:.4f}' for x in vector[:MAX_PRINT_ELEMENTS])}, ... ({len(vector) - MAX_PRINT_ELEMENTS} more elements)]")
        
        print("-" * 20)
        context_text += doc.page_content + "\n\n"
        
    print("--------------------------------------------------------------------------------")
    return context_text

print(f"4. Running query: '{USER_QUERY}'")

# Execute the new vector debugging function
_ = debug_retrieved_vectors(USER_QUERY, retriever, embeddings_model)

# Execute the actual RAG chain
response = rag_chain.invoke({"input": USER_QUERY})

print("\n--- Model Response (Grounded in your PDF) ---")
print(response.content) 

# Optional: Cleanup
del vector_store
print("\n--- RAG Process Complete ---")
