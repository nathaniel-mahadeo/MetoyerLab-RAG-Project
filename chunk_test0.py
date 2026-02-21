# chunk_test.py
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
import os 
import re # <-- IMPORT for robust cleaning


# --- 2. Configuration Variables and Key Assignment ---
DOCUMENT_PATH = "The-Design-of-Everyday-Things-Revised-and-Expanded-Edition.pdf"
CHROMA_DB_PATH = "./chroma_db_rag"
GENERATION_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/embedding-001"
# ⬇️ ADJUSTMENT: Reverting the query back to the original, focusing on complexity.
USER_QUERY = "Explain how a lock-in constraint maps to the 7-stages of behavior and give a concrete example."

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
def clean_document_text(documents):
    """
    Performs aggressive cleaning by keeping only alphanumeric characters and
    common punctuation, then collapsing all resulting excessive whitespace.
    This effectively nukes any hidden formatting or junk characters.
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
        # This makes the remaining junk chunks very short.
        cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
        
        # 3. Restore some minimal structure (add a newline after a period followed by a space)
        # We replace the single space we created back with two newlines for better paragraph separation.
        cleaned_content = cleaned_content.replace('. ', '.\n\n')
        
        # Update the document object with the cleaned content
        doc.page_content = cleaned_content.strip()
        cleaned_documents.append(doc)
        
    return cleaned_documents

print("Applying FINAL text cleaning (Regex-based pattern removal and aggressive whitespace collapse)...")

# We don't pass a list of strings anymore; the cleaning function uses regex internally.
documents = clean_document_text(documents) 


# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200, 
    length_function=len
)
chunks = text_splitter.split_documents(documents)

# 🧹 FINAL STEP: Filter out short/empty chunks
# We use a very high MIN_CHUNK_LENGTH now that we are confident the junk is collapsed to 
# only a few characters. 
MIN_CHUNK_LENGTH = 500 # <-- EVEN MORE AGGRESSIVE FILTERING
filtered_chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) > MIN_CHUNK_LENGTH]

print(f"Initial chunks: {len(chunks)}. Filtered chunks: {len(filtered_chunks)}.")
chunks = filtered_chunks # Use the filtered list for embedding
print(f"Document loaded and split into {len(chunks)} meaningful chunks.")


# --- 4. Embed and Store Data (The Vector Database) ---

print(f"2. Generating embeddings and storing in {CHROMA_DB_PATH}...")
client_config = {'api_key': API_KEY}
embeddings_model = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL, 
    api_key=API_KEY, 
    client_options=client_config 
)

# This step converts text chunks to vectors and stores them in the Chroma database
# NOTE: This will overwrite the previous vector store with clean documents!
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

# ⬆️ ADJUSTMENT 3: Switching to Max Marginal Relevance (MMR) for diverse results
retriever = vector_store.as_retriever(
    search_type="mmr", # New search type for diversity
    search_kwargs={"k": 8, "fetch_k": 50} # k=8 documents, fetch_k=50 for diversity calculation
) 

PROMPT_TEMPLATE = """
Answer the question based on the following context. If you can't find the answer, just say that you don't know.

CONTEXT:
{context}

QUESTION: {input}
"""
prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

def format_docs(docs):
    """Formats the list of document objects into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

setup_and_retrieval = RunnableParallel(
    input=RunnablePassthrough(), 
    context=(lambda x: x["input"]) | retriever | format_docs,
)

rag_chain = setup_and_retrieval | prompt | llm

print("RAG chain defined using pure LCEL.")


# --- 6. Run the Query and Get the Final Answer ---

print(f"4. Running query: '{USER_QUERY}'")

# --- DEBUG RETRIEVAL: PULL THE RAW CONTEXT FOR INSPECTION ---
retrieved_context = setup_and_retrieval.invoke({"input": USER_QUERY})["context"]

# Note: The debug output will now show 8 chunks, and they should be diverse.
print("\n--- DEBUG: Retrieved Context Chunks (8 closest matches using MMR) ---")
print("--------------------------------------------------------------------------------")
print(retrieved_context) # Print the raw text passed to the LLM
print("--------------------------------------------------------------------------------")

response = rag_chain.invoke({"input": USER_QUERY})

print("\n--- Model Response (Grounded in your PDF) ---")
print(response.content) 

# Optional: Cleanup
del vector_store
print("\n--- RAG Process Complete ---")
