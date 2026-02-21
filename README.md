Gemini RAG: The Design of Everyday Things
This project implements a Retrieval-Augmented Generation (RAG) pipeline using Google Gemini and ChromaDB. 
It allows users to query the book The Design of Everyday Things (Revised and Expanded Edition) either in its entirety or by specific chapters, providing grounded answers and page citations.

Prerequisites
PDF Source: Ensure the file The-Design-of-Everyday-Things-Revised-and-Expanded-Edition.pdf is in the root directory.
Environment: Windows PowerShell.
Setup Instructions
Navigate to Project Folder:
PowerShell
cd your-project-folder-path


Install Required Libraries:
PowerShell
pip install langchain-google-genai langchain-chroma langchain-community langchain-text-splitters pypdf pdfminer.six


Initialize Virtual Environment:
PowerShell
python -m venv venv


Configure Execution Policy:
PowerShell
Set-ExecutionPolicy RemoteSigned -Scope Process


Activate Virtual Environment:
PowerShell
.\venv\Scripts\Activate.ps1


File Overview & Workflow
The files are listed in the order they were developed and utilized during the project lifecycle.

Phase 1: Verification and Preliminary Testing
test_api.py: A simple connectivity test to verify the Gemini API is functional.
Usage: python test_api.py
Expected Output: "Computers learn from data to make smart decisions."
chunk_test.py: Preliminary RAG script using PDFMinerLoader for robust text extraction and aggressive Regex-based cleaning.
Usage: python chunk_test.py

Phase 2: Data Preparation
chapter_map.py: Configuration file that maps physical PDF page numbers to chapters, including a page offset for front matter.
extract_chapters.py: A utility script that uses the data in chapter_map.py to slice the master PDF into individual chapter PDF files stored in data/chapters/.

Phase 3: Core RAG Implementation
rag_app.py: The main application script for basic RAG queries on the full book using PyPDFLoader to preserve page metadata.
rag_utils.py: The backend engine containing logic for cleaning documents, building the Gemini RAG pipeline, and calculating citations based on page offsets.
similarity_test.py: A debugging tool to inspect retrieved vector chunks and verify the embedding model's dimensions.

Phase 4: Final Interactive Tool
rag_script.py: The final frontend controller. It provides a terminal-based menu for selecting between the full book or specific chapters and maintains a chat loop for user questions.
Usage: python rag_script.py

Key Features
Multi-Level Querying: Toggle between querying the full book or isolated chapters.
Smart Cleaning: Custom Regex patterns remove recurring junk text (e.g., ISBNs, file headers) and collapse excessive whitespace.
Citation Support: Automatically calculates the original book page number for retrieved text, even when using chapter-specific slices.
Optimized Retrieval: Uses Max Marginal Relevance (MMR) for diverse search results and Google Gemini Flash 2.5 for efficient generation.

