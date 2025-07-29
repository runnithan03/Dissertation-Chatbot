import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle

# Folder containing your PDFs
DATA_DIR = "docs"
CHUNKS_FILE = "chunks.pkl"

def extract_text_from_pdfs(folder_path=DATA_DIR):
    """
    Reads all PDF files in the given folder and extracts their text.
    """
    all_text = ""
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            print(f"Reading: {pdf_path}")
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"
    return all_text

def create_chunks(chunk_size=500, chunk_overlap=100):
    """
    Extract text and split it into overlapping chunks.
    """
    raw_text = extract_text_from_pdfs()
    print(f"Total characters extracted: {len(raw_text)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(raw_text)
    print(f"Number of chunks: {len(chunks)}")

    # Save chunks to file for later use
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Chunks saved to {CHUNKS_FILE}")
    return chunks

if __name__ == "__main__":
    chunks = create_chunks()
    print("Sample chunk:", chunks[0])