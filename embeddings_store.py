import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "chunks.pkl"
FAISS_INDEX_FILE = "faiss_index.bin"
EMBEDDINGS_FILE = "embeddings.npy"

def load_chunks():
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    return chunks

def create_embeddings(chunks):
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    print(f"Embeddings shape: {embeddings.shape}")  # (num_chunks, 384)
    return embedding_model, embeddings

def save_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, FAISS_INDEX_FILE)
    np.save(EMBEDDINGS_FILE, embeddings)
    print(f"FAISS index saved to {FAISS_INDEX_FILE} with {faiss_index.ntotal} vectors.")
    return faiss_index

if __name__ == "__main__":
    chunks = load_chunks()
    model, embeddings = create_embeddings(chunks)
    save_faiss_index(embeddings)
