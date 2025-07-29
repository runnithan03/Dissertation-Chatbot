import pickle
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Files from previous stages
CHUNKS_FILE = "chunks.pkl"
FAISS_INDEX_FILE = "faiss_index.bin"
EMBEDDINGS_FILE = "embeddings.npy"

# 1. Load Chunks
def load_chunks():
    with open(CHUNKS_FILE, "rb") as f:
        return pickle.load(f)

# 2. Load Embedding Model
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# 3. Load FAISS Index
def load_faiss_index():
    return faiss.read_index(FAISS_INDEX_FILE)

# 4. Load LLM (Flan-T5-Large)
def load_llm():
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=llm_model, tokenizer=tokenizer)

# 5. Retrieve Chunks from FAISS
def retrieve_relevant_chunks(question, embedding_model, faiss_index, chunks, k=3):
    query_embedding = embedding_model.encode([question], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, k)
    return [chunks[idx] for idx in indices[0]]

# 6. Query RAG Pipeline
def query_rag_pipeline(question, embedding_model, faiss_index, chunks, llm_pipeline, k=3, max_tokens=150):
    retrieved_chunks = retrieve_relevant_chunks(question, embedding_model, faiss_index, chunks, k)
    context = "\n".join(retrieved_chunks)
    prompt = (
        f"Based on the context, answer the question in 3 sentences or less.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer only (do not repeat the context):"
    )
    response = llm_pipeline(prompt, max_new_tokens=max_tokens)
    return response[0]['generated_text'].strip()

if __name__ == "__main__":
    print("Loading components...")
    chunks = load_chunks()
    embedding_model = load_embedding_model()
    faiss_index = load_faiss_index()
    llm_pipeline = load_llm()

    test_question = "Explain how MRCE differs from RRRR in my dissertation."
    answer = query_rag_pipeline(test_question, embedding_model, faiss_index, chunks, llm_pipeline)
    print("Answer:")
    print(answer)