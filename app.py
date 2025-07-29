import pickle
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import gradio as gr

# File paths
CHUNKS_FILE = "chunks.pkl"
FAISS_INDEX_FILE = "faiss_index.bin"
EMBEDDINGS_FILE = "embeddings.npy"

# --- Load RAG Components ---
def load_rag_components():
    # Load chunks
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)

    # Load embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load FAISS index
    faiss_index = faiss.read_index(FAISS_INDEX_FILE)

    # Load LLM
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    llm_pipeline = pipeline("text2text-generation", model=llm_model, tokenizer=tokenizer)

    return chunks, embedding_model, faiss_index, llm_pipeline


# --- RAG Query Pipeline ---
def retrieve_relevant_chunks(question, embedding_model, faiss_index, chunks, k=3):
    query_embedding = embedding_model.encode([question], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, k)
    return [chunks[idx] for idx in indices[0]]

def query_rag_pipeline(question, chunks, embedding_model, faiss_index, llm_pipeline, k=3, max_tokens=150):
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


# --- Chatbot Function ---
def rag_chat(question, history):
    return query_rag_pipeline(question, chunks, embedding_model, faiss_index, llm_pipeline)


# --- Launch Gradio App ---
chunks, embedding_model, faiss_index, llm_pipeline = load_rag_components()

chatbot = gr.ChatInterface(
    fn=rag_chat,
    title="📘 Dissertation Q&A Assistant",
    description="Ask questions about my dissertation and referenced research papers.",
    examples=[
        "Explain how MRCE differs from RRRR.",
        "Summarise Chapter 3 in 3 sentences.",
        "What is the main finding of the equity fund analysis?"
    ]
)

if __name__ == "__main__":
    chatbot.launch(share=True)