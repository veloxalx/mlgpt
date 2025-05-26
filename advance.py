# Expert-level optimized RAG system for querying PDFs using FAISS + LLMs

import os
import numpy as np
import faiss
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
from typing import List

# --- Configurations ---
PDF_PATH = "sample_research_paper.pdf"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 5
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
CROSS_ENCODER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
LLM_NAME = "gpt2"  # Swap with OpenAI API or LLaMA model for prod

# --- Load models ---
print("[INFO] Loading models...")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
reranker = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
qa_model = pipeline("text-generation", model=LLM_NAME)

# --- Step 1: Extract text from PDF ---


def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

# --- Step 2: Chunk text ---


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk.strip())
    return chunks

# --- Step 3: Embed and Index chunks ---


def build_faiss_index(chunks: List[str]):
    embeddings = embedder.encode(
        chunks, convert_to_numpy=True, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings, chunks

# --- Step 4: Retrieve with FAISS and rerank with CrossEncoder ---


def retrieve_chunks(query: str, index, chunks: List[str], top_k: int) -> List[str]:
    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(
        query_vec, top_k * 3)  # Get more for reranking
    candidates = [chunks[i] for i in indices[0]]

    # Rerank
    scores = reranker.predict([(query, chunk) for chunk in candidates])
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [candidates[i] for i in top_indices]

# --- Step 5: Generate Answer with LLM ---


def generate_answer(query: str, context_chunks: List[str]) -> str:
    context = "\n".join(context_chunks)
    prompt = f"""
Answer the question using the provided context:

Context:
{context}

Question: {query}
Answer:
"""
    result = qa_model(prompt, max_length=300, do_sample=False)[
        0]['generated_text']
    return result[len(prompt):].strip()


# --- Main entry point ---
if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"[ERROR] PDF file not found at: {PDF_PATH}")
        exit(1)
        
    print("[INFO] Reading and processing PDF...")
    text = extract_text_from_pdf(PDF_PATH)
    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    index, embeddings, chunks = build_faiss_index(chunks)

    print("[READY] System initialized. Ask questions about the document.")
    while True:
        query = input("\nQuestion (type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        top_chunks = retrieve_chunks(query, index, chunks, top_k=TOP_K)
        answer = generate_answer(query, top_chunks)
        print("\nAnswer:\n", answer)
