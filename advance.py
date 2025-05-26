import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import sys
try:
    import fitz  # PyMuPDF
except ImportError:
    print("Please install PyMuPDF: pip install PyMuPDF")
    sys.exit(1)

# Load model for embedding
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load model for answer generation (optional: replace with OpenAI API or Llama)
qa_model = pipeline("text-generation", model="gpt2")

# Step 1: Extract text from PDF


def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Step 2: Chunk the text


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# Step 3: Embed and index


def build_faiss_index(chunks):
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings, chunks

# Step 4: Query processing


def query_index(question, index, chunks, top_k=3):
    question_vec = embedder.encode([question])
    distances, indices = index.search(np.array(question_vec), top_k)
    return [chunks[i] for i in indices[0]]

# Step 5: Generate Answer


def generate_answer(question, context_chunks):
    context = "\n".join(context_chunks)
    prompt = f"Answer the question based on the context:\n{context}\n\nQuestion: {question}\nAnswer:"
    answer = qa_model(prompt, max_length=300, do_sample=False)[
        0]['generated_text']
    return answer


# Main function
if __name__ == "__main__":
    # Load a research PDF
    pdf_path = "sample_research_paper.pdf"  # Change this to your PDF file
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

index, embeddings, chunks = build_faiss_index(chunks)

while True:
    question = input("\nAsk a question (type 'exit' to quit): ")
    if question.lower() == "exit":
        break
    top_chunks = query_index(question, index, chunks)
    answer = generate_answer(question, top_chunks)
    print("\nAnswer:\n", answer)
