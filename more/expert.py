import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load data
resumes = pd.read_csv("resumes.csv")  # Should contain a 'text' column
jobs = pd.read_csv("jobs.csv")  # Should contain a 'description' column

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
resume_embeddings = model.encode(
    resumes['text'].tolist(), convert_to_numpy=True)
job_embeddings = model.encode(
    jobs['description'].tolist(), convert_to_numpy=True)

# Build FAISS index
d = resume_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(np.array(resume_embeddings))

# Search: top 3 resumes for each job
k = 3
for i, job_vec in enumerate(job_embeddings):
    D, I = index.search(np.array([job_vec]), k)
    print(f"\nTop {k} Resumes for Job {i+1}:\n")
    for idx in I[0]:
        print(f"- Resume #{idx+1}: {resumes.iloc[idx]['text'][:200]}...\n")
