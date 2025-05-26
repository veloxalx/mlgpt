import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Sample Data
resumes = pd.DataFrame({
    'text': [
        "Experienced software engineer with Python and ML expertise.",
        "Data scientist skilled in NLP, deep learning, and PyTorch.",
        "Frontend developer with React and UI/UX design experience.",
        "DevOps engineer with AWS, Docker, and CI/CD pipeline experience.",
        "Business analyst with skills in SQL, Tableau, and Excel."
    ]
})

jobs = pd.DataFrame({
    'text': [
        "Looking for a machine learning engineer with Python experience.",
        "Need a frontend developer who is strong in React and design.",
        "Seeking a cloud engineer with DevOps skills and AWS experience."
    ]
})

# Encode using SBERT
model = SentenceTransformer('all-MiniLM-L6-v2')
resume_embeddings = model.encode(resumes['text'].tolist())
job_embeddings = model.encode(jobs['text'].tolist())

# Build FAISS index (CONFUSION LOL)
d = resume_embeddings[0].shape[0]  # confusion lol
index = faiss.IndexFlatL2(d)  # confusion lol
index.add(np.array(resume_embeddings))

# Search: top 3 resumes per job
for i, job_vec in enumerate(job_embeddings):
    D, I = index.search(np.array([job_vec]), k=3)
    print(f"\nTop 3 Resumes for Job {i+1}:\n")
    for idx in I[0]:
        print(f"- {resumes.iloc[idx]['text']}")
