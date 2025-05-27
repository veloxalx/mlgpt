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
    resumes['text'].tolist(), convert_to_numpy=True)  # tolist converts the pandas Series to a list, convert_to_numpy=True converts the embeddings to numpy arrays
job_embeddings = model.encode(
    jobs['description'].tolist(), convert_to_numpy=True)  # tolist converts the pandas Series to a list, convert_to_numpy=True converts the embeddings to numpy arrays

# Build FAISS index
# get the first resume embedding vector , then the size of the first dimension
# d is assigned the length of the first resume embedding vector
d = resume_embeddings[0].shape[0]
index = faiss.IndexFlatL2(d)  # confusion lol
# add to index array of resume embeddings
index.add(np.array(resume_embeddings))

# Search: top 3 resumes for each job

for i, job_vec in enumerate(job_embeddings):
    # D is distances, I is indices of nearest neighbors
    D, I = index.search(np.array([job_vec]), k=3)
    print(f"\nTop 3 Resumes for Job {i+1}:\n")
    for idx in I[0]:  # I[0] contains indices of the top 3 resumes , idx is the index of the resume in the resumes DataFrame
        # iloc is used to access the row in the DataFrame by index
        print(f"- {resumes.iloc[idx]['text']}")
