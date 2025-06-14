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
})  # text column contains the resume texts

jobs = pd.DataFrame({
    'text': [
        "Looking for a machine learning engineer with Python experience.",
        "Need a frontend developer who is strong in React and design.",
        "Seeking a cloud engineer with DevOps skills and AWS experience."
    ]
})  # text column contains the job descriptions

# Encode using SBERT
model = SentenceTransformer('all-MiniLM-L6-v2')
# tolist converts the pandas Series to a list
resume_embeddings = model.encode(resumes['text'].tolist())
# tolist converts the pandas Series to a list
job_embeddings = model.encode(jobs['text'].tolist())

# Build FAISS index
# get the first resume embedding vector , then the size of the first dimension
# d is assigned the length of the first resume embedding vector
d = resume_embeddings[0].shape[0]
# IndexFlatL2 is used for L2 distance (Euclidean distance) search
index = faiss.IndexFlatL2(d)
# add to index array of resume embeddings
index.add(np.array(resume_embeddings))

# Search: top 3 resumes per job
for i, job_vec in enumerate(job_embeddings):
    # D is distances, I is indices of nearest neighbors
    # index.search takes a numpy array of job_vec and k is the number of nearest neighbors to find
    D, I = index.search(np.array([job_vec]), k=3)
    print(f"\nTop 3 Resumes for Job {i+1}:\n")
    for idx in I[0]:  # I[0] contains indices of the top 3 resumes , idx is the index of the resume in the resumes DataFrame
        # iloc is used to access the row in the DataFrame by index
        print(f"- {resumes.iloc[idx]['text']}")
