from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss

resumes = pd.DataFrame({
    'text': [
        "Experienced software engineer skilled in Java and Spring Boot.",
        "Data scientist with strong Python and machine learning background.",
        "Frontend developer with React and Next.js experience.",
        "DevOps engineer with experience in AWS and Docker.",
        "Full-stack developer with expertise in Node.js and MongoDB."
    ]
})

jobs = pd.Series([
    "Looking for a React developer",
    "Seeking an experienced Python data scientist",
    "Hiring a full-stack engineer with Node.js skills"
])

model = SentenceTransformer('all-MiniLM-L6-v2')
resume_embeddings = model.encode(resumes['text'].tolist())
job_embeddings = model.encode(jobs.tolist())

d = resume_embeddings[0].shape[0]
index = faiss.IndexFlatL2(d)
index.add(np.array(resume_embeddings))

k = 3
for i, job_vec in enumerate(job_embeddings):
    D, I = index.search(np.array([job_vec]), k)
    print(f"\nTop {k} resume matches for job: '{jobs[i]}'\n")
    for rank, idx in enumerate(I[0]):
        print(f"{rank+1}. {resumes.iloc[idx]['text']}")
