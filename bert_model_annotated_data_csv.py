import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Read job description and resumes from CSV files, skipping the headers
job_description_df = pd.read_csv("/home/sneha/Job_description_1.csv", header=0)
resumes_df = pd.read_csv("/home/sneha/resumes_1.csv", header=0)

# Construct job description string from the relevant columns
Experience = job_description_df.iloc[0, 0]
Education = job_description_df.iloc[0, 1]
Skill = job_description_df.iloc[0, 2]

# Combine the relevant columns to form the job description string
job_description = f"Experience: {Experience}\nEducation: {Education}\nSkill: {Skill}"

# Combine relevant columns to form resumes strings
resumes = resumes_df.apply(lambda row: f"Experience: {row[0]}\nEducation: {row[1]}\nSkill: {row[2]}", axis=1).tolist()

# Encode job description and resumes using BERT tokenizer
job_description_tokens = tokenizer(job_description, return_tensors='pt', padding=True, truncation=True)
resume_tokens = tokenizer(resumes, return_tensors='pt', padding=True, truncation=True)

# Get BERT embeddings for job description and resumes
with torch.no_grad():
    job_description_embedding = model(**job_description_tokens).last_hidden_state.mean(dim=1)
    resume_embeddings = model(**resume_tokens).last_hidden_state.mean(dim=1)

# Compute cosine similarity between job description and resumes
similarities = cosine_similarity(job_description_embedding, resume_embeddings)

# Calculate percentage match
percentage_matches = similarities[0] * 100

# Rank resumes based on percentage matches
ranked_resumes = sorted(zip(resumes, percentage_matches), key=lambda x: x[1], reverse=True)

# Print ranked resumes
for rank, (resume, percentage_match) in enumerate(ranked_resumes, 1):
    print(f"Rank {rank}: Percentage Match = {percentage_match:.2f}%")
    print(resume)
    print("-" * 50)
