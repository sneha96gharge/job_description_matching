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

# Extract specific columns from job description and resumes
job_description_experience = job_description_df.iloc[0, 0]
job_description_education = job_description_df.iloc[0, 1]
job_description_skill = job_description_df.iloc[0, 2]

resumes_experience = resumes_df.iloc[:, 0].tolist()
resumes_education = resumes_df.iloc[:, 1].tolist()
resumes_skill = resumes_df.iloc[:, 2].tolist()

# Function to compute BERT embeddings and cosine similarity
def compute_similarity(text1, texts2):
    tokens1 = tokenizer(text1, return_tensors='pt', padding=True, truncation=True)
    tokens2 = tokenizer(texts2, return_tensors='pt', padding=True, truncation=True)
    
    with torch.no_grad():
        embeddings1 = model(**tokens1).last_hidden_state.mean(dim=1)
        embeddings2 = model(**tokens2).last_hidden_state.mean(dim=1)
    
    similarities = cosine_similarity(embeddings1, embeddings2)
    return similarities.squeeze().tolist()

# Compute percentage match for each column
percentage_match_experience = compute_similarity(job_description_experience, resumes_experience)
percentage_match_education = compute_similarity(job_description_education, resumes_education)
percentage_match_skill = compute_similarity(job_description_skill, resumes_skill)

# Calculate total percentage match and break it down
for i, (exp, edu, skl) in enumerate(zip(percentage_match_experience, percentage_match_education, percentage_match_skill), 1):
    total_percentage = exp + edu + skl
    weight_exp = exp / total_percentage
    weight_edu = edu / total_percentage
    weight_skl = skl / total_percentage
    
    total_percentage_match = (exp * weight_exp) + (edu * weight_edu) + (skl * weight_skl)
    
    print(f"Resume {i}:")
    print(f"Total Percentage Match = {total_percentage_match * 100:.2f}%")
    print(f"Experience Percentage Match = {exp * 100:.2f}%")
    print(f"Education Percentage Match = {edu * 100:.2f}%")
    print(f"Skill Percentage Match = {skl * 100:.2f}%")
    print("-" * 50)
