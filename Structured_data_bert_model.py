import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read job description and resume from CSV files, skipping the headers
job_description_df = pd.read_csv("/home/sneha/Job_description_1.csv", header=0)
resume_df = pd.read_csv("/home/sneha/resumes_1.csv", header=0)

# Construct job description string from the relevant columns
Experience = job_description_df.iloc[0, 0]
Education = job_description_df.iloc[0, 1]
Skill = job_description_df.iloc[0, 2]

# Combine the relevant columns to form the job description string
job_description = f"Experience: {Experience}\nEducation: {Education}\nSkill: {Skill}"

# Combine relevant columns to form the resume string
resume = f"Experience: {resume_df.iloc[0, 0]}\nEducation: {resume_df.iloc[0, 1]}\nSkill: {resume_df.iloc[0, 2]}"

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform job description and resume into TF-IDF vectors
job_description_tfidf = vectorizer.fit_transform([job_description])
resume_tfidf = vectorizer.transform([resume])

# Calculate cosine similarity between job description and resume
total_similarity = cosine_similarity(job_description_tfidf, resume_tfidf)[0][0] * 100

# Extract TF-IDF vectors for Experience, Education, and Skill from the resume
experience_idx = vectorizer.vocabulary_.get('experience', -1)
education_idx = vectorizer.vocabulary_.get('education', -1)
skill_idx = vectorizer.vocabulary_.get('skill', -1)

# Reshape TF-IDF vectors using np.repeat
def reshape_tfidf(tfidf_matrix, vocab_idx):
    if vocab_idx != -1:
        tfidf_array = tfidf_matrix[:, vocab_idx].toarray()
        if tfidf_array.size > 1:
            reshaped_array = np.repeat(tfidf_array, 45).reshape(1, 45)
        else:
            reshaped_array = tfidf_array.reshape(1, 1)
    else:
        reshaped_array = None
    return reshaped_array

experience_tfidf_reshaped = reshape_tfidf(resume_tfidf, experience_idx)
education_tfidf_reshaped = reshape_tfidf(resume_tfidf, education_idx)
skill_tfidf_reshaped = reshape_tfidf(resume_tfidf, skill_idx)

# Calculate individual percentage matches for Experience, Education, and Skill
percentage_match_experience = cosine_similarity(job_description_tfidf, experience_tfidf_reshaped)[0][0] * 100 if experience_tfidf_reshaped is not None else 0
percentage_match_education = cosine_similarity(job_description_tfidf, education_tfidf_reshaped)[0][0] * 100 if education_tfidf_reshaped is not None else 0
percentage_match_skill = cosine_similarity(job_description_tfidf, skill_tfidf_reshaped)[0][0] * 100 if skill_tfidf_reshaped is not None else 0

# Calculate breakdown percentages from the total similarity
breakdown_percentage_experience = (percentage_match_experience / total_similarity) * 100 if total_similarity > 0 else 0
breakdown_percentage_education = (percentage_match_education / total_similarity) * 100 if total_similarity > 0 else 0
breakdown_percentage_skill = (percentage_match_skill / total_similarity) * 100 if total_similarity > 0 else 0

# Print results
print(f"Total Percentage Match = {total_similarity:.2f}%")
print(f"Breakdown Percentage for Experience = {breakdown_percentage_experience:.2f}%")
print(f"Breakdown Percentage for Education = {breakdown_percentage_education:.2f}%")
print(f"Breakdown Percentage for Skill = {breakdown_percentage_skill:.2f}%")
