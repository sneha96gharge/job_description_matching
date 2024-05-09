import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform job description and resumes into TF-IDF vectors
job_description_tfidf = vectorizer.fit_transform([job_description])
resume_tfidfs = vectorizer.transform(resumes)

# Compute cosine similarity between job description and resumes
similarities = cosine_similarity(job_description_tfidf, resume_tfidfs)

# Calculate percentage match
percentage_matches = similarities[0] * 100

# Rank resumes based on percentage matches
ranked_resumes = sorted(zip(resumes, percentage_matches), key=lambda x: x[1], reverse=True)

# Print ranked resumes
for rank, (resume, percentage_match) in enumerate(ranked_resumes, 1):
    print(f"Rank {rank}: Percentage Match = {percentage_match:.2f}%")
    print(resume)
    print("-" * 50)
