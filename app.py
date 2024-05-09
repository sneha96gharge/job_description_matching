import streamlit as st  # type: ignore
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
import base64
import os

# openai_api_key = os.getenv("OPENAI_API_KEY")

def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def preprocess(text):
    tokens = text.lower().split()
    return ' '.join(tokens)

def matching(job_description_text, resume_text):
    dataset = {
        'query_document': {'text': job_description_text},
        'resume_documents': {'text': resume_text},
    }
    preprocessed_query = preprocess(dataset['query_document']['text'])
    preprocessed_resumes = preprocess(dataset['resume_documents']['text'])
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([preprocessed_query] + [preprocessed_resumes])
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    print(similarity_scores[0])
    return similarity_scores[0]

def main():
    load_dotenv()

    

    st.set_page_config(page_title="Resume-JobDescription matcher", page_icon=":memo:", layout="centered", initial_sidebar_state="expanded")
    
    main_bg = "/home/sneha/Downloads/pngtree-luxury-mandala-design-png-vector-png-image_2707823_edited.jpg"
    main_bg_ext = "jpg"

    side_bg = "/home/sneha/Downloads/pngtree-luxury-mandala-design-png-vector-png-image_2707823_edited.jpg"
    side_bg_ext = "jpg"

    st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
   .sidebar .sidebar-content {{
        background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)

    st.header("Matching percentage: ")


    match_score = None

    with st.sidebar:
        st.subheader("Your documents")
        job_description_pdf = st.file_uploader(
            "Upload the Job Description PDF here", accept_multiple_files=False)
        resume_pdf = st.file_uploader(
            "Upload the Resume PDF here", accept_multiple_files=False)
        
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                resume_text = get_pdf_text(resume_pdf)
                job_description_text = get_pdf_text(job_description_pdf)

                match_score = matching(job_description_text, resume_text)

    if match_score is not None:  # Check if match_score has been assigned a value
        # Display matching score outside of the sidebar
        st.markdown(f"<h1 style='font-size:36px;'>{match_score * 100:.2f}%</h1>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
