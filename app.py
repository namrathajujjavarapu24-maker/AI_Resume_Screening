import os
import gradio as gr
import pandas as pd

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load AI model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""

    reader = PdfReader(pdf_path)

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# Find matching and missing skills
def find_matching_keywords(job_description, resume_text):

    keywords = [
        "python",
        "java",
        "javascript",
        "machine learning",
        "deep learning",
        "artificial intelligence",
        "nlp",
        "sql",
        "mongodb",
        "tensorflow",
        "pytorch",
        "scikit-learn",
        "pandas",
        "numpy",
        "html",
        "css",
        "react",
        "git",
        "github",
        "api",
        "data science"
    ]

    job_lower = job_description.lower()
    resume_lower = resume_text.lower()

    required = []

    for keyword in keywords:
        if keyword in job_lower:
            required.append(keyword)

    matched = []
    missing = []

    for keyword in required:

        if keyword in resume_lower:
            matched.append(keyword)
        else:
            missing.append(keyword)

    return matched, missing


# Main screening function
def screen_resumes(job_description, resume_files):

    if not job_description.strip():
        return "Please enter a job description.", None

    if not resume_files:
        return "Please upload at least one resume.", None

    resume_texts = []
    resume_names = []

    for file in resume_files:

        text = extract_text_from_pdf(file)

        resume_texts.append(text)
        resume_names.append(os.path.basename(file))

    # Job embedding
    job_embedding = model.encode([job_description])

    # Resume embeddings
    resume_embeddings = model.encode(resume_texts)

    # Similarity
    scores = cosine_similarity(
        job_embedding,
        resume_embeddings
    )[0]

    results = []

    for i in range(len(resume_texts)):

        matched, missing = find_matching_keywords(
            job_description,
            resume_texts[i]
        )

        results.append({
            "Resume": resume_names[i],
            "Match Score": round(float(scores[i]) * 100, 2),
            "Matched Skills": ", ".join(matched) if matched else "None",
            "Missing Skills": ", ".join(missing) if missing else "None"
        })

    # Create table
    df = pd.DataFrame(results)

    # Rank candidates
    df = df.sort_values(
        by="Match Score",
        ascending=False
    ).reset_index(drop=True)

    # Add rank
    df.insert(
        0,
        "Rank",
        range(1, len(df) + 1)
    )

    return "Screening completed successfully!", df


# Create web interface
with gr.Blocks(title="AI Resume Screening System") as demo:

    gr.Markdown(
        """
        # 🤖 AI-Based Resume Screening System

        Automatically screen and rank resumes based on job requirements.

        **NLP + Transformer Embeddings + Cosine Similarity**
        """
    )

    job_description = gr.Textbox(
        label="📋 Job Description",
        placeholder="Enter the job requirements...",
        lines=8
    )

    resume_files = gr.File(
        label="📄 Upload Resumes (PDF)",
        file_count="multiple",
        file_types=[".pdf"],
        type="filepath"
    )

    screen_button = gr.Button(
        "🔍 Screen Resumes",
        variant="primary"
    )

    status = gr.Textbox(
        label="Status"
    )

    results = gr.Dataframe(
        label="🏆 Candidate Ranking",
        interactive=False
    )

    screen_button.click(
        fn=screen_resumes,
        inputs=[job_description, resume_files],
        outputs=[status, results]
    )


# Start application
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
