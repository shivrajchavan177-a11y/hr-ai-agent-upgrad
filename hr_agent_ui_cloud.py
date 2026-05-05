import os
import tempfile
import streamlit as st
import google.generativeai as genai

from langchain_community.document_loaders import PyPDFLoader

# -------------------------------------------------
# 🔑 GOOGLE API KEY
# -------------------------------------------------
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# -------------------------------------------------
# 🖥️ Streamlit App Config
# -------------------------------------------------
st.set_page_config(page_title="HR Resume Screening AI Agent", layout="wide")
st.title("🤖 HR Resume Screening AI Agent (Gemini Powered)")
st.write(
    "Upload resumes, paste a Job Description, and let the AI agent "
    "shortlist candidates using Google Gemini AI."
)

# -------------------------------------------------
# 📤 Upload Resume PDFs
# -------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload multiple resume PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

# -------------------------------------------------
# 📝 Job Description Input
# -------------------------------------------------
job_description = st.text_area(
    "Paste Job Description",
    height=180,
    placeholder="Example: Looking for a Data Scientist with Python, ML, SQL..."
)

# -------------------------------------------------
# ▶️ Run Button
# -------------------------------------------------
if st.button("Run AI Resume Screening"):

    if not uploaded_files:
        st.warning("Please upload at least one resume PDF.")
        st.stop()

    if not job_description.strip():
        st.warning("Please provide a Job Description.")
        st.stop()

    # -------------------------------------------------
    # 📄 Load Resumes
    # -------------------------------------------------
    documents = []

    with st.spinner("Loading resumes..."):
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            # Combine pages into one resume text
            full_text = " ".join([doc.page_content for doc in docs])

            documents.append({
                "name": file.name,
                "content": full_text
            })

            os.remove(tmp_path)

    st.success(f"Loaded {len(documents)} resumes")

    # -------------------------------------------------
    # 🤖 Gemini Model
    # -------------------------------------------------
    model = genai.GenerativeModel("gemini-1.5-flash")

    # -------------------------------------------------
    # 📊 Evaluate Candidates
    # -------------------------------------------------
    st.subheader("📊 AI Evaluation Results")

    for idx, doc in enumerate(documents, start=1):

        with st.expander(f"Candidate {idx} — {doc['name']}"):
            with st.spinner("Evaluating candidate..."):

                prompt = f"""
You are an HR AI Agent.

Evaluate the resume against the Job Description.

Job Description:
{job_description}

Resume:
{doc['content']}

Tasks:
1. Score the candidate out of 10
2. Key strengths
3. Missing skills
4. Final decision: Shortlist / Maybe / Reject

Keep it concise and professional.
"""

                response = model.generate_content(prompt)

            st.write(response.text)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("HR AI Agent | Powered by Gemini | Streamlit")
