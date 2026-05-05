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

st.title("🤖 HR Resume Screening AI Agent (Gemini)")
st.write(
    "Upload resumes, paste a Job Description, and let AI shortlist candidates."
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
    height=180
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
    resumes = []

    with st.spinner("Loading resumes..."):
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            # Combine all pages into one text
            full_text = " ".join([doc.page_content for doc in docs])

            resumes.append({
                "name": file.name,
                "content": full_text
            })

            os.remove(tmp_path)

    st.success(f"Loaded {len(resumes)} resumes")

    # -------------------------------------------------
    # 🤖 Gemini Model
    # -------------------------------------------------
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    # -------------------------------------------------
    # 📊 Evaluate Candidates
    # -------------------------------------------------
    st.subheader("📊 AI Evaluation Results")

    for idx, resume in enumerate(resumes, start=1):

        with st.expander(f"Candidate {idx} — {resume['name']}"):

            with st.spinner("Evaluating candidate..."):

                prompt = f"""
You are an HR AI Agent.

Evaluate the resume against the Job Description.

Job Description:
{job_description}

Resume:
{resume['content']}

Tasks:
1. Score the candidate out of 10
2. Key strengths
3. Missing skills
4. Final decision: Shortlist / Maybe / Reject

Give clear and professional output.
"""

                try:
                    response = model.generate_content(prompt)
                    st.write(response.text)

                except Exception as e:
                    st.error("Error while processing this resume.")
                    st.write(str(e))

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("HR AI Agent | Gemini Powered | Streamlit")
