import os
import tempfile
import streamlit as st
from google import genai

from langchain_community.document_loaders import PyPDFLoader

# -------------------------------------------------
# 🔑 GOOGLE API KEY
# -------------------------------------------------
client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

# -------------------------------------------------
# 🖥️ Streamlit App
# -------------------------------------------------
st.set_page_config(page_title="HR Resume Screening AI Agent", layout="wide")

st.title("🤖 HR Resume Screening AI Agent (Gemini)")
st.write("Upload resumes and evaluate candidates using AI")

uploaded_files = st.file_uploader(
    "Upload Resume PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

job_description = st.text_area("Paste Job Description", height=180)

# -------------------------------------------------
# ▶️ Run Button
# -------------------------------------------------
if st.button("Run AI Screening"):

    if not uploaded_files:
        st.warning("Upload resumes")
        st.stop()

    if not job_description.strip():
        st.warning("Enter Job Description")
        st.stop()

    resumes = []

    # Load PDFs
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        full_text = " ".join([doc.page_content for doc in docs])

        resumes.append({
            "name": file.name,
            "content": full_text
        })

        os.remove(tmp_path)

    st.success(f"{len(resumes)} resumes loaded")

    # -------------------------------------------------
    # 📊 Evaluation
    # -------------------------------------------------
    for idx, r in enumerate(resumes, start=1):

        with st.expander(f"Candidate {idx} — {r['name']}"):

            prompt = f"""
You are an HR AI Agent.

Evaluate this resume based on the job description.

Job Description:
{job_description}

Resume:
{r['content']}

Give:
1. Score /10
2. Strengths
3. Missing skills
4. Final decision
"""

            try:
                response = client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=prompt
                )

                st.write(response.text)

            except Exception as e:
                st.error("Error processing resume")
                st.write(str(e))

# Footer
st.markdown("---")
st.caption("Gemini AI Resume Screener")
