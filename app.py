import streamlit as st
import pdfplumber
import docx
from pipeline.pipeline import RAGchain

# ---------- File Reading Utility ----------
def read_file(uploaded_file):
    """Reads text from PDF, DOCX, or TXT"""
    if uploaded_file is None:
        return ""

    if uploaded_file.type == "application/pdf":
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text

    elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])

    elif uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")

    else:
        return "Unsupported file type."


# ---------- Streamlit App ----------
st.set_page_config(page_title="Resume Ranker LLM", page_icon="📄")

st.title("📄 Resume Ranker LLM")
st.write("Upload a **Job Description** file and multiple **Resumes** to proceed.")

# Upload job description (strictly one file)
job_description_file = st.file_uploader(
    "Upload Job Description (Only one file)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=False
)

# Upload resumes (multiple files allowed)
resume_files = st.file_uploader(
    "Upload Resumes (Multiple files allowed)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

resumes=[]
# Process button
if st.button("Process Files"):
    if not job_description_file:
        st.error("⚠ Please upload exactly ONE Job Description file.")
    elif not resume_files:
        st.error("⚠ Please upload at least one Resume.")
    else:
        st.success("✅ Files uploaded successfully!")
        
        # Read job description text
        jd_text = read_file(job_description_file)
        st.subheader("Job Description Content:")
        st.text_area("JD Text", jd_text, height=200)

        # Read resumes
        for resume_file in resume_files:
            resume_text = read_file(resume_file)
            resumes.append(resume_text)
            # st.subheader(f"Resume: {resume_file.name}")
            # st.text_area("Resume Text", resume_text, height=200)

    c=RAGchain(jd_text, resumes)
    res = c.pipeline()
    
    st.write(res)

        
