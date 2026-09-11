# NLP-Powered Resume–Job Matching & Candidate Ranking System

**Author : Arpit Pandey**

An NLP-powered resume screening and candidate ranking system that analyzes resumes and matches them with job descriptions using semantic similarity, skill matching, experience analysis, education, certifications, and explainable scoring.

## 🚀 Project Overview

Recruiters often have to review a large number of resumes for a single job opening. This project automates the initial resume screening process by extracting important information from resumes and comparing candidates against a given job description.

The system processes resumes in PDF, DOCX, and TXT formats and generates an overall candidate match score.

## ✨ Key Features

- 📄 Resume parsing from PDF, DOCX, and TXT files
- 🧠 NLP-based resume analysis
- 🔍 Job description analysis
- 🛠️ Technical skill extraction
- 💼 Work experience extraction
- 🎓 Education analysis
- 📜 Certification extraction
- 👨‍💻 Job-role identification
- 🔗 Semantic similarity using Sentence Transformers
- 📊 Explainable candidate scoring
- 🏆 Automatic candidate ranking
- ✅ Matched skills identification
- ❌ Missing skills identification
- ➕ Additional skills identification
- 📈 Interactive score breakdown
- 📋 Top candidate highlighting
- 📥 CSV results export
- 🌐 Interactive Streamlit web interface

## 🧩 How It Works

The system follows these main steps:

1. **Upload Resumes**
   - Users upload candidate resumes in PDF, DOCX, or TXT format.

2. **Provide Job Description**
   - A job description is entered into the application.

3. **Resume Processing**
   - The system extracts text and relevant information from each resume.

4. **Information Extraction**
   - Skills
   - Experience
   - Education
   - Certifications
   - Job roles

5. **Semantic Matching**
   - Resume and job-description text are converted into embeddings using a Sentence Transformer model.
   - Semantic similarity is calculated between the candidate resume and job description.

6. **Candidate Scoring**
   - The system combines multiple factors including:
     - Semantic similarity
     - Skill coverage
     - Experience match
     - Education match

7. **Candidate Ranking**
   - Candidates are ranked according to their overall match score.

8. **Explainable Results**
   - The application shows matched, missing, and additional skills along with the score breakdown.

## 🛠️ Technologies Used

- Python
- Streamlit
- Natural Language Processing (NLP)
- spaCy
- Sentence Transformers
- PyTorch
- Scikit-learn
- Pandas
- NumPy
- PyPDF
- python-docx
- Matplotlib

## 🧠 NLP & Machine Learning

The project uses:

### spaCy

Used for natural language processing and information extraction from resumes and job descriptions.

### Sentence Transformers

The project uses the `all-MiniLM-L6-v2` model to generate sentence/document embeddings and calculate semantic similarity.

### Scoring System

Candidate ranking is based on a weighted scoring framework combining different recruitment-related factors.

## 📊 Output

For each candidate, the system provides:

- Overall Match Score
- Semantic Similarity Score
- Skill Coverage
- Experience Match
- Education Match
- Matched Skills
- Missing Skills
- Additional Skills
- Candidate Ranking

## 👨‍💻 Author

**Arpit Pandey**
