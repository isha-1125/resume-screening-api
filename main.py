from fastapi import FastAPI, UploadFile, File, Form
from utils import extract_text, calculate_score

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Resume Screening API is running"}

@app.post("/analyze/")
async def analyze_resume(
    file: UploadFile = File(...),
    job_desc: str = Form(...)
):
    resume_text = extract_text(await file.read())
    score, matched = calculate_score(resume_text, job_desc)

    return {
        "match_score": score,
        "matched_keywords": matched
    }