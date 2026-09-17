import io
import docx
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from core.resume_analyzer import (
    extract_resume_text,
    clean_resume_text,
    detect_resume_sections,
    extract_skills,
    extract_ann_skill_vector,
    calculate_ats_score,
    calculate_job_match,
    generate_rule_based_careers,
    build_resume_issues,
    generate_skill_gap,
    CAREER_PROFILES,
    ANN_8_SKILLS,
)
from core.database import (
    save_resume_scan,
    get_user_resume_scans,
    delete_resume_scan,
)

router = APIRouter(prefix="/api/resume", tags=["Resume & ATS Studio"])


class AnalyzeTextRequest(BaseModel):
    user_id: Optional[int] = None
    resume_text: str
    resume_name: Optional[str] = "Pasted_Resume.txt"
    target_job: Optional[str] = None
    job_description: Optional[str] = None


def _perform_resume_analysis(
    text: str,
    resume_name: str,
    user_id: Optional[int] = None,
    target_job: Optional[str] = None,
    job_description: Optional[str] = None,
) -> Dict[str, Any]:
    cleaned_text = clean_resume_text(text)
    if not cleaned_text.strip():
        raise HTTPException(status_code=400, detail="Resume content is empty or could not be parsed.")

    sections = detect_resume_sections(cleaned_text)
    skills = extract_skills(cleaned_text)
    ann_vector = extract_ann_skill_vector(skills)
    overall_ats, breakdown = calculate_ats_score(cleaned_text, sections)
    matched_careers = generate_rule_based_careers(skills)

    job_skills = []
    match_score = None
    matched_job_skills = []
    missing_job_skills = []

    if target_job and target_job in CAREER_PROFILES:
        job_skills = CAREER_PROFILES[target_job]
        match_score, matched_job_skills, missing_job_skills = calculate_job_match(skills, job_skills)
    elif job_description:
        job_skills = extract_skills(job_description)
        if job_skills:
            match_score, matched_job_skills, missing_job_skills = calculate_job_match(skills, job_skills)

    skill_gap = generate_skill_gap(skills, job_skills, matched_careers)
    raw_issues = build_resume_issues(cleaned_text, sections)
    issues = [
        {"title": i[0], "why": i[1], "fix": i[2]}
        for i in raw_issues
    ]

    # Save scan to database
    if user_id:
        save_resume_scan(
            user_id=user_id,
            resume_name=resume_name,
            target_job=target_job or (matched_careers[0]["career"] if matched_careers else "General"),
            ats_score=float(overall_ats),
            match_score=float(match_score) if match_score is not None else None,
            extracted_skills=skills,
            issues=issues,
        )

    return {
        "success": True,
        "resume_name": resume_name,
        "ats_score": overall_ats,
        "breakdown": breakdown,
        "extracted_skills": skills,
        "ann_vector": {
            "skills": ANN_8_SKILLS,
            "vector": ann_vector,
        },
        "target_job": target_job,
        "match_score": match_score,
        "matched_job_skills": matched_job_skills,
        "missing_job_skills": missing_job_skills,
        "skill_gap": skill_gap,
        "matched_careers": matched_careers,
        "issues": issues,
        "sections_detected": {k: bool(v) for k, v in sections.items()},
        "word_count": len(cleaned_text.split()),
    }


@router.post("/analyze-text")
def analyze_text(payload: AnalyzeTextRequest):
    return _perform_resume_analysis(
        text=payload.resume_text,
        resume_name=payload.resume_name or "Pasted_Resume.txt",
        user_id=payload.user_id,
        target_job=payload.target_job,
        job_description=payload.job_description,
    )


@router.post("/analyze-file")
async def analyze_file(
    file: UploadFile = File(...),
    user_id: Optional[int] = Form(None),
    target_job: Optional[str] = Form(None),
    job_description: Optional[str] = Form(None),
):
    filename = file.filename or "uploaded_resume"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty. Please select a valid resume document.")

    extracted_text = ""
    try:
        if ext == "pdf":
            extracted_text = extract_resume_text(io.BytesIO(content))
        elif ext in ["docx", "doc"]:
            doc = docx.Document(io.BytesIO(content))
            extracted_text = "\n".join([p.text for p in doc.paragraphs if p.text])
        elif ext == "txt":
            for enc in ("utf-8", "latin-1", "cp1252"):
                try:
                    extracted_text = content.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format '.{ext}'. Please upload a PDF, DOCX, or TXT resume.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file '{filename}': {str(e)}")

    if not extracted_text.strip():
        raise HTTPException(
            status_code=400,
            detail=f"No readable text could be extracted from '{filename}'. If this document is a scanned image or photo PDF, please convert it to a text PDF or paste the text in the 'Paste Plain Text' tab."
        )

    return _perform_resume_analysis(
        text=extracted_text,
        resume_name=filename,
        user_id=user_id,
        target_job=target_job,
        job_description=job_description,
    )


@router.get("/target-roles")
def get_target_roles():
    return {
        "roles": list(CAREER_PROFILES.keys()),
    }
