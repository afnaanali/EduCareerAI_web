from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from core.interview_generator import (
    generate_interview_questions,
    SUPPORTED_ROLES,
)

router = APIRouter(prefix="/api/interview", tags=["AI Interview Question Generator"])


class InterviewGenerateRequest(BaseModel):
    role: str = "Data Scientist / ML Engineer"
    category: Optional[str] = "all"
    difficulty: Optional[str] = "all"
    count: Optional[int] = 5
    focus_topics: Optional[str] = None


@router.get("/roles")
def get_supported_roles():
    """Returns the list of pre-configured popular roles."""
    return {
        "success": True,
        "roles": SUPPORTED_ROLES,
    }


@router.post("/generate")
def generate_questions(payload: InterviewGenerateRequest):
    """
    Generates tailored interview questions based on the selected role,
    category (behavioral, technical, leadership, situational),
    and seniority level.
    """
    try:
        questions = generate_interview_questions(
            role=payload.role,
            category=payload.category or "all",
            difficulty=payload.difficulty or "all",
            count=payload.count or 5,
            focus_topics=payload.focus_topics,
        )
        return {
            "success": True,
            "role": payload.role,
            "count": len(questions),
            "questions": questions,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate interview questions: {str(e)}")
