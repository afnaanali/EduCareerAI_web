from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from core.interview_generator import (
    generate_interview_questions,
    generate_answer_and_tips,
    SUPPORTED_ROLES,
)

router = APIRouter(prefix="/api/interview", tags=["AI Interview Question Generator"])


class InterviewGenerateRequest(BaseModel):
    role: str = "Data Scientist / ML Engineer"
    category: Optional[str] = "all"
    difficulty: Optional[str] = "all"
    count: Optional[int] = 5
    focus_topics: Optional[str] = None


class InterviewAnswerTipsRequest(BaseModel):
    question: str
    role: Optional[str] = "Data Scientist / ML Engineer"
    category: Optional[str] = "behavioral"
    difficulty: Optional[str] = "Mid-Level"
    hint: Optional[str] = None
    model_answer: Optional[str] = None
    user_response: Optional[str] = None


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


@router.post("/answer-and-tips")
def get_answer_and_tips(payload: InterviewAnswerTipsRequest):
    """
    Generates or retrieves the gold-standard recommended answer along with
    strategic tips, essential keywords, and comparative feedback if the user
    provided an answer.
    """
    try:
        result = generate_answer_and_tips(
            question=payload.question,
            role=payload.role or "Data Scientist / ML Engineer",
            category=payload.category or "behavioral",
            difficulty=payload.difficulty or "Mid-Level",
            hint=payload.hint,
            model_answer=payload.model_answer,
            user_response=payload.user_response,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer and tips: {str(e)}")

