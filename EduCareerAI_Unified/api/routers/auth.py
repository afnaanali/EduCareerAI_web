from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any

from core.database import (
    register_user,
    authenticate_user,
    save_student_profile,
    load_student_profile,
    get_user_career_assessments,
    get_user_resume_scans,
    delete_career_assessment,
    delete_resume_scan,
)

router = APIRouter(prefix="/api/auth", tags=["Authentication & User"])


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None


class LoginRequest(BaseModel):
    identifier: str
    password: str


class ProfileUpdateRequest(BaseModel):
    user_id: int
    profile: Dict[str, Any]


@router.post("/register")
def register(payload: RegisterRequest):
    ok, msg, user = register_user(
        username=payload.username.strip(),
        email=payload.email.strip(),
        password=payload.password,
        full_name=payload.full_name.strip() if payload.full_name else None,
    )
    if not ok:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)
    return {"success": True, "message": msg, "user": user}


@router.post("/login")
def login(payload: LoginRequest):
    ok, msg, user = authenticate_user(
        identifier=payload.identifier.strip(),
        password=payload.password,
    )
    if not ok:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=msg)
    
    # Load profile and recent history if exists
    profile = load_student_profile(user["id"])
    assessments = get_user_career_assessments(user["id"], limit=1)
    resume_scans = get_user_resume_scans(user["id"], limit=1)
    return {
        "success": True,
        "message": msg,
        "user": user,
        "profile": profile,
        "latest_assessment": assessments[0] if assessments else None,
        "latest_resume": resume_scans[0] if resume_scans else None,
    }


@router.get("/profile/{user_id}")
def get_profile(user_id: int):
    profile = load_student_profile(user_id)
    assessments = get_user_career_assessments(user_id, limit=1)
    resume_scans = get_user_resume_scans(user_id, limit=1)
    return {
        "user_id": user_id,
        "profile": profile,
        "latest_assessment": assessments[0] if assessments else None,
        "latest_resume": resume_scans[0] if resume_scans else None,
    }


@router.post("/profile")
def update_profile(payload: ProfileUpdateRequest):
    success = save_student_profile(payload.user_id, payload.profile)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save profile to database")
    return {"success": True, "message": "Profile saved successfully"}


@router.get("/history/{user_id}")
def get_user_history(user_id: int):
    assessments = get_user_career_assessments(user_id)
    resume_scans = get_user_resume_scans(user_id)
    return {
        "user_id": user_id,
        "assessments": assessments,
        "resume_scans": resume_scans,
    }


@router.delete("/history/assessment/{user_id}/{assessment_id}")
def remove_assessment(user_id: int, assessment_id: int):
    ok = delete_career_assessment(assessment_id, user_id)
    return {"success": ok}


@router.delete("/history/resume/{user_id}/{scan_id}")
def remove_resume_scan(user_id: int, scan_id: int):
    ok = delete_resume_scan(scan_id, user_id)
    return {"success": ok}
