from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

from core.roadmap_generator import get_roadmap_for_career, CAREER_ROADMAPS

router = APIRouter(prefix="/api/roadmap", tags=["Learning Roadmaps"])


@router.get("/careers")
def list_roadmap_careers():
    careers = list(CAREER_ROADMAPS.keys())
    return {
        "careers": careers,
        "total": len(careers),
    }


@router.get("/{career_name}")
def get_roadmap(career_name: str):
    roadmap = get_roadmap_for_career(career_name)
    if not roadmap:
        raise HTTPException(
            status_code=404,
            detail=f"Roadmap not found for '{career_name}'. Available: {list(CAREER_ROADMAPS.keys())}"
        )
    return {
        "career": career_name,
        "roadmap": roadmap,
    }
