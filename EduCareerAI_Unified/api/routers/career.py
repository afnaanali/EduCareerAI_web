from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

from core.career_recommender import (
    predict_top_careers,
    load_career_model,
    CAREER_FIELDS,
    HOBBY_COLUMNS,
    GRADE_COLUMNS,
    GRADE_LABELS,
    APTITUDE_COLUMNS,
    APTITUDE_LABELS,
)
from core.course_recommender import (
    predict_top_courses,
    load_course_model,
    CAREER_TO_COURSE_FIELD_MAP,
)
from core.database import save_career_assessment, save_student_profile

# Preload models into memory for sub-millisecond inference
try:
    load_career_model()
    load_course_model()
except Exception as _e:
    pass

router = APIRouter(prefix="/api/career", tags=["Career & Course Intelligence"])


class CareerPredictionRequest(BaseModel):
    user_id: Optional[int] = None
    field: str = "Computer Science & IT"
    hobbies: Dict[str, int] = Field(default_factory=dict)
    grades: Dict[str, float] = Field(default_factory=dict)
    aptitudes: Dict[str, float] = Field(default_factory=dict)
    top_n: int = 5


@router.get("/options")
def get_career_options():
    return {
        "fields": CAREER_FIELDS,
        "hobbies": [
            {"id": h, "label": h.replace("hobby_", "").replace("_", " ").title()}
            for h in HOBBY_COLUMNS
        ],
        "grades": [
            {"id": g, "label": GRADE_LABELS.get(g, g)}
            for g in GRADE_COLUMNS
        ],
        "aptitudes": [
            {"id": a, "label": APTITUDE_LABELS.get(a, a)}
            for a in APTITUDE_COLUMNS
        ],
    }


@router.post("/predict")
async def predict_careers(payload: CareerPredictionRequest):
    try:
        # Construct flat student_dict
        student_dict = {"field": payload.field}
        
        for h in HOBBY_COLUMNS:
            student_dict[h] = payload.hobbies.get(h, 0)
        for g in GRADE_COLUMNS:
            student_dict[g] = float(payload.grades.get(g, 65.0))
        for a in APTITUDE_COLUMNS:
            student_dict[a] = float(payload.aptitudes.get(a, 65.0))

        # 1. Predict top careers
        career_df = predict_top_careers(
            student_dict=student_dict,
            selected_field=payload.field,
            top_n=payload.top_n,
        )
        careers_list = career_df.to_dict(orient="records")

        # 2. Predict corresponding top courses
        course_field_filter = CAREER_TO_COURSE_FIELD_MAP.get(payload.field, None)
        courses_list = predict_top_courses(
            student_dict,
            field_filter=course_field_filter,
            top_n=6,
        )

        # 3. Save to database if user_id is passed
        if payload.user_id:
            save_student_profile(payload.user_id, student_dict)
            save_career_assessment(
                user_id=payload.user_id,
                careers_df_or_list=careers_list,
                courses_list=courses_list,
            )

        return {
            "success": True,
            "careers": careers_list,
            "courses": courses_list,
            "input_summary": {
                "field": payload.field,
                "academic_avg": round(sum(student_dict[g] for g in GRADE_COLUMNS) / len(GRADE_COLUMNS), 1),
                "aptitude_avg": round(sum(student_dict[a] for a in APTITUDE_COLUMNS) / len(APTITUDE_COLUMNS), 1),
                "active_hobbies_count": sum(student_dict[h] for h in HOBBY_COLUMNS),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
