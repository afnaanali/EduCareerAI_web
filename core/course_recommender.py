import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

HOBBY_COLUMNS = [
    "hobby_coding",
    "hobby_gaming",
    "hobby_reading",
    "hobby_writing",
    "hobby_music",
    "hobby_drawing_art",
    "hobby_sports",
    "hobby_cooking",
    "hobby_photography",
    "hobby_travel",
    "hobby_science_experiments",
    "hobby_volunteering",
    "hobby_debating",
    "hobby_robotics",
    "hobby_fashion",
    "hobby_business_trading",
]

GRADE_COLUMNS = [
    "grade_math",
    "grade_science",
    "grade_lang",
    "grade_social",
    "grade_cs",
]

APTITUDE_COLUMNS = [
    "score_analytical",
    "score_numeric",
    "score_verbal",
    "score_creative",
    "score_social",
]

# Field mapping bridge connecting Career Field names to Course Model field_filter categories
CAREER_TO_COURSE_FIELD_MAP = {
    "Engineering": "Engineering",
    "Medical & Healthcare": "Health",
    "Business & Finance": "Business",
    "Science & Research": "Science",
    "Arts & Design": "Arts",
    "Computer Science & IT": "STEM",
    "Education": "Education",
    "Law & Government": "Law",
    "Media & Communication": "Media",
    "Sports & Fitness": "Health",
}

_COURSE_MODEL_CACHE = None

def load_course_model(model_dir: str = None):
    """Loads the trained MultiOutputClassifier course recommendation model and transformers."""
    global _COURSE_MODEL_CACHE
    if _COURSE_MODEL_CACHE is not None:
        return _COURSE_MODEL_CACHE

    if model_dir is None:
        model_dir = Path(__file__).parent.parent / "models"
    else:
        model_dir = Path(model_dir)

    model_path = model_dir / "course_recommendation_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Course model file not found at: {model_path}")

    data = joblib.load(model_path)
    _COURSE_MODEL_CACHE = {
        "model": data["model"],
        "preprocessor": data["preprocessor"],
        "label_encoders": data["label_encoders"],
        "target_columns": data["target_columns"],
        "fields": data.get("fields", [
            "Architecture", "Arts", "Business", "Education", "Engineering",
            "Health", "Law", "Media", "STEM", "Science"
        ]),
    }
    return _COURSE_MODEL_CACHE


def predict_top_courses(student_dict: dict, field_filter: str = None, career_field: str = None, model_dir: str = None) -> list:
    """
    Predicts Top 5 recommended courses using the trained pipeline.
    Accepts either direct course `field_filter` or `career_field` (which gets automatically bridged).
    """
    bundle = load_course_model(model_dir)
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    label_encoders = bundle["label_encoders"]
    target_columns = bundle["target_columns"]
    available_fields = bundle["fields"]

    # Resolve course field filter
    if not field_filter and career_field:
        field_filter = CAREER_TO_COURSE_FIELD_MAP.get(career_field, "STEM")
    elif not field_filter:
        field_filter = available_fields[0]

    if field_filter not in available_fields:
        # Fallback to mapped or first
        field_filter = CAREER_TO_COURSE_FIELD_MAP.get(field_filter, available_fields[0])

    # Build student row dict
    row = {"field_filter": field_filter}
    for h in HOBBY_COLUMNS:
        row[h] = student_dict.get(h, 0)
    for g in GRADE_COLUMNS:
        row[g] = float(student_dict.get(g, 50.0))
    for a in APTITUDE_COLUMNS:
        row[a] = float(student_dict.get(a, 50.0))

    expected_columns = ["field_filter"] + HOBBY_COLUMNS + GRADE_COLUMNS + APTITUDE_COLUMNS
    student_df = pd.DataFrame([row])[expected_columns]

    processed = preprocessor.transform(student_df)
    raw_preds = model.predict(processed)

    if hasattr(raw_preds, "ndim") and raw_preds.ndim == 1:
        raw_preds = raw_preds.reshape(1, -1)

    recommendations = []
    for i, target_col in enumerate(target_columns):
        pred_val = raw_preds[0, i]
        encoder = label_encoders[target_col]
        try:
            course_name = encoder.inverse_transform([int(pred_val)])[0]
        except Exception:
            course_name = str(pred_val)
        recommendations.append({
            "rank": i + 1,
            "target_slot": target_col,
            "course": course_name,
            "field": field_filter
        })

    return recommendations
