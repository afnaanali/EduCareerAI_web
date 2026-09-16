import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# Cross-version compatibility for unpickling HistGradientBoostingClassifier & ColumnTransformer
try:
    import sklearn._loss._loss as _cy_loss
    sys.modules["sklearn.ensemble._hist_gradient_boosting._loss"] = _cy_loss
    sys.modules["_loss"] = _cy_loss
except Exception:
    pass

try:
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        _ct._RemainderColsList = _RemainderColsList
except Exception:
    pass



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

GRADE_LABELS = {
    "grade_math": "Math",
    "grade_science": "Science",
    "grade_lang": "Language",
    "grade_social": "Social Science",
    "grade_cs": "Computer Science",
}

APTITUDE_COLUMNS = [
    "score_analytical",
    "score_numeric",
    "score_verbal",
    "score_creative",
    "score_social",
]

APTITUDE_LABELS = {
    "score_analytical": "Analytical Aptitude",
    "score_numeric": "Numerical Aptitude",
    "score_verbal": "Verbal Aptitude",
    "score_creative": "Creative Aptitude",
    "score_social": "Social Aptitude",
}

CAREER_FIELDS = [
    "Engineering",
    "Medical & Healthcare",
    "Business & Finance",
    "Science & Research",
    "Arts & Design",
    "Computer Science & IT",
    "Education",
    "Law & Government",
    "Media & Communication",
    "Sports & Fitness",
]

FIELD_KEYWORDS = {
    "Engineering": ["engineer", "architect", "technician"],
    "Medical & Healthcare": ["doctor", "nurse", "medical", "physiotherapist", "psychologist", "health", "laboratory"],
    "Business & Finance": ["accountant", "manager", "marketing", "business", "finance", "financial", "trading", "operations", "human resource", "hr", "economist"],
    "Science & Research": ["scientist", "research", "physicist", "chemist", "biologist", "astronomer", "mathematician", "statistician", "geologist", "oceanographer", "ocean", "space"],
    "Arts & Design": ["designer", "artist", "animator", "fashion", "interior", "vfx", "illustrator", "drawing"],
    "Computer Science & IT": ["software", "developer", "programmer", "data scientist", "data analyst", "machine learning", "ai engineer", "cloud engineer", "devops", "network engineer", "cyber", "forensics", "web developer", "game developer", "it project"],
    "Education": ["teacher", "professor", "trainer", "education", "special education", "instructor"],
    "Law & Government": ["lawyer", "advocate", "government", "officer", "naval", "coast guard", "crime", "forensic"],
    "Media & Communication": ["journalist", "writer", "content", "editor", "film", "director", "media", "public relations", "social media", "video"],
    "Sports & Fitness": ["coach", "fitness", "sports", "physiotherapist", "yoga", "athlete"],
}

_CAREER_MODEL_CACHE = None

def load_career_model(model_dir: str = None):
    """Loads the trained HistGradientBoosting career model."""
    global _CAREER_MODEL_CACHE
    if _CAREER_MODEL_CACHE is not None:
        return _CAREER_MODEL_CACHE

    if model_dir is None:
        model_dir = Path(__file__).parent.parent / "models"
    else:
        model_dir = Path(model_dir)

    model_path = model_dir / "career_ranking_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Career model file not found at: {model_path}")

    data = joblib.load(model_path)
    _CAREER_MODEL_CACHE = {
        "model": data["model"],
        "feature_columns": data["feature_columns"],
        "career_list": data.get("career_list", []),
        "accuracy": data.get("accuracy", None),
    }
    return _CAREER_MODEL_CACHE


def create_career_features(student_dict: dict) -> pd.DataFrame:
    """Calculates the exact 40 features engineered for career ranking."""
    df = pd.DataFrame([student_dict])

    # 1. Academic averages
    df["academic_average"] = (
        df["grade_math"] + df["grade_science"] + df["grade_lang"] + df["grade_social"] + df["grade_cs"]
    ) / 5

    df["stem_average"] = (df["grade_math"] + df["grade_science"] + df["grade_cs"]) / 3
    df["language_social_average"] = (df["grade_lang"] + df["grade_social"]) / 2

    # 2. Aptitude averages
    df["aptitude_average"] = (
        df["score_analytical"] + df["score_numeric"] + df["score_verbal"] + df["score_creative"] + df["score_social"]
    ) / 5

    df["logical_strength"] = (df["score_analytical"] + df["score_numeric"]) / 2
    df["communication_strength"] = (df["score_verbal"] + df["score_social"]) / 2
    df["creative_strength"] = (df["score_creative"] + df["score_verbal"]) / 2

    # 3. Domain Strengths
    df["technology_strength"] = (
        df["grade_cs"]
        + df["score_analytical"]
        + df["score_numeric"]
        + df["hobby_coding"] * 100
        + df["hobby_robotics"] * 100
    ) / 5

    df["science_strength"] = (
        df["grade_science"]
        + df["score_analytical"]
        + df["score_numeric"]
        + df["hobby_science_experiments"] * 100
    ) / 4

    df["business_strength"] = (
        df["score_numeric"]
        + df["score_social"]
        + df["score_verbal"]
        + df["hobby_business_trading"] * 100
    ) / 4

    df["creative_strength_combined"] = (
        df["score_creative"]
        + df["grade_lang"]
        + df["hobby_drawing_art"] * 100
        + df["hobby_music"] * 100
        + df["hobby_writing"] * 100
        + df["hobby_fashion"] * 100
    ) / 6

    df["social_strength"] = (
        df["score_social"]
        + df["score_verbal"]
        + df["grade_social"]
        + df["hobby_debating"] * 100
        + df["hobby_volunteering"] * 100
    ) / 5

    # 4. Hobby count and Overall
    df["hobby_count"] = df[HOBBY_COLUMNS].sum(axis=1)
    df["overall_strength"] = (df["academic_average"] + df["aptitude_average"]) / 2

    return df


def apply_career_field_boost(results_df: pd.DataFrame, selected_field: str) -> pd.DataFrame:
    """Applies +0.20 field compatibility boost based on field keywords."""
    keywords = FIELD_KEYWORDS.get(selected_field, [])

    def calculate_boost(career_name):
        career_text = str(career_name).lower()
        for kw in keywords:
            if kw.lower() in career_text:
                return 0.20
        return 0.0

    results_df["Field_Boost"] = results_df["Career"].apply(calculate_boost)
    results_df["Final_Score"] = results_df["Score"] + results_df["Field_Boost"]
    return results_df


def predict_top_careers(student_dict: dict, selected_field: str, top_n: int = 5, model_dir: str = None) -> pd.DataFrame:
    """Predicts Top N career recommendations using the trained HistGradientBoosting model."""
    bundle = load_career_model(model_dir)
    ranking_model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    student_df = create_career_features(student_dict)

    missing = [c for c in feature_columns if c not in student_df.columns]
    if missing:
        raise ValueError(f"Missing required career feature columns: {missing}")

    X = student_df[feature_columns].values

    probabilities = ranking_model.predict_proba(X)[0]
    classes = ranking_model.classes_

    results = pd.DataFrame({"Career": classes, "Score": probabilities})

    # Filter out NOT_RECOMMENDED
    results = results[results["Career"].astype(str).str.upper() != "NOT_RECOMMENDED"].copy()

    # Normalize raw model scores
    total_score = results["Score"].sum()
    if total_score > 0:
        results["Score"] = results["Score"] / total_score

    # Apply field boost
    results = apply_career_field_boost(results, selected_field)

    # Sort
    results = results.sort_values(by="Final_Score", ascending=False).reset_index(drop=True)

    # Final normalization
    total_final = results["Final_Score"].sum()
    if total_final > 0:
        results["Final_Score"] = results["Final_Score"] / total_final

    results["Suitability_Percent"] = (results["Final_Score"] * 100).round(2)
    return results.head(top_n).copy()
