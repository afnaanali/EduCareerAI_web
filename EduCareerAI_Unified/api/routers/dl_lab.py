import io
import base64
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from core.dl_models import (
    predict_career_ann,
    predict_digit_cnn,
    preprocess_handwritten_digit,
    analyze_uploaded_marksheet,
    compare_rnn_and_lstm,
    analyze_interview_tone_coach,
    ANN_SKILLS,
)

router = APIRouter(prefix="/api/lab", tags=["Deep Learning Lab"])


class ANNPredictRequest(BaseModel):
    skill_vector: List[int] = Field(..., min_items=8, max_items=8)


class SentimentToneRequest(BaseModel):
    text: str


@router.get("/ann-skills")
def get_ann_skills():
    return {
        "skills": ANN_SKILLS,
    }


@router.post("/ann-predict")
def predict_ann(payload: ANNPredictRequest):
    try:
        results = predict_career_ann(payload.skill_vector)
        formatted_predictions = [
            {
                "Career": str(item["career"]),
                "Confidence": float(item["probability"] / 100.0),
                "Probability": float(item["probability"] / 100.0),
            }
            for item in results.get("breakdown", [])
        ]
        return {
            "success": True,
            "skill_vector": payload.skill_vector,
            "predictions": formatted_predictions,
            "top_career": results.get("top_career"),
            "confidence": float(results.get("confidence", 0) / 100.0),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cnn-digit")
async def predict_digit(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
):
    try:
        if file:
            content = await file.read()
        elif image_base64:
            if "," in image_base64:
                image_base64 = image_base64.split(",")[1]
            content = base64.b64decode(image_base64)
        else:
            raise HTTPException(status_code=400, detail="Provide an image file or base64 image data")

        # Try multi-strategy digit analysis
        try:
            canvas, meta = preprocess_handwritten_digit(content)
            pred_dict = predict_digit_cnn(canvas)
            p_digit = int(pred_dict["predicted_digit"])
            p_conf = float(pred_dict["confidence"] / 100.0)
            p_probs = [float(p / 100.0) for p in pred_dict["probabilities"]]
        except Exception:
            # Fallback to full marksheet analyzer
            sheet_res = analyze_uploaded_marksheet(content)
            if sheet_res.get("success") and sheet_res.get("digits"):
                first_d = sheet_res["digits"][0]
                p_digit = int(first_d["digit"])
                p_conf = float(first_d["confidence"] / 100.0)
                p_probs = [float(p / 100.0) for p in first_d.get("probabilities", [0]*10)]
            else:
                raise

        return {
            "success": True,
            "predicted_digit": p_digit,
            "confidence": p_conf,
            "probabilities": p_probs,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CNN Digit Inference Error: {str(e)}")


@router.post("/cnn-marksheet")
async def process_marksheet(file: UploadFile = File(...)):
    try:
        content = await file.read()
        analysis = analyze_uploaded_marksheet(content)
        return {
            "success": True,
            "analysis": analysis,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sentiment-tone")
def analyze_sentiment_and_tone(payload: SentimentToneRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    try:
        comparison = compare_rnn_and_lstm(payload.text)
        coach = analyze_interview_tone_coach(payload.text)

        # Build tone_feedback list for the frontend
        feedback_list = []
        feedback_list.append(f"Tone Assessment: {coach['tone_label']} ({coach['tone_badge']}) — Impact Score: {coach['impact_score']}/100.")
        
        if coach.get("found_action_verbs"):
            feedback_list.append(f"Strong action verbs detected: {', '.join(coach['found_action_verbs'])}.")
        else:
            feedback_list.append("No strong action verbs detected. Try incorporating verbs like 'spearheaded', 'engineered', 'optimized', or 'architected'.")

        if coach.get("found_weak_phrases"):
            for item in coach["found_weak_phrases"]:
                feedback_list.append(f"Consider replacing '{item['phrase']}' with '{item['fix']}'.")
        else:
            feedback_list.append("Clear assertive phrasing — no hesitant filler expressions detected.")

        star_missing = [k for k, v in coach.get("star_breakdown", {}).items() if not v]
        if star_missing:
            feedback_list.append(f"STAR Framework Note: Consider adding more detail on your {', '.join(star_missing)}.")
        else:
            feedback_list.append("Complete STAR structure (Situation, Task, Action, Result) detected.")

        coach_dict = {
            **coach,
            "tone_feedback": feedback_list,
            "actionable_tips": feedback_list,
        }

        norm_comparison = {
            "lstm": {
                "sentiment": comparison["lstm"]["sentiment"],
                "confidence": float(comparison["lstm"]["confidence"] / 100.0),
                "raw_score": comparison["lstm"]["raw_score"],
            },
            "rnn": {
                "sentiment": comparison["rnn"]["sentiment"],
                "confidence": float(comparison["rnn"]["confidence"] / 100.0),
                "raw_score": comparison["rnn"]["raw_score"],
            },
        }

        return {
            "success": True,
            "comparison": norm_comparison,
            "coach": coach_dict,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

