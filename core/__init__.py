# EduCareerAI Unified Core Package
from .career_recommender import predict_top_careers, load_career_model
from .course_recommender import predict_top_courses, load_course_model
from .resume_analyzer import (
    extract_resume_text,
    detect_resume_sections,
    extract_skills,
    extract_ann_skill_vector,
    calculate_ats_score,
    calculate_job_match,
    generate_rule_based_careers,
    build_resume_issues,
    generate_skill_gap,
)
from .chatbot import get_llm_model, generate_llm_response, build_system_prompt
from .dl_models import (
    predict_career_ann,
    predict_digit_cnn,
    predict_sentiment_rnn,
    predict_sentiment_lstm,
    compare_rnn_and_lstm,
    analyze_interview_tone_coach,
)

from .roadmap_generator import get_roadmap_for_career
from .database import (
    init_db,
    register_user,
    authenticate_user,
    save_student_profile,
    load_student_profile,
    save_career_assessment,
    get_user_career_assessments,
    delete_career_assessment,
    save_resume_scan,
    get_user_resume_scans,
    delete_resume_scan,
    save_chat_message,
    load_user_chat_history,
    clear_user_chat_history,
)
