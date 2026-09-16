import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# EduCareerAI Unified - Production Release v1.1.2
# Add unified root to Python path
sys.path.insert(0, str(Path(__file__).parent))


# Global cross-version compatibility for unpickling Scikit-Learn pipelines and models
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


from core.career_recommender import (
    predict_top_careers,
    CAREER_FIELDS,
    HOBBY_COLUMNS,
    GRADE_COLUMNS,
    GRADE_LABELS,
    APTITUDE_COLUMNS,
    APTITUDE_LABELS,
)
from core.course_recommender import (
    predict_top_courses,
    CAREER_TO_COURSE_FIELD_MAP,
    load_course_model,
)
from core.resume_analyzer import (
    extract_resume_text,
    detect_resume_sections,
    extract_skills,
    extract_ann_skill_vector,
    calculate_ats_score,
    calculate_job_match,
    generate_rule_based_careers,
    build_resume_issues,
    generate_skill_gap,
    ANN_8_SKILLS,
)
from core.chatbot import (
    generate_llm_response,
    build_system_prompt,
)
from core.dl_models import (
    predict_career_ann,
    predict_digit_cnn,
    compare_rnn_and_lstm,
    analyze_interview_tone_coach,
)

from core.roadmap_generator import (
    get_roadmap_for_career,
    CAREER_ROADMAPS,
)
from core.database import (
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
from core.rag_engine import (
    process_uploaded_file,
    rebuild_index_from_chunks,
    search_relevant_chunks,
    format_doc_context_for_prompt,
    build_source_attribution,
    MAX_CHUNKS_PER_SESSION,
)

# Initialize SQLite database schema
init_db()

# ============================================================
# PAGE CONFIGURATION & STYLES
# ============================================================

st.set_page_config(
    page_title="EduCareerAI — Unified Education & Career Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "student_profile" not in st.session_state:
    st.session_state.student_profile = {
        "field": CAREER_FIELDS[0],
        **{h: 0 for h in HOBBY_COLUMNS},
        **{g: 65.0 for g in GRADE_COLUMNS},
        **{a: 65.0 for a in APTITUDE_COLUMNS},
    }

if "career_recommendations" not in st.session_state:
    st.session_state.career_recommendations = None

if "course_recommendations" not in st.session_state:
    st.session_state.course_recommendations = None

if "selected_roadmap_career" not in st.session_state:
    st.session_state.selected_roadmap_career = "Data Analyst"

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

if "resume_name" not in st.session_state:
    st.session_state.resume_name = ""

if "resume_analysis" not in st.session_state:
    st.session_state.resume_analysis = None

if "last_saved_resume_name" not in st.session_state:
    st.session_state.last_saved_resume_name = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "pending_chat_prompt" not in st.session_state:
    st.session_state.pending_chat_prompt = None

if "ann_skills_input" not in st.session_state:
    st.session_state.ann_skills_input = [1, 1, 0, 0, 0, 1, 1, 1]

# ── RAG Document Store ──────────────────────────────────────────────────────
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []           # List[dict] all chunks from all files
if "doc_vectorizer" not in st.session_state:
    st.session_state.doc_vectorizer = None     # Fitted TfidfVectorizer
if "doc_tfidf_matrix" not in st.session_state:
    st.session_state.doc_tfidf_matrix = None   # Sparse TF-IDF matrix
if "doc_filenames" not in st.session_state:
    st.session_state.doc_filenames = []        # List[str] of loaded file names
if "doc_file_status" not in st.session_state:
    st.session_state.doc_file_status = {}      # {filename: status_str}


# ============================================================
# SIDEBAR NAVIGATION & CONTROLS
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
            <span style="font-size:32px;">🎓</span>
            <div>
                <h2 style="margin:0; font-size:22px; font-weight:800; color:#1E1B4B;">EduCareerAI</h2>
                <span style="font-size:12px; font-weight:600; color:#6366F1;">Unified Intelligence Platform</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # User Account Card & Authentication
    if st.session_state.current_user is None:
        st.markdown(
            """
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:8px 12px; margin-bottom:10px;">
                <span style="font-size:12px; font-weight:600; color:#64748B;">👤 Mode: <b>Guest</b> (Data in-memory)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("🔑 Login / 📝 Register", expanded=False):
            tab_login, tab_register = st.tabs(["Sign In", "Sign Up"])
            with tab_login:
                with st.form("sidebar_login_form"):
                    l_ident = st.text_input("Username or Email", key="sidebar_l_ident")
                    l_pwd = st.text_input("Password", type="password", key="sidebar_l_pwd")
                    l_sub = st.form_submit_button("Sign In", use_container_width=True)
                    if l_sub:
                        ok, msg, user = authenticate_user(l_ident, l_pwd)
                        if ok:
                            st.session_state.current_user = user
                            # Load profile from SQLite if available
                            saved_prof = load_student_profile(user["id"])
                            if saved_prof:
                                st.session_state.student_profile.update(saved_prof)
                            # Load chat logs
                            saved_chats = load_user_chat_history(user["id"])
                            if saved_chats:
                                st.session_state.chat_messages = saved_chats
                            st.toast(f"Welcome back, {user['username']}!", icon="👋")
                            st.rerun()
                        else:
                            st.error(msg)
            with tab_register:
                with st.form("sidebar_register_form"):
                    r_user = st.text_input("Username", key="sidebar_r_user")
                    r_email = st.text_input("Email", key="sidebar_r_email")
                    r_name = st.text_input("Full Name", key="sidebar_r_name")
                    r_pwd = st.text_input("Password (min 6)", type="password", key="sidebar_r_pwd")
                    r_sub = st.form_submit_button("Create Account", use_container_width=True)
                    if r_sub:
                        ok, msg, user = register_user(r_user, r_email, r_pwd, r_name)
                        if ok:
                            st.session_state.current_user = user
                            st.toast("Account created and logged in!", icon="🎉")
                            st.rerun()
                        else:
                            st.error(msg)
    else:
        user = st.session_state.current_user
        display_name = user.get("full_name") or user["username"]
        initial = display_name[0].upper()
        st.markdown(
            f"""
            <div class="user-badge">
                <div class="user-avatar">{initial}</div>
                <div style="overflow:hidden;">
                    <div style="font-weight:700; font-size:14px; color:#1E1B4B; white-space:nowrap; text-overflow:ellipsis; overflow:hidden;">{display_name}</div>
                    <div style="font-size:11px; color:#6366F1;">@{user['username']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_prof, col_out = st.columns(2)
        with col_prof:
            if st.button("💾 Save", key="btn_save_prof_top", help="Save current profile inputs to SQLite", use_container_width=True):
                save_student_profile(user["id"], st.session_state.student_profile)
                st.toast("Profile saved to database!", icon="💾")
        with col_out:
            if st.button("🚪 Logout", key="btn_logout_top", use_container_width=True):
                st.session_state.current_user = None
                st.toast("Logged out.", icon="👋")
                st.rerun()

    st.divider()

    nav_selection = st.radio(
        "Navigation Hub",
        [
            "🏠 Dashboard",
            "🎯 Career & Course Navigator",
            "🧭 Learning Roadmap",
            "📄 Resume & ATS Studio",
            "🤖 AI Assistant",
            "📊 My History & Records",
            "🧠 AI / Deep Learning Lab",
            "ℹ️ About Platform",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Active Session Context")
    if st.session_state.career_recommendations is not None:
        top_c = st.session_state.career_recommendations.iloc[0]["Career"]
        st.markdown(f"🎯 **Top Career:** `{top_c}`")
    if st.session_state.course_recommendations:
        top_crs = st.session_state.course_recommendations[0]["course"]
        st.markdown(f"📚 **Top Course:** `{top_crs}`")
    if st.session_state.resume_name:
        st.markdown(f"📄 **Resume:** `{st.session_state.resume_name}`")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🧹 Reset All Sessions", use_container_width=True):
        st.session_state.career_recommendations = None
        st.session_state.course_recommendations = None
        st.session_state.resume_text = ""
        st.session_state.resume_name = ""
        st.session_state.resume_analysis = None
        st.session_state.last_saved_resume_name = None
        st.session_state.chat_messages = []
        st.rerun()



# ============================================================
# 1. 🏠 DASHBOARD
# ============================================================

if nav_selection == "🏠 Dashboard":
    st.markdown(
        """
        <div class="hero-container">
            <h1 class="hero-title">Welcome to EduCareerAI</h1>
            <p class="hero-subtitle">
                The next-generation unified platform bridging machine learning career ranking, synchronized degree pathways, 
                ATS resume diagnostics, conversational AI assistants, and deep learning neural architectures.
            </p>
            <div class="hero-badge-row">
                <span class="hero-badge">⚡ HistGB ML Classifier</span>
                <span class="hero-badge">🎓 MultiOutput Pathways</span>
                <span class="hero-badge">📄 9-Vector ATS Diagnostics</span>
                <span class="hero-badge">💬 Document-Aware RAG</span>
                <span class="hero-badge">🧠 Neural Lab (CNN, ANN, LSTM)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Trained AI Engines</div>
                <div class="metric-value">6 Models</div>
                <div class="metric-subtext">✨ GGUF • HistGB • MultiOutput • DL</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Career Fields</div>
                <div class="metric-value">10 Sectors</div>
                <div class="metric-subtext">🌐 STEM, Tech, Health, Business</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Assessment Inputs</div>
                <div class="metric-value">27 Features</div>
                <div class="metric-subtext">🎯 16 Hobbies • 5 Grades • 5 Aptitudes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">ATS Diagnostics</div>
                <div class="metric-value">9 Vectors</div>
                <div class="metric-subtext">🛡️ Formatting, Skills, Keywords</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    
    dash_col_chart, dash_col_launch = st.columns([1.1, 1.3])
    
    with dash_col_chart:
        st.markdown("### 🌐 Intelligence Radar")
        st.caption("Multimodal evaluation across all 6 embedded engines")
        
        radar_categories = [
            "Career Ranking",
            "Academic Degrees",
            "ATS Diagnostics",
            "Doc-Aware RAG",
            "Neural Lab",
            "Profile Memory"
        ]
        radar_values = [95, 90, 88, 92, 85, 94]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_values + [radar_values[0]],
            theta=radar_categories + [radar_categories[0]],
            fill='toself',
            fillcolor='rgba(99, 102, 241, 0.25)',
            line=dict(color='#4F46E5', width=2.5),
            marker=dict(size=6, color='#6366F1'),
            name='Engine Capability'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, linecolor='#E2E8F0'),
                angularaxis=dict(tickfont=dict(size=11, family="Plus Jakarta Sans", color="#475569"))
            ),
            showlegend=False,
            margin=dict(l=35, r=35, t=20, b=20),
            height=280,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})

    with dash_col_launch:
        st.markdown("### 🚀 Quick Launch Hubs")
        st.caption("Instant navigation into primary intelligence modules")
        
        q1, q2 = st.columns(2)
        with q1:
            st.markdown(
                """
                <div class="feature-launch-card">
                    <div style="font-size:24px;">🎯</div>
                    <div class="feature-title">Career Navigator</div>
                    <p class="feature-desc">Dual assessment computing Top 5 careers & matching degrees.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with q2:
            st.markdown(
                """
                <div class="feature-launch-card">
                    <div style="font-size:24px;">📄</div>
                    <div class="feature-title">Resume Studio</div>
                    <p class="feature-desc">ATS compliance audits, skill gaps, & 8-skill neural vectors.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        q3, q4 = st.columns(2)
        with q3:
            st.markdown(
                """
                <div class="feature-launch-card">
                    <div style="font-size:24px;">🧭</div>
                    <div class="feature-title">Learning Roadmap</div>
                    <p class="feature-desc">Step-by-step milestones, capstones & certificates.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with q4:
            st.markdown(
                """
                <div class="feature-launch-card">
                    <div style="font-size:24px;">🧠</div>
                    <div class="feature-title">Neural DL Lab</div>
                    <p class="feature-desc">Interactive ANN, CNN digit classifier, & RNN tone coach.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# 2. 🎯 CAREER & COURSE NAVIGATOR
# ============================================================

elif nav_selection == "🎯 Career & Course Navigator":
    st.markdown(
        """
        <div class="hero-container" style="padding:24px 30px; margin-bottom:20px;">
            <h2 style="margin:0; font-size:26px; font-weight:800;">🎯 Unified Career & Course Navigator</h2>
            <p style="margin:4px 0 0 0; font-size:14px; color:#C7D2FE;">
                Single comprehensive student assessment driving both Tabular Machine Learning Career Ranking and Multi-Output Degree Pathways.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("📝 Student Profile & Assessment Form", expanded=(st.session_state.career_recommendations is None)):
        st.subheader("1. Preferred Broad Field")
        selected_field = st.selectbox(
            "Select your field of interest",
            CAREER_FIELDS,
            index=CAREER_FIELDS.index(st.session_state.student_profile.get("field", CAREER_FIELDS[0])),
            key="nav_field_select"
        )
        st.session_state.student_profile["field"] = selected_field

        st.subheader("2. Hobbies & Extracurricular Activities")
        st.caption("Select all activities that you regularly enjoy:")
        hobby_cols = st.columns(4)
        for i, h in enumerate(HOBBY_COLUMNS):
            h_label = h.replace("hobby_", "").replace("_", " ").title()
            with hobby_cols[i % 4]:
                checked = st.checkbox(
                    h_label,
                    value=bool(st.session_state.student_profile.get(h, 0)),
                    key=f"nav_chk_{h}"
                )
                st.session_state.student_profile[h] = 1 if checked else 0

        st.subheader("3. Academic Grades (0 - 100)")
        grade_cols = st.columns(5)
        for i, g in enumerate(GRADE_COLUMNS):
            with grade_cols[i]:
                g_val = st.number_input(
                    GRADE_LABELS[g],
                    min_value=0.0,
                    max_value=100.0,
                    value=float(st.session_state.student_profile.get(g, 65.0)),
                    step=1.0,
                    key=f"nav_num_{g}"
                )
                st.session_state.student_profile[g] = g_val

        st.subheader("4. Aptitude Assessment Scores (0 - 100)")
        apt_cols = st.columns(5)
        for i, a in enumerate(APTITUDE_COLUMNS):
            with apt_cols[i]:
                a_val = st.number_input(
                    APTITUDE_LABELS[a],
                    min_value=0.0,
                    max_value=100.0,
                    value=float(st.session_state.student_profile.get(a, 65.0)),
                    step=1.0,
                    key=f"nav_num_{a}"
                )
                st.session_state.student_profile[a] = a_val

        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("🔮 Generate Unified Career & Course Recommendations", type="primary", use_container_width=True)
        if predict_clicked:
            with st.spinner("Computing machine learning predictions across ranking and multi-output models..."):
                try:
                    # 1. Career Model Prediction
                    careers_df = predict_top_careers(
                        student_dict=st.session_state.student_profile,
                        selected_field=st.session_state.student_profile["field"],
                        top_n=5
                    )
                    st.session_state.career_recommendations = careers_df

                    # 2. Course Model Prediction
                    courses_list = predict_top_courses(
                        student_dict=st.session_state.student_profile,
                        career_field=st.session_state.student_profile["field"]
                    )
                    st.session_state.course_recommendations = courses_list
                    st.session_state.selected_roadmap_career = careers_df.iloc[0]["Career"]

                    # Auto-persist to SQLite if logged in
                    if st.session_state.current_user:
                        save_career_assessment(
                            user_id=st.session_state.current_user["id"],
                            careers_df_or_list=careers_df,
                            courses_list=courses_list,
                        )
                        save_student_profile(
                            user_id=st.session_state.current_user["id"],
                            profile_data=st.session_state.student_profile,
                        )
                        st.toast("Assessment saved to your account history!", icon="💾")

                    st.success("Recommendations successfully computed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Prediction error: {e}")


    # Display Dual Results side-by-side
    if st.session_state.career_recommendations is not None and st.session_state.course_recommendations is not None:
        st.divider()
        
        # Student Profile Radar Chart
        st.markdown("### 📊 Student Competency & Grade Signature")
        prof_col_radar, prof_col_summary = st.columns([1.2, 1])
        
        with prof_col_radar:
            r_categories = [GRADE_LABELS[g] for g in GRADE_COLUMNS] + [APTITUDE_LABELS[a] for a in APTITUDE_COLUMNS]
            r_vals = [float(st.session_state.student_profile.get(g, 65.0)) for g in GRADE_COLUMNS] + [float(st.session_state.student_profile.get(a, 65.0)) for a in APTITUDE_COLUMNS]
            
            fig_stud = go.Figure()
            fig_stud.add_trace(go.Scatterpolar(
                r=r_vals + [r_vals[0]],
                theta=r_categories + [r_categories[0]],
                fill='toself',
                fillcolor='rgba(79, 70, 229, 0.25)',
                line=dict(color='#4F46E5', width=2),
                marker=dict(size=6, color='#6366F1'),
                name='Student Score'
            ))
            fig_stud.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], linecolor='#E2E8F0'),
                    angularaxis=dict(tickfont=dict(size=10, family="Plus Jakarta Sans", color="#475569"))
                ),
                showlegend=False,
                margin=dict(l=40, r=40, t=25, b=25),
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_stud, use_container_width=True, config={'displayModeBar': False})
            
        with prof_col_summary:
            top_career_match = st.session_state.career_recommendations.iloc[0]
            st.markdown(
                f"""
                <div class="glass-panel" style="margin-top:10px;">
                    <div style="font-size:12px; font-weight:700; color:#6366F1; text-transform:uppercase; letter-spacing:0.05em;">AI Consensus Match</div>
                    <div style="font-size:24px; font-weight:800; color:#1E1B4B; margin:4px 0;">{top_career_match['Career']}</div>
                    <div style="font-size:14px; color:#475569; margin-bottom:12px;">Field: <b>{st.session_state.student_profile['field']}</b> • Confidence: <b>{top_career_match['Suitability_Percent']:.1f}%</b></div>
                    <div style="background:#EEF2FF; border-radius:8px; padding:10px 14px; font-size:13px; color:#3730A3;">
                        💡 <b>Recommendation Bridge:</b> This profile exhibits strong quantitative alignment. Check the synchronized degree pathway on the right.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        st.markdown("<br>", unsafe_allow_html=True)
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown("### 🏆 Top 5 Career Recommendations")
            st.caption(f"Engine: HistGradientBoosting • Field: **{st.session_state.student_profile['field']}**")
            
            top_careers = st.session_state.career_recommendations
            for idx, row in top_careers.iterrows():
                rank = idx + 1
                c_name = row["Career"]
                score = row["Suitability_Percent"]
                badge_class = f"rank-card-{rank}" if rank <= 3 else "metric-card"

                st.markdown(
                    f"""
                    <div class="{badge_class}">
                        <div class="rank-header">
                            <div>
                                <span class="rank-title">#{rank} {c_name}</span>
                            </div>
                            <div>
                                <span class="confidence-pill">{score:.2f}% Match</span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                # Action buttons per career
                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    if st.button(f"🧭 View Roadmap", key=f"btn_road_{idx}", use_container_width=True):
                        st.session_state.selected_roadmap_career = c_name
                        st.info(f"Loaded '{c_name}' into the Learning Roadmap hub.")
                with bcol2:
                    if st.button(f"🤖 Ask AI", key=f"btn_chat_{idx}", use_container_width=True):
                        st.session_state.pending_chat_prompt = (
                            f"Create a step-by-step career guide and skill plan for becoming a '{c_name}'."
                        )
                        st.info(f"Prompt sent to AI Assistant for '{c_name}'.")

        with res_col2:
            st.markdown("### 📚 Top 5 Academic Course Pathways")
            mapped_field = CAREER_TO_COURSE_FIELD_MAP.get(st.session_state.student_profile['field'], "STEM")
            st.caption(f"Engine: MultiOutputClassifier • Category Bridge: **{mapped_field}**")

            for item in st.session_state.course_recommendations:
                rank = item["rank"]
                crs_name = item["course"]
                badge_class = f"rank-card-{rank}" if rank <= 3 else "metric-card"

                st.markdown(
                    f"""
                    <div class="{badge_class}">
                        <div class="rank-header">
                            <div>
                                <span class="rank-title">#{rank} {crs_name}</span>
                            </div>
                            <div>
                                <span class="confidence-pill" style="background:#06B6D4;">Target Degree</span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(f"🤖 Ask AI about {crs_name[:20]}...", key=f"btn_crs_chat_{rank}", use_container_width=True):
                    st.session_state.pending_chat_prompt = f"Tell me about the degree '{crs_name}', prerequisites, and career prospects."
                    st.info(f"Course query sent to AI Assistant.")

        # Educational Bridge Notice
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("ℹ️ Transparent Career-to-Course Connection Details"):
            st.info(
                "**ML Architecture Note**: The career recommendations are generated via the `HistGradientBoostingClassifier` "
                "operating on 40 engineered numerical features. The course recommendations are generated via the `MultiOutputClassifier` "
                "with a `ColumnTransformer` encoder. The system uses a transparent rule mapping between career field categories and degree clusters."
            )


# ============================================================
# 3. 🧭 LEARNING ROADMAP
# ============================================================

elif nav_selection == "🧭 Learning Roadmap":
    st.markdown(
        """
        <div class="hero-container" style="padding:24px 30px; margin-bottom:20px;">
            <h2 style="margin:0; font-size:26px; font-weight:800;">🧭 Career Roadmap & Training Engine</h2>
            <p style="margin:4px 0 0 0; font-size:14px; color:#C7D2FE;">
                Structured multi-phase milestones, portfolio capstones, and industry certifications.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    career_options = list(CAREER_ROADMAPS.keys())
    if st.session_state.selected_roadmap_career not in career_options:
        career_options.insert(0, st.session_state.selected_roadmap_career)

    selected_c = st.selectbox(
        "Select Target Career Pathway",
        career_options,
        index=career_options.index(st.session_state.selected_roadmap_career),
        key="roadmap_career_select"
    )
    st.session_state.selected_roadmap_career = selected_c

    roadmap = get_roadmap_for_career(selected_c)

    mcol1, mcol2, mcol3 = st.columns([2, 1, 1])
    with mcol1:
        st.markdown(f"### {roadmap['title']}")
        st.write(roadmap["description"])
    with mcol2:
        st.metric("Estimated Timeline", roadmap["duration"])
    with mcol3:
        if st.button("🤖 Ask AI to Customize", key="btn_custom_road", use_container_width=True):
            st.session_state.pending_chat_prompt = (
                f"Please create a customized, weekly study roadmap for a '{selected_c}' with project milestones."
            )
            st.info(f"Customization prompt queued for AI Assistant.")

    st.markdown("#### 🎯 Core Skill Requirements")
    skills_html = "".join([f'<span class="skill-tag">{s}</span>' for s in roadmap.get("key_skills", [])])
    st.markdown(skills_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### 🚀 Step-by-Step Training Milestones")
    for phase_info in roadmap["phases"]:
        topics_list = "".join([f"<li>{t}</li>" for t in phase_info["topics"]])
        st.markdown(
            f"""
            <div class="phase-card">
                <div class="phase-title">{phase_info['phase']}</div>
                <ul class="phase-topics">{topics_list}</ul>
                <div class="phase-project">💡 Milestone Project: {phase_info['project']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### 📜 Recommended Industry Certifications")
    for cert in roadmap.get("certifications", []):
        st.markdown(f"- 🏅 **{cert}**")


# ============================================================
# 4. 📄 RESUME & ATS STUDIO
# ============================================================

elif nav_selection == "📄 Resume & ATS Studio":
    st.markdown(
        """
        <div class="hero-container" style="padding:24px 30px; margin-bottom:20px;">
            <h2 style="margin:0; font-size:26px; font-weight:800;">📄 Smart Resume & ATS Studio</h2>
            <p style="margin:4px 0 0 0; font-size:14px; color:#C7D2FE;">
                Upload your PDF resume for ATS readiness audits, skill extraction, and deep learning ANN integration.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_up, col_job = st.columns([1, 1])
    with col_up:
        uploaded_pdf = st.file_uploader("Upload Resume (PDF)", type=["pdf"], key="resume_pdf_uploader")
        if uploaded_pdf is not None and uploaded_pdf.name != st.session_state.resume_name:
            try:
                raw_text = extract_resume_text(uploaded_pdf)
                if raw_text:
                    st.session_state.resume_text = raw_text[:8000]
                    st.session_state.resume_name = uploaded_pdf.name
                    st.session_state.resume_analysis = None
                    st.success(f"Loaded '{uploaded_pdf.name}' successfully!")
                else:
                    st.error("No readable text detected in this PDF.")
            except Exception as e:
                st.error(f"Could not parse PDF: {e}")

    with col_job:
        job_desc_input = st.text_area(
            "Target Job Description (Optional)",
            height=120,
            placeholder="Paste target job requirements to calculate keyword match and skill gaps...",
            key="job_desc_text"
        )

    if st.session_state.resume_text:
        # Run Analysis
        sections = detect_resume_sections(st.session_state.resume_text)
        skills = extract_skills(st.session_state.resume_text)
        ats_score, ats_breakdown = calculate_ats_score(st.session_state.resume_text, sections)
        issues = build_resume_issues(st.session_state.resume_text, sections)
        rule_careers = generate_rule_based_careers(skills)

        job_skills = extract_skills(job_desc_input) if job_desc_input else []
        job_score, matched_j, missing_j = calculate_job_match(skills, job_skills)
        skill_gaps = generate_skill_gap(skills, job_skills, rule_careers)

        # Auto-persist scan if user is logged in
        if st.session_state.current_user and st.session_state.last_saved_resume_name != st.session_state.resume_name:
            save_resume_scan(
                user_id=st.session_state.current_user["id"],
                resume_name=st.session_state.resume_name,
                ats_score=ats_score,
                match_score=job_score if job_desc_input else None,
                target_job=job_desc_input[:100] if job_desc_input else "General Review",
                extracted_skills=skills,
                issues=[{"issue": i, "why": w, "fix": f} for i, w, f in issues],
            )
            st.session_state.last_saved_resume_name = st.session_state.resume_name
            st.toast("Resume scan recorded in your account history!", icon="📄")

        st.divider()
        
        gauge_col, metrics_col = st.columns([1, 1.2])
        with gauge_col:
            # Gauge chart for ATS Score
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=ats_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "ATS Readiness Index", 'font': {'size': 18, 'family': 'Outfit', 'color': '#0F172A'}},
                number={'font': {'size': 36, 'family': 'Outfit', 'color': '#0F172A'}, 'suffix': '/100'},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#CBD5E1"},
                    'bar': {'color': "#4F46E5"},
                    'bgcolor': "white",
                    'borderwidth': 1,
                    'bordercolor': "#E2E8F0",
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.15)'},
                        {'range': [50, 75], 'color': 'rgba(245, 158, 11, 0.15)'},
                        {'range': [75, 100], 'color': 'rgba(16, 185, 129, 0.15)'}
                    ],
                    'threshold': {
                        'line': {'color': "#10B981", 'width': 3},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            gauge_fig.update_layout(
                height=240,
                margin=dict(l=20, r=20, t=30, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(gauge_fig, use_container_width=True, config={'displayModeBar': False})

        with metrics_col:
            st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
            if ats_score >= 75:
                status_badge = '<span class="status-pill pill-success" style="font-size:14px;">🟢 ATS Optimized</span>'
                status_desc = "Your resume structure strongly aligns with modern Applicant Tracking Systems."
            elif ats_score >= 50:
                status_badge = '<span class="status-pill pill-info" style="font-size:14px; background:#FEF3C7; color:#B45309;">🟡 Needs Polish</span>'
                status_desc = "Your resume has a solid base but could benefit from targeted structural fixes."
            else:
                status_badge = '<span class="status-pill" style="font-size:14px; background:#FEE2E2; color:#B91C1C;">🔴 High ATS Risk</span>'
                status_desc = "Key resume sections or quantifiable metrics are missing. Check suggestions below."

            st.markdown(
                f"""
                <div class="glass-panel" style="padding:18px 20px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                        <span style="font-weight:700; font-size:15px; color:#1E1B4B;">Diagnostic Status</span>
                        {status_badge}
                    </div>
                    <p style="font-size:13px; color:#475569; margin-bottom:12px;">{status_desc}</p>
                    <div style="display:flex; gap:16px;">
                        <div>
                            <div class="metric-label">Detected Skills</div>
                            <div style="font-size:20px; font-weight:800; color:#4F46E5;">{len(skills)}</div>
                        </div>
                        <div style="border-left:1px solid #E2E8F0; padding-left:16px;">
                            <div class="metric-label">Job Match Alignment</div>
                            <div style="font-size:20px; font-weight:800; color:{'#10B981' if job_desc_input else '#64748B'};">
                                {f'{job_score}%' if job_desc_input else 'N/A (Paste JD)'}
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        tab_overview, tab_breakdown, tab_ann_bridge, tab_issues = st.tabs([
            "📊 Overview & Skills",
            "🎯 ATS Category Breakdown",
            "⚡ Feed to Career ANN",
            "⚠️ Resume Improvements"
        ])

        with tab_overview:
            st.markdown("#### 🛠️ Detected Technical Skills")
            if skills:
                st.markdown("".join([f'<span class="skill-tag skill-tag-matched">✓ {s}</span>' for s in skills]), unsafe_allow_html=True)
            else:
                st.info("No standard technical skills detected from the predefined vocabulary.")

            if job_desc_input:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### 🎯 Job Description Alignment")
                st.write("**Matched Skills:** " + (", ".join(matched_j) if matched_j else "None"))
                st.write("**Missing Keywords:** " + (", ".join(missing_j) if missing_j else "None"))

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 💼 Rule-Based Career Matches")
            for item in rule_careers:
                st.write(f"- **{item['career']}** ({item['match']}% match) — Evidence: {', '.join(item['why'])}")

        with tab_breakdown:
            st.markdown("#### 📊 Category-by-Category ATS Points")
            breakdown_df = pd.DataFrame([
                {"Category": cat, "Score": f"{val} pts"} for cat, val in ats_breakdown.items()
            ])
            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

        with tab_ann_bridge:
            st.markdown("#### ⚡ 8-Skill Neural Vector Extraction")
            st.write("Maps your resume directly to the 8 core skills used by the trained Career ANN:")
            ann_vec = extract_ann_skill_vector(skills)
            ann_cols = st.columns(4)
            for i, skill in enumerate(ANN_8_SKILLS):
                with ann_cols[i % 4]:
                    status = "✅ Present" if ann_vec[i] == 1 else "❌ Not Found"
                    st.write(f"**{skill}**: {status}")

            if st.button("🚀 Run Career ANN on Resume Vector", key="btn_run_ann_resume", type="primary"):
                st.session_state.ann_skills_input = ann_vec
                try:
                    ann_res = predict_career_ann(ann_vec)
                    st.success(f"ANN Top Prediction: **{ann_res['top_career']}** ({ann_res['confidence']:.2f}% confidence)")
                except Exception as e:
                    st.error(f"ANN inference error: {e}")

        with tab_issues:
            st.markdown("#### 🔍 Detected Improvement Opportunities")
            if issues:
                for issue, why, fix in issues:
                    with st.expander(f"⚠️ {issue}"):
                        st.write(f"**Why it matters:** {why}")
                        st.write(f"**Actionable Fix:** {fix}")
            else:
                st.success("No major structural formatting issues detected.")


# ============================================================
# 5. 🤖 AI ASSISTANT
# ============================================================

elif nav_selection == "🤖 AI Assistant":
    st.markdown(
        """
        <div class="hero-container" style="padding:24px 30px; margin-bottom:20px;">
            <h2 style="margin:0; font-size:26px; font-weight:800;">🤖 AI Study & Career Assistant</h2>
            <p style="margin:4px 0 0 0; font-size:14px; color:#C7D2FE;">
                Powered by SmolLM2-135M-Instruct GGUF + document-aware RAG. Upload study files and ask questions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Mode & Sliders ───────────────────────────────────────────────────────
    top_c1, top_c2, top_c3 = st.columns([1.5, 1, 1])
    with top_c1:
        assistant_mode = st.radio(
            "Assistant Mode",
            ["📚 Study Assistant", "💼 Career Assistant"],
            horizontal=True,
            key="assistant_mode_radio"
        )
    with top_c2:
        temperature = st.slider("Creativity (Temperature)", 0.1, 1.0, 0.7, 0.1, key="chat_temp")
    with top_c3:
        max_tokens = st.slider("Response Length (Tokens)", 256, 2048, 1024, 64, key="chat_max_tok")

    # ── Document Upload Panel ────────────────────────────────────────────────
    with st.expander(
        f"📚 Study Documents  —  {'✅ ' + str(len(st.session_state.doc_filenames)) + ' file(s) loaded' if st.session_state.doc_filenames else 'No files uploaded yet'}",
        expanded=not bool(st.session_state.doc_filenames),
    ):
        uploaded_docs = st.file_uploader(
            "Upload PDF, DOCX, or TXT files to ask the AI questions about them:",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="doc_rag_uploader",
            help="Files are processed in-session only — never stored permanently.",
        )

        # Process newly uploaded files
        if uploaded_docs:
            newly_added = False
            for uploaded_file in uploaded_docs:
                fname = uploaded_file.name
                if fname not in st.session_state.doc_filenames:
                    # Guard total chunk count
                    if len(st.session_state.doc_chunks) >= MAX_CHUNKS_PER_SESSION:
                        st.warning(
                            f"⚠️ Session chunk limit ({MAX_CHUNKS_PER_SESSION}) reached. "
                            "Clear some documents before adding more."
                        )
                        break

                    with st.spinner(f"⏳ Processing '{fname}'..."):
                        file_bytes = uploaded_file.read()
                        new_chunks, status_msg = process_uploaded_file(file_bytes, fname)

                    if new_chunks:
                        st.session_state.doc_chunks.extend(new_chunks)
                        st.session_state.doc_filenames.append(fname)
                        st.session_state.doc_file_status[fname] = f"✓ {status_msg}"
                        newly_added = True
                    else:
                        st.session_state.doc_file_status[fname] = f"⚠️ {status_msg}"
                        st.error(f"Could not process '{fname}': {status_msg}")

            # Rebuild TF-IDF index whenever new chunks are added
            if newly_added:
                with st.spinner("🔍 Building search index..."):
                    vectorizer, matrix = rebuild_index_from_chunks(st.session_state.doc_chunks)
                    st.session_state.doc_vectorizer = vectorizer
                    st.session_state.doc_tfidf_matrix = matrix
                st.success(f"📚 {len(st.session_state.doc_filenames)} document(s) ready for Q&A!")
                st.rerun()

        # Show loaded files
        if st.session_state.doc_filenames:
            st.markdown("**Loaded Documents:**")
            for fname in st.session_state.doc_filenames:
                status = st.session_state.doc_file_status.get(fname, "✓ Ready")
                icon = "✅" if status.startswith("✓") else "⚠️"
                st.markdown(
                    f"""<div style="display:flex; align-items:center; gap:10px;
                                    background:#F8FAFC; border:1px solid #E2E8F0;
                                    border-radius:8px; padding:7px 12px; margin:4px 0;">
                        <span style="font-size:18px;">📄</span>
                        <span style="flex:1; font-weight:600; font-size:13px; color:#1E1B4B;">{fname}</span>
                        <span style="font-size:12px; color:#10B981;">{icon} {status}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"<div style='margin-top:8px; font-size:12px; color:#6366F1;'>"
                f"📚 <b>{len(st.session_state.doc_filenames)} document(s)</b> loaded — "
                f"{len(st.session_state.doc_chunks)} chunks indexed</div>",
                unsafe_allow_html=True,
            )

            if st.button("🗑️ Clear All Documents", key="btn_clear_docs", use_container_width=True):
                st.session_state.doc_chunks = []
                st.session_state.doc_vectorizer = None
                st.session_state.doc_tfidf_matrix = None
                st.session_state.doc_filenames = []
                st.session_state.doc_file_status = {}
                st.toast("All documents cleared.", icon="🗑️")
                st.rerun()
        else:
            st.info(
                "📖 **Tip:** Upload your lecture notes, textbooks, or any study material above. "
                "The AI will answer questions directly from your documents AND from general knowledge."
            )

    # ── Quick Suggestions ────────────────────────────────────────────────────
    st.caption("Quick Suggestions:")
    sug_cols = st.columns(4)
    quick_suggestions = [
        "Explain Python decorators with examples",
        "Create a 4-week study plan for SQL",
        "How do I prepare for a Technical ML interview?",
        "Suggest top projects for a DevOps portfolio"
    ]
    for i, (col, suggestion) in enumerate(zip(sug_cols, quick_suggestions)):
        with col:
            if st.button(suggestion[:32] + "…", key=f"quick_sug_{i}", use_container_width=True):
                st.session_state.pending_chat_prompt = suggestion
                st.rerun()

    c_col_clear, _ = st.columns([1, 4])
    with c_col_clear:
        if st.button("🧹 Clear Chat", key="btn_clear_chat_hist", use_container_width=True):
            st.session_state.chat_messages = []
            if st.session_state.current_user:
                clear_user_chat_history(st.session_state.current_user["id"])
            st.toast("Chat history cleared.", icon="🧹")
            st.rerun()

    st.divider()

    # ── Chat History Display ──────────────────────────────────────────────────
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Chat Input ────────────────────────────────────────────────────────────
    _has_docs = bool(st.session_state.doc_filenames)
    _placeholder = (
        "Ask anything about your uploaded documents or any study/career topic..."
        if _has_docs else
        "Ask EduCareer AI anything about studies, roadmaps, or careers..."
    )
    user_input = st.chat_input(_placeholder)

    if st.session_state.pending_chat_prompt:
        user_input = st.session_state.pending_chat_prompt
        st.session_state.pending_chat_prompt = None

    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # ── RAG Retrieval ────────────────────────────────────────────────────
        relevant_chunks = []
        source_block = ""
        doc_context_str = ""
        used_docs = False

        if (
            st.session_state.doc_chunks
            and st.session_state.doc_vectorizer is not None
            and st.session_state.doc_tfidf_matrix is not None
        ):
            relevant_chunks = search_relevant_chunks(
                query=user_input,
                vectorizer=st.session_state.doc_vectorizer,
                matrix=st.session_state.doc_tfidf_matrix,
                chunks=st.session_state.doc_chunks,
            )
            if relevant_chunks:
                doc_context_str = format_doc_context_for_prompt(relevant_chunks)
                source_block = build_source_attribution(relevant_chunks)
                used_docs = True

        # ── Build System Prompt (existing function, enriched with doc context) ─
        career_list = (
            st.session_state.career_recommendations.to_dict(orient="records")
            if st.session_state.career_recommendations is not None else None
        )
        sys_prompt = build_system_prompt(
            mode=assistant_mode,
            resume_text=st.session_state.resume_text,
            ats_score=(
                st.session_state.resume_analysis.get("ats_score")
                if st.session_state.resume_analysis else None
            ),
            career_recs=career_list,
            course_recs=st.session_state.course_recommendations,
        )

        # Append document context + instruction to system prompt when relevant
        if used_docs and doc_context_str:
            sys_prompt += (
                "\n\nIMPORTANT INSTRUCTIONS FOR DOCUMENT CONTEXT:\n"
                "The user has uploaded study documents. Use the DOCUMENT CONTEXT below "
                "when it is relevant to answering the user's question.\n"
                "Do not fabricate information or claim it comes from a document when it does not.\n"
                "If the documents do not contain the answer, use your general knowledge and "
                "indicate this clearly.\n\n"
                + doc_context_str
            )
        elif st.session_state.doc_filenames:
            # Files are loaded but nothing relevant found — tell LLM to use general knowledge
            sys_prompt += (
                "\n\nNote: The user has uploaded documents but the current question does not "
                "appear to be answered by those documents. Please answer using your general knowledge."
            )

        # Build message list (keep last 6 messages for context)
        model_messages = [{"role": "system", "content": sys_prompt}]
        model_messages.extend(st.session_state.chat_messages[-6:])

        # ── Generate Response ────────────────────────────────────────────────
        with st.chat_message("assistant"):
            with st.spinner("AI is formulating response..."):
                response_text = generate_llm_response(
                    messages=model_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

            # Prepend source indicator
            if used_docs:
                display_header = "📄 *Based on your uploaded documents*"
            elif st.session_state.doc_filenames:
                display_header = "🌐 *General AI knowledge (not found in uploaded documents)*"
            else:
                display_header = ""

            full_response = (
                (display_header + "\n\n" if display_header else "")
                + response_text
                + (source_block if used_docs else "")
            )
            st.markdown(full_response)

        st.session_state.chat_messages.append({"role": "assistant", "content": full_response})

        # Persist to SQLite if logged in
        if st.session_state.current_user:
            save_chat_message(st.session_state.current_user["id"], "user", user_input)
            save_chat_message(st.session_state.current_user["id"], "assistant", full_response)


# ============================================================
# 6. 📊 MY HISTORY & RECORDS (SQLITE PERSISTENCE)
# ============================================================

elif nav_selection == "📊 My History & Records":
    st.markdown(
        """
        <div class="hero-container" style="padding:24px 30px; margin-bottom:20px;">
            <h2 style="margin:0; font-size:26px; font-weight:800;">📊 Saved History & Profile Records</h2>
            <p style="margin:4px 0 0 0; font-size:14px; color:#C7D2FE;">
                Persistent SQLite records of your AI career assessments, degree pathways, and resume ATS diagnostic history.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.current_user is None:
        st.warning("⚠️ You are currently in **Guest Mode**. Your sessions are temporarily kept in memory.")
        st.info("Log in or create a free account via the sidebar to save your assessment records, track your ATS improvement, and store customized learning roadmaps!")
        
        # Inline login card for quick access
        with st.container():
            st.markdown("### 🔑 Quick Sign In / Register")
            c_in1, c_in2 = st.columns(2)
            with c_in1:
                with st.form("quick_login_form"):
                    st.markdown("#### Log In")
                    ql_ident = st.text_input("Username / Email", key="ql_ident")
                    ql_pwd = st.text_input("Password", type="password", key="ql_pwd")
                    if st.form_submit_button("Sign In", type="primary", use_container_width=True):
                        ok, msg, u = authenticate_user(ql_ident, ql_pwd)
                        if ok:
                            st.session_state.current_user = u
                            saved_prof = load_student_profile(u["id"])
                            if saved_prof:
                                st.session_state.student_profile.update(saved_prof)
                            saved_chats = load_user_chat_history(u["id"])
                            if saved_chats:
                                st.session_state.chat_messages = saved_chats
                            st.rerun()
                        else:
                            st.error(msg)
            with c_in2:
                with st.form("quick_reg_form"):
                    st.markdown("#### Create Account")
                    qr_user = st.text_input("Username", key="qr_user")
                    qr_email = st.text_input("Email", key="qr_email")
                    qr_name = st.text_input("Full Name", key="qr_name")
                    qr_pwd = st.text_input("Password", type="password", key="qr_pwd")
                    if st.form_submit_button("Register Account", use_container_width=True):
                        ok, msg, u = register_user(qr_user, qr_email, qr_pwd, qr_name)
                        if ok:
                            st.session_state.current_user = u
                            st.rerun()
                        else:
                            st.error(msg)
    else:
        user = st.session_state.current_user
        user_id = user["id"]

        # Fetch records from SQLite
        career_records = get_user_career_assessments(user_id)
        resume_records = get_user_resume_scans(user_id)

        # Overview Metrics
        avg_ats = (
            sum(r["ats_score"] for r in resume_records) / len(resume_records)
            if resume_records else 0.0
        )
        latest_c = career_records[0]["top_career"] if career_records else "None yet"

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Assessments", f"{len(career_records)}")
        with m2:
            st.metric("Resumes Analyzed", f"{len(resume_records)}")
        with m3:
            st.metric("Avg ATS Score", f"{avg_ats:.1f}/100" if resume_records else "N/A")
        with m4:
            st.metric("Latest Career", latest_c)

        st.divider()

        hist_tab1, hist_tab2, hist_tab3 = st.tabs([
            f"🎯 Career Assessments ({len(career_records)})",
            f"📄 Resume ATS Scans ({len(resume_records)})",
            "👤 Academic Profile Settings"
        ])

        # 1. Career Assessments Tab
        with hist_tab1:
            if not career_records:
                st.info("No saved career assessments yet. Run a prediction in '🎯 Career & Course Navigator' to generate one!")
            else:
                for record in career_records:
                    rec_id = record["id"]
                    top_c = record["top_career"]
                    conf = record["top_confidence"]
                    top_crs = record["top_course"] or "General Curriculum"
                    created = record["created_at"]

                    with st.container():
                        st.markdown(
                            f"""
                            <div class="history-card">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <div>
                                        <span style="font-size:16px; font-weight:700; color:#1E1B4B;">🎯 {top_c}</span>
                                        <span class="status-pill pill-success" style="margin-left:10px;">{conf:.1f}% Confidence</span>
                                        <div style="font-size:13px; color:#64748B; margin-top:4px;">
                                            📚 Recommended Degree: <b>{top_crs}</b> • 🕒 Saved on <code>{created}</code>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        btn_c1, btn_c2, _ = st.columns([1, 1, 3])
                        with btn_c1:
                            if st.button(f"📥 Restore Assessment", key=f"btn_load_asmt_{rec_id}", use_container_width=True):
                                if record["careers"]:
                                    st.session_state.career_recommendations = pd.DataFrame(record["careers"])
                                if record["courses"]:
                                    st.session_state.course_recommendations = record["courses"]
                                st.session_state.selected_roadmap_career = top_c
                                st.toast(f"Loaded '{top_c}' into active session!", icon="🎯")
                                st.rerun()
                        with btn_c2:
                            if st.button(f"🗑️ Delete Record", key=f"btn_del_asmt_{rec_id}", use_container_width=True):
                                delete_career_assessment(rec_id, user_id)
                                st.toast("Record deleted.", icon="🗑️")
                                st.rerun()

        # 2. Resume ATS Scans Tab
        with hist_tab2:
            if not resume_records:
                st.info("No resume scans saved yet. Upload a PDF in '📄 Resume & ATS Studio' to record a diagnostic scan!")
            else:
                # Score trend chart
                if len(resume_records) > 1:
                    st.markdown("#### 📈 ATS Score Improvement Trend")
                    trend_df = pd.DataFrame([
                        {"Date": r["created_at"][:16], "ATS Score": r["ats_score"]}
                        for r in reversed(resume_records)
                    ])
                    st.line_chart(trend_df.set_index("Date"))

                for scan in resume_records:
                    scan_id = scan["id"]
                    r_name = scan["resume_name"]
                    score = scan["ats_score"]
                    t_job = scan["target_job"] or "General Evaluation"
                    skills_cnt = len(scan["skills"])
                    issues_cnt = len(scan["issues"])
                    created = scan["created_at"]

                    with st.container():
                        st.markdown(
                            f"""
                            <div class="history-card">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <div>
                                        <span style="font-size:16px; font-weight:700; color:#1E1B4B;">📄 {r_name}</span>
                                        <span class="status-pill pill-info" style="margin-left:10px;">ATS: {score:.1f}/100</span>
                                        <div style="font-size:13px; color:#64748B; margin-top:4px;">
                                            🎯 Target: <b>{t_job}</b> • 🛠️ {skills_cnt} Skills • ⚠️ {issues_cnt} Suggestions • 🕒 <code>{created}</code>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        del_c, _ = st.columns([1, 4])
                        with del_c:
                            if st.button(f"🗑️ Delete Scan", key=f"btn_del_scan_{scan_id}", use_container_width=True):
                                delete_resume_scan(scan_id, user_id)
                                st.toast("Scan record deleted.", icon="🗑️")
                                st.rerun()

        # 3. Profile Tab
        with hist_tab3:
            st.markdown("#### 👤 Student Profile Data")
            st.write(f"**Username:** `{user['username']}` | **Email:** `{user['email']}`")
            st.write(f"**Target Field:** `{st.session_state.student_profile.get('field', 'Not set')}`")
            
            with st.expander("🔍 View Active Profile Feature Values", expanded=False):
                st.json(st.session_state.student_profile)

            if st.button("💾 Sync Current Form Inputs to SQLite", key="btn_sync_profile_db", type="primary"):
                save_student_profile(user_id, st.session_state.student_profile)
                st.toast("Profile data successfully saved to SQLite database!", icon="💾")


# ============================================================
# 7. 🧠 AI / DEEP LEARNING LAB
# ============================================================

elif nav_selection == "🧠 AI / Deep Learning Lab":

    st.markdown(
        """
        <div class="hero-container" style="padding:24px 30px; margin-bottom:20px;">
            <h2 style="margin:0; font-size:26px; font-weight:800;">🧠 AI & Deep Learning Innovation Lab</h2>
            <p style="margin:4px 0 0 0; font-size:14px; color:#C7D2FE;">
                Hands-on interactive neural playground for ANN, CNN, RNN, and LSTM architectures.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    lab_tab1, lab_tab2, lab_tab3 = st.tabs([
        "⚡ Career ANN (8 Skills → Career)",
        "✍️ MNIST Digit CNN (28x28 → Digit)",
        "🎤 AI Interview & Tone Coach (Recurrent NLP)"
    ])

    # 1. ANN Tab
    with lab_tab1:
        st.markdown("### ⚡ Artificial Neural Network (ANN) Career Predictor")
        st.caption("Architecture: Dense(16, ReLU) → Dropout(0.2) → Dense(8, ReLU) → Dense(5, Softmax)")
        st.write("Toggle your technical skills to predict real-time softmax career probabilities:")

        ann_cols = st.columns(4)
        active_skills = []
        for i, skill in enumerate(ANN_8_SKILLS):
            with ann_cols[i % 4]:
                default_val = bool(st.session_state.ann_skills_input[i]) if len(st.session_state.ann_skills_input) > i else False
                val = st.checkbox(skill, value=default_val, key=f"ann_toggle_{skill}")
                active_skills.append(1 if val else 0)

        st.session_state.ann_skills_input = active_skills

        if st.button("🔮 Predict Career via ANN", key="btn_ann_pred", type="primary"):
            try:
                res = predict_career_ann(active_skills)
                st.markdown(
                    f"""
                    <div class="glass-panel" style="padding:16px 20px; margin-top:12px;">
                        <span style="font-size:13px; font-weight:700; color:#6366F1; text-transform:uppercase;">ANN Consensus Prediction</span>
                        <div style="font-size:22px; font-weight:800; color:#1E1B4B; margin-top:2px;">
                            🎯 {res['top_career']} <span class="status-pill pill-success" style="font-size:13px; margin-left:8px;">{res['confidence']:.2f}% Confidence</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Plotly Probability distribution
                prob_df = pd.DataFrame(res["breakdown"])
                fig_ann = px.bar(
                    prob_df,
                    x="probability",
                    y="career",
                    orientation="h",
                    text=[f"{p:.1f}%" for p in prob_df["probability"]],
                    labels={"probability": "Softmax Probability (%)", "career": "Career Role"},
                    color="probability",
                    color_continuous_scale=["#C7D2FE", "#4F46E5"]
                )
                fig_ann.update_layout(
                    height=240,
                    margin=dict(l=20, r=20, t=15, b=15),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    coloraxis_showscale=False,
                    yaxis={'categoryorder':'total ascending'}
                )
                fig_ann.update_traces(textposition='outside')
                st.plotly_chart(fig_ann, use_container_width=True, config={'displayModeBar': False})
            except Exception as e:
                st.error(f"ANN execution error: {e}")

    # 2. CNN Tab
    with lab_tab2:
        st.markdown("### ✍️ Convolutional Neural Network (CNN) Digit Classifier")
        st.caption("Architecture: Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(64) → Dense(10, Softmax)")
        st.write("Select an authentic handwritten digit from the MNIST test benchmark:")

        c_pick1, c_pick2 = st.columns([1, 1])
        with c_pick1:
            digit_choice = st.selectbox(
                "Select Target Digit",
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                index=2,
                key="cnn_digit_select"
            )
        with c_pick2:
            sample_variant = st.selectbox(
                "Select Sample Variation",
                [1, 2, 3, 4, 5],
                format_func=lambda x: f"Sample #{x}",
                index=0,
                key="cnn_sample_var"
            )

        # Load authentic MNIST sample from local benchmark dataset
        sample_path = Path(__file__).parent / "data" / "mnist_benchmark_samples.npz"
        if sample_path.exists():
            data_npz = np.load(sample_path)
            sample_key = f"{digit_choice}_sample_{sample_variant}"
            if sample_key in data_npz:
                sample_img = data_npz[sample_key]
            else:
                sample_img = data_npz.get(f"{digit_choice}_sample_1", np.zeros((28, 28), dtype="float32"))
        else:
            sample_img = np.zeros((28, 28), dtype="float32")

        c_col1, c_col2 = st.columns([1, 2])
        with c_col1:
            st.image(
                sample_img,
                caption=f"Handwritten Digit {digit_choice} (Sample #{sample_variant})",
                width=170,
                clamp=True
            )
        with c_col2:
            if st.button("🔍 Classify Image with CNN", key="btn_cnn_pred", type="primary"):
                try:
                    cnn_res = predict_digit_cnn(sample_img)
                    p_digit = cnn_res['predicted_digit']
                    p_conf = cnn_res['confidence']
                    
                    if p_digit == digit_choice:
                        st.success(f"🎯 Correct Match! Predicted Digit: **{p_digit}** ({p_conf:.2f}% confidence)")
                    else:
                        st.warning(f"Predicted Digit: **{p_digit}** ({p_conf:.2f}% confidence)")

                    cnn_df = pd.DataFrame({
                        "Digit": [f"Digit {d}" for d in range(10)],
                        "Probability": cnn_res["probabilities"]
                    })
                    fig_cnn = px.bar(
                        cnn_df,
                        x="Digit",
                        y="Probability",
                        labels={"Probability": "Confidence (%)", "Digit": "Class"},
                        color="Probability",
                        color_continuous_scale=["#E0E7FF", "#6366F1"]
                    )
                    fig_cnn.update_layout(
                        height=220,
                        margin=dict(l=10, r=10, t=10, b=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        coloraxis_showscale=False
                    )
                    st.plotly_chart(fig_cnn, use_container_width=True, config={'displayModeBar': False})
                except Exception as e:
                    st.error(f"CNN execution error: {e}")

    # 3. 🎤 AI Interview & Cover Letter Tone Coach Tab
    with lab_tab3:
        st.markdown("### 🎤 AI Interview & Cover Letter Tone Coach")
        st.caption("Powered by Recurrent Neural Networks (LSTM & SimpleRNN) for sequential tone, confidence & impact evaluation.")

        # ── Question Bank & Practice ──────────────────────────────────────────
        INTERVIEW_QUESTION_BANK = {
            "💼 Behavioral (STAR Method)": [
                "Tell me about a time you faced a major technical challenge and how you resolved it.",
                "Describe a situation where you had to learn a new technology quickly under pressure.",
                "Give an example of a time you collaborated with a difficult team member to deliver a project.",
                "Tell me about a project where you took ownership and drove it to completion independently.",
                "Describe a time when you made a mistake in your work. What did you learn from it?",
                "Tell me about a time you had to prioritize multiple deadlines simultaneously.",
                "Give an example where you disagreed with your manager. How did you handle it?",
                "Describe a situation where you improved an inefficient process.",
                "Tell me about your most impactful technical accomplishment so far.",
                "Describe a time you received critical feedback and how you responded.",
            ],
            "🐍 Data Science & Machine Learning": [
                "Walk me through how you would build a churn prediction model from scratch.",
                "Explain the bias-variance tradeoff and how you mitigate it.",
                "What is the difference between L1 (Lasso) and L2 (Ridge) regularization?",
                "How would you handle a highly imbalanced dataset in a classification task?",
                "Explain the difference between bagging and boosting ensemble methods.",
                "What evaluation metrics would you use for a fraud detection model and why?",
                "Explain how a Random Forest handles feature importance.",
                "What is cross-validation and why is it preferred over a single train-test split?",
                "Describe how you would deploy a machine learning model to production.",
                "What is the difference between supervised, unsupervised, and reinforcement learning?",
            ],
            "💻 Software Engineering": [
                "Explain the SOLID principles with a real-world example.",
                "What is the difference between REST and GraphQL APIs?",
                "How would you design a URL shortening service like bit.ly?",
                "Explain the concept of database indexing and when you would use it.",
                "What is the difference between horizontal and vertical scaling?",
                "How does a hash table work internally?",
                "Explain the CAP theorem in distributed systems.",
                "What are the differences between SQL and NoSQL databases?",
                "How would you optimize a slow database query?",
                "Explain microservices architecture vs. monolithic architecture.",
            ],
            "⚙️ DevOps & Cloud": [
                "Explain the CI/CD pipeline you have worked with. What tools did you use?",
                "What is the difference between Docker containers and virtual machines?",
                "How does Kubernetes handle auto-scaling of containerized workloads?",
                "Explain blue-green deployment vs. canary deployment strategies.",
                "What is Infrastructure as Code and which IaC tools have you used?",
                "How would you monitor a production system and set up alerting?",
                "Explain the difference between AWS EC2, Lambda, and ECS.",
                "What is a service mesh and when would you use Istio?",
                "How do you securely manage secrets and credentials in a cloud environment?",
                "What is GitOps and how does ArgoCD implement it?",
            ],
            "🔒 Cyber Security": [
                "What is the difference between authentication and authorization?",
                "Explain SQL injection and how you would prevent it.",
                "What is a Man-in-the-Middle (MITM) attack and how can it be mitigated?",
                "What is the CIA triad in information security?",
                "How does HTTPS/TLS protect data in transit?",
                "What is a penetration test and what are its phases?",
                "Explain the OWASP Top 10 vulnerabilities.",
                "What is the principle of least privilege and why is it important?",
                "How would you respond to a ransomware incident?",
                "What is Zero Trust architecture?",
            ],
            "🌐 Web Development": [
                "Explain the difference between client-side and server-side rendering.",
                "What is the event loop in JavaScript and how does it work?",
                "How does React's virtual DOM differ from the real DOM?",
                "Explain HTTP/2 improvements over HTTP/1.1.",
                "What is CORS and how do you handle it?",
                "How would you optimize a web page for performance?",
                "What is server-side rendering (SSR) vs. static site generation (SSG)?",
                "Explain the difference between localStorage, sessionStorage, and cookies.",
                "What is a REST API and what are its key constraints?",
                "How does OAuth 2.0 work for user authentication?",
            ],
            "🤝 HR & Situational": [
                "Why do you want to work at our company specifically?",
                "Where do you see yourself in 5 years?",
                "What is your greatest professional strength?",
                "Describe your ideal work environment.",
                "How do you stay updated with the latest technology trends?",
                "What is your approach to handling tight deadlines?",
                "Why are you leaving your current position?",
                "What do you consider your most valuable transferable skill?",
                "How do you handle failure or setbacks?",
                "Tell me something about yourself not on your resume.",
            ],
        }

        SAMPLE_ANSWERS = {
            "🌟 High-Impact STAR Response (Engineering Incident)": (
                "When our microservices faced severe API latency spikes during high traffic, I spearheaded the investigation "
                "and engineered an automated Redis caching layer to reduce database load. I resolved the bottleneck, "
                "accelerated response times by 65%, and saved the team over 15 hours of manual debugging per week."
            ),
            "⚠️ Passive / Hesitant Response (Needs Coaching)": (
                "I think I was responsible for doing some data analysis. I sort of tried to build a small script in Python, "
                "but I only know a little bit about machine learning. I guess it helped the team a bit."
            ),
            "📄 Strong Cover Letter Opening": (
                "I am excited to apply for the Data Science role at your organization. During my capstone project, I architected "
                "a customer churn prediction pipeline using Random Forests and Scikit-Learn that achieved 92% precision, "
                "directly reducing customer attrition by 18% and optimizing retention campaign targeting."
            ),
            "🎓 Fresher Introducing Themselves": (
                "I recently graduated in Computer Science with a specialization in Machine Learning. During my studies, "
                "I built three end-to-end projects: a sentiment analysis pipeline, a real-time object detection system, "
                "and a career recommendation engine. I am eager to apply this technical foundation in a production environment "
                "and collaborate with experienced engineering teams."
            ),
            "💻 Software Engineer STAR (System Design)": (
                "During a peak traffic event, our monolithic backend experienced critical performance degradation. "
                "I designed and implemented a microservices decomposition strategy, migrating the most load-intensive modules "
                "to isolated Docker containers orchestrated via Kubernetes. This architectural refactoring reduced system latency "
                "by 80% and increased fault isolation, enabling zero-downtime deployments."
            ),
            "✍️ Custom Interview Response": ""
        }

        # ── Section 1: Question Bank ───────────────────────────────────────────
        with st.expander("📚 Interview Question Bank — Browse by Domain", expanded=True):
            qbank_domain = st.selectbox(
                "Select Interview Category:",
                list(INTERVIEW_QUESTION_BANK.keys()),
                key="qbank_domain_select"
            )
            questions_list = INTERVIEW_QUESTION_BANK[qbank_domain]

            st.markdown(f"**{len(questions_list)} Questions — {qbank_domain}**")
            for i, q in enumerate(questions_list, 1):
                col_q, col_load = st.columns([10, 2])
                with col_q:
                    st.markdown(f"**{i}.** {q}")
                with col_load:
                    if st.button("Practice ↓", key=f"qbank_load_{qbank_domain}_{i}", help="Load this question to practice"):
                        st.session_state["tone_practice_question"] = q
                        st.session_state["tone_practice_area"] = ""
                        st.toast(f"Question {i} loaded! Scroll down to answer.", icon="🎤")

        st.divider()

        # ── Section 2: Practice Arena ─────────────────────────────────────────
        st.markdown("#### 🎯 Practice Arena — Answer & Get AI Coaching")

        if "tone_practice_question" in st.session_state and st.session_state["tone_practice_question"]:
            st.info(f"❓ **Practice Question:** {st.session_state['tone_practice_question']}")

        col_preset, col_sample = st.columns([3, 2])
        with col_preset:
            preset_choice = st.selectbox(
                "Or load a sample answer template:",
                list(SAMPLE_ANSWERS.keys()),
                index=0,
                key="tone_coach_preset_select"
            )
        with col_sample:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📋 Load Sample Answer", key="btn_load_sample_ans"):
                st.session_state["tone_prefill"] = SAMPLE_ANSWERS[preset_choice]

        prefill_text = st.session_state.get("tone_prefill", SAMPLE_ANSWERS["🌟 High-Impact STAR Response (Engineering Incident)"])
        user_speech = st.text_area(
            "✍️ Your Interview Answer / Cover Letter Paragraph:",
            value=prefill_text,
            height=140,
            placeholder=(
                "Type your response here. Use the STAR method:\n"
                "• Situation: Set the context\n"
                "• Task: What was your goal?\n"
                "• Action: What exactly did YOU do?\n"
                "• Result: What measurable outcome did you achieve?"
            ),
            key="tone_coach_input_area"
        )

        col_analyze, col_clear = st.columns([3, 1])
        with col_analyze:
            analyze_clicked = st.button("⚡ Analyze Tone & Communication Impact", key="btn_eval_tone_coach", type="primary", use_container_width=True)
        with col_clear:
            if st.button("🗑️ Clear", key="btn_clear_tone", use_container_width=True):
                st.session_state["tone_prefill"] = ""
                st.session_state.pop("tone_practice_question", None)
                st.rerun()

        if analyze_clicked:
            if not user_speech.strip():
                st.warning("Please enter a response to analyze.")
            else:
                with st.spinner("🧠 Running LSTM & RNN analysis..."):
                    try:
                        eval_result = analyze_interview_tone_coach(user_speech)

                        st.divider()
                        # ── Score Banner
                        score = eval_result["impact_score"]
                        score_color = "#10B981" if score >= 80 else "#6366F1" if score >= 55 else "#F59E0B"
                        st.markdown(
                            f"""
                            <div style="background:linear-gradient(135deg,{score_color}22,{score_color}11);
                                        border:2px solid {score_color}44; border-radius:14px;
                                        padding:16px 24px; margin-bottom:16px; display:flex;
                                        align-items:center; gap:20px;">
                                <div style="font-size:48px; font-weight:900; color:{score_color};">{score:.0f}</div>
                                <div>
                                    <div style="font-size:20px; font-weight:800; color:#1E1B4B;">{eval_result['tone_badge']}</div>
                                    <div style="font-size:14px; color:#64748B; margin-top:4px;">{eval_result['tone_label']} — Word Count: {eval_result['word_count']} words</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # ── 4 Metric Columns
                        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
                        with t_col1:
                            st.metric("🎯 Impact Score", f"{score}/100")
                        with t_col2:
                            st.metric("📝 Tone Category", eval_result["tone_label"].split(" ")[0])
                        with t_col3:
                            st.metric("💪 Action Verbs", f"{len(eval_result['found_action_verbs'])}")
                        with t_col4:
                            st.metric("⭐ STAR Pillars", f"{eval_result['star_score']}/4")

                        # ── Detailed Tabs
                        coach_t1, coach_t2, coach_t3, coach_t4 = st.tabs([
                            "🎯 STAR & Tone Diagnostics",
                            "🧠 LSTM vs RNN Neural Analysis",
                            "✨ Executive Rephrasing",
                            "📖 Coaching Tips & Best Practices"
                        ])

                        with coach_t1:
                            st.markdown("#### 🌟 STAR Framework Alignment")
                            s_map = eval_result["star_breakdown"]
                            star_cols = st.columns(4)
                            star_items = [
                                ("Situation", "Set the scene & context", s_map["Situation"]),
                                ("Task", "State your goal/challenge", s_map["Task"]),
                                ("Action", "Describe YOUR specific actions", s_map["Action"]),
                                ("Result", "Quantify the outcome (%, $, time)", s_map["Result"]),
                            ]
                            for col, (label, tip, present) in zip(star_cols, star_items):
                                with col:
                                    status = "✅" if present else "❌"
                                    color = "#10B981" if present else "#EF4444"
                                    st.markdown(
                                        f"""<div style="border:1px solid {color}33; border-radius:10px; padding:12px; text-align:center; background:{color}11;">
                                            <div style="font-size:22px;">{status}</div>
                                            <div style="font-weight:700; color:#1E1B4B;">{label}</div>
                                            <div style="font-size:11px; color:#64748B; margin-top:4px;">{tip}</div>
                                        </div>""",
                                        unsafe_allow_html=True,
                                    )

                            st.markdown("<br>", unsafe_allow_html=True)

                            # Action Verbs Found
                            if eval_result["found_action_verbs"]:
                                st.markdown("#### 💪 Strong Action Verbs Detected:")
                                verb_html = " ".join([
                                    f'<span style="background:#10B98122; color:#10B981; border:1px solid #10B98144; border-radius:6px; padding:3px 10px; font-size:13px; font-weight:600; margin:3px; display:inline-block;">{v}</span>'
                                    for v in eval_result["found_action_verbs"]
                                ])
                                st.markdown(verb_html, unsafe_allow_html=True)

                            # Weak Phrases
                            if eval_result["found_weak_phrases"]:
                                st.markdown("<br>", unsafe_allow_html=True)
                                st.markdown("#### ⚠️ Weak / Hesitant Phrases Detected:")
                                for item in eval_result["found_weak_phrases"]:
                                    st.warning(f"🔄 Replace **'{item['phrase']}'** ➔ **'{item['fix']}'** for stronger authority.")
                            else:
                                st.success("🎉 No weak or hesitant filler phrases detected! Your language is assertive.")

                        with coach_t2:
                            st.markdown("#### 🧠 Recurrent Neural Network Sequential Analysis")
                            st.caption("LSTM uses gated memory cells to capture long-range dependencies, giving more accurate sentiment than SimpleRNN.")

                            rnn_d = eval_result["recurrent_eval"]["rnn"]
                            lstm_d = eval_result["recurrent_eval"]["lstm"]

                            r_col1, r_col2 = st.columns(2)
                            with r_col1:
                                st.markdown(
                                    f"""<div class="metric-card">
                                        <div class="metric-label">🔁 SimpleRNN (Vanilla)</div>
                                        <div class="metric-value">{rnn_d['sentiment']} {rnn_d['emoji']}</div>
                                        <div style="font-size:14px;margin-top:8px;">Confidence: <b>{rnn_d['confidence']:.2f}%</b></div>
                                        <div style="font-size:12px;color:#64748B;">Sigmoid Output: {rnn_d['raw_score']:.4f}</div>
                                        <div style="font-size:11px;color:#94A3B8;margin-top:6px;">⚠️ Prone to vanishing gradient — loses early context</div>
                                    </div>""",
                                    unsafe_allow_html=True,
                                )
                            with r_col2:
                                st.markdown(
                                    f"""<div class="metric-card">
                                        <div class="metric-label">🧠 LSTM (Long Short-Term Memory)</div>
                                        <div class="metric-value">{lstm_d['sentiment']} {lstm_d['emoji']}</div>
                                        <div style="font-size:14px;margin-top:8px;">Confidence: <b>{lstm_d['confidence']:.2f}%</b></div>
                                        <div style="font-size:12px;color:#64748B;">Sigmoid Output: {lstm_d['raw_score']:.4f}</div>
                                        <div style="font-size:11px;color:#10B981;margin-top:6px;">✅ Gated memory — retains context over long sequences</div>
                                    </div>""",
                                    unsafe_allow_html=True,
                                )

                            st.markdown("""
                            **🔬 Why LSTM > SimpleRNN for Interview Analysis:**
                            - **Forget Gate:** Filters irrelevant short-term words (filler phrases)
                            - **Input Gate:** Strengthens encoding of impact words (action verbs, metrics)
                            - **Output Gate:** Produces a context-aware sentiment embedding
                            """)

                        with coach_t3:
                            st.markdown("#### ✨ AI-Powered Executive Rephrasing")
                            if eval_result["found_weak_phrases"]:
                                st.markdown("**Before → After Transformation:**")
                                st.markdown(
                                    f"""<div style="background:#FEFCE8; border-left:4px solid #F59E0B; border-radius:8px; padding:14px 18px; font-size:14px; line-height:1.7;">
                                    {eval_result['rephrased_preview']}
                                    </div>""",
                                    unsafe_allow_html=True,
                                )
                                st.caption("Bold phrases indicate suggested replacements for greater executive impact.")
                            else:
                                st.success("✅ Your response is already polished with strong executive language!")
                                st.markdown(
                                    f"""<div style="background:#F0FDF4; border-left:4px solid #10B981; border-radius:8px; padding:14px 18px; font-size:14px; line-height:1.7;">
                                    {user_speech}
                                    </div>""",
                                    unsafe_allow_html=True,
                                )

                        with coach_t4:
                            st.markdown("#### 📖 Interview Coaching Best Practices")
                            tips_col1, tips_col2 = st.columns(2)
                            with tips_col1:
                                st.markdown("""
**🌟 STAR Method Framework:**
- **S — Situation:** Set the scene. What was the context? Who was involved?
- **T — Task:** What specific challenge or goal were you responsible for?
- **A — Action:** What exact steps did *you* personally take? Use strong action verbs.
- **R — Result:** What measurable outcome did you achieve? (%, $, time saved, users impacted)

---

**💡 Power Action Verbs to Use:**
| Domain | Verbs |
|--------|-------|
| Engineering | Architected, Engineered, Optimized, Deployed |
| Leadership | Spearheaded, Orchestrated, Mentored, Aligned |
| Impact | Accelerated, Reduced, Increased, Delivered |
| Data | Analyzed, Modeled, Visualized, Automated |

---

**⚠️ Phrases to AVOID:**
- ~~"I think I was..."~~ → "I led / I delivered"
- ~~"I sort of tried to..."~~ → "I executed / I implemented"
- ~~"Was responsible for..."~~ → "Orchestrated / Managed"
- ~~"Just" / "basically"~~ → Be specific and direct
""")
                            with tips_col2:
                                st.markdown("""
**🎯 Length Guidelines:**
- **Behavioral Questions:** 90–150 words (60–90 seconds spoken)
- **Technical Questions:** 100–200 words with concrete examples
- **HR/Culture Questions:** 50–100 words, genuine & concise
- **Cover Letter Paragraph:** 80–120 words per section

---

**🚀 Interview Confidence Formula:**
1. **Prepare 5 strong STAR stories** adaptable to any question
2. **Quantify everything:** "improved by ~30%" beats "improved significantly"
3. **Mirror the job description** — use their exact keywords
4. **Pause before answering** — 3 seconds of thinking = confidence
5. **Close every behavioral answer** with a lesson learned

---

**📊 Score Interpretation:**
| Score | Meaning |
|-------|---------|
| 80–100 | 🌟 Executive & High Impact |
| 55–79  | 👍 Professional & Clear |
| 15–54  | ⚠️ Needs Assertiveness |
""")

                    except Exception as e:
                        st.error(f"Tone coaching analysis error: {e}")

        # ── Section 3: Quick Coaching Reference (always visible) ────────────────
        st.divider()
        with st.expander("💡 Quick Reference: Strong vs Weak Language Cheat Sheet", expanded=False):
            st.markdown("""
| ❌ Weak / Passive Phrase | ✅ Strong / Executive Replacement |
|--------------------------|-----------------------------------|
| "I think I was responsible for..." | "I led / I owned / I drove..." |
| "I sort of tried to..." | "I executed / I implemented / I delivered..." |
| "I feel like it worked..." | "The outcome demonstrated / Results confirmed..." |
| "I just did some analysis..." | "I conducted a comprehensive analysis of..." |
| "Was responsible for..." | "Orchestrated / Managed / Directed..." |
| "I only know a little about..." | "I have foundational expertise in..." and I am actively expanding..." |
| "I guess it helped..." | "This initiative resulted in a measurable improvement of..." |
| "Basically..." | Remove it — be direct and specific |
| "Kind of / Sort of" | Use precise qualifiers instead |
| "I tried to..." | "I delivered / I engineered / I resolved..." |
""")

        with st.expander("📋 Common Interview Question Frameworks", expanded=False):
            st.markdown("""
**1. "Tell me about yourself" — The Professional Pitch Formula (60 seconds):**
> *[Current Role/Education] + [Key Technical Strengths] + [Biggest Achievement] + [Why This Company/Role]*

**2. "Why do you want this role?" — The Alignment Framework:**
> *[Specific company mission/product] + [How your skills match] + [What you want to learn/contribute]*

**3. "Where do you see yourself in 5 years?" — The Growth Narrative:**
> *[Short-term technical mastery] + [Mid-term leadership/impact] + [Long-term vision aligned with company]*

**4. "What's your greatest weakness?" — The Growth Story:**
> *[Real but non-critical weakness] + [Steps you've already taken] + [Measurable progress made]*

**5. "Why are you leaving your current role?" — The Forward Focus:**
> *[Positive framing only] + [Seeking growth/challenge] + [Excited about THIS opportunity]*
""")



# ============================================================
# 7. ℹ️ ABOUT PLATFORM
# ============================================================

elif nav_selection == "ℹ️ About Platform":
    st.markdown(
        """
        <div class="hero-container" style="padding:24px 30px; margin-bottom:20px;">
            <h2 style="margin:0; font-size:26px; font-weight:800;">ℹ️ About EduCareerAI</h2>
            <p style="margin:4px 0 0 0; font-size:14px; color:#C7D2FE;">
                Architectural specification, model lineage, and verified technical components.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### 🏗️ Technical Architecture & Components

        EduCareerAI brings together three distinct domains of machine intelligence:

        1. **Tabular Machine Learning**:
           - **Career Ranking Engine**: `HistGradientBoostingClassifier` trained on 40 engineered numerical features across grades, aptitudes, and extracurricular activities with keyword compatibility boosts.
           - **Course Pathway Engine**: `MultiOutputClassifier` with Scikit-Learn `ColumnTransformer` preprocessing categorical fields and continuous scores.

        2. **Conversational LLM**:
           - 4-bit Quantized `SmolLM2-135M-Instruct` executed locally via `llama-cpp-python` with real-time prompt injection of resume facts, ATS diagnostics, and active career recommendations.

        3. **Deep Learning Neural Networks (Keras / TensorFlow)**:
           - **Career ANN**: Multi-layer perceptron mapping binary technical skill vectors to 5 tech career classes.
           - **MNIST CNN**: 2D Convolutional neural network recognizing visual digit patterns.
           - **SimpleRNN & LSTM**: Sequential recurrent networks trained on the IMDB sentiment benchmark.

        4. **Resume & ATS Diagnostic Suite**:
           - Deterministic parsing algorithms evaluating 9 key hiring criteria: contact completeness, structural sections, technical skills, measurable bullet achievements, and job description alignment.
        """
    )
