"""
core/database.py — SQLite Database Layer for EduCareerAI
Provides persistent storage for users, student profiles, assessment history,
resume ATS analysis scans, and AI assistant chat logs.
"""

import os
import sqlite3
import hashlib
import secrets
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

DB_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DB_DIR / "educareer_ai.db"


def get_connection() -> sqlite3.Connection:
    """Returns a SQLite connection with row factory enabled."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    """Initializes the database schema if tables do not exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # 1. Users Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            full_name TEXT,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 2. Student Profiles Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS student_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER UNIQUE NOT NULL,
            target_field TEXT,
            hobbies_json TEXT,
            grades_json TEXT,
            aptitudes_json TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        );
    """)

    # 3. Career Assessments Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS career_assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            top_career TEXT NOT NULL,
            top_confidence REAL,
            top_course TEXT,
            careers_json TEXT NOT NULL,
            courses_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        );
    """)

    # 4. Resume Scans Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resume_scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            resume_name TEXT NOT NULL,
            target_job TEXT,
            ats_score REAL NOT NULL,
            match_score REAL,
            skills_json TEXT,
            issues_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        );
    """)

    # 5. Chat Logs Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        );
    """)

    conn.commit()
    conn.close()


# ============================================================
# AUTHENTICATION & USER MANAGEMENT
# ============================================================

def _hash_password(password: str, salt: str) -> str:
    """Hashes password using PBKDF2 HMAC SHA-256."""
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000
    ).hex()


def register_user(username: str, email: str, password: str, full_name: str = "") -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Registers a new user.
    Returns: (success: bool, message: str, user_dict: Optional[dict])
    """
    username = username.strip().lower()
    email = email.strip().lower()

    if not username or not email or not password:
        return False, "Username, email, and password are required.", None

    if len(password) < 6:
        return False, "Password must be at least 6 characters long.", None

    salt = secrets.token_hex(16)
    pwd_hash = _hash_password(password, salt)

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO users (username, email, full_name, password_hash, salt)
            VALUES (?, ?, ?, ?, ?)
            """,
            (username, email, full_name.strip(), pwd_hash, salt),
        )
        user_id = cursor.lastrowid
        conn.commit()
        user = {
            "id": user_id,
            "username": username,
            "email": email,
            "full_name": full_name.strip(),
        }
        return True, "Registration successful!", user
    except sqlite3.IntegrityError as e:
        err_msg = str(e).lower()
        if "username" in err_msg:
            return False, "Username is already taken.", None
        elif "email" in err_msg:
            return False, "Email address is already registered.", None
        return False, "An account with these details already exists.", None
    finally:
        conn.close()


def authenticate_user(identifier: str, password: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Authenticates a user with username or email + password.
    Returns: (success: bool, message: str, user_dict: Optional[dict])
    """
    identifier = identifier.strip().lower()
    if not identifier or not password:
        return False, "Please enter your username/email and password.", None

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, username, email, full_name, password_hash, salt
        FROM users
        WHERE username = ? OR email = ?
        """,
        (identifier, identifier),
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return False, "Invalid username/email or password.", None

    expected_hash = row["password_hash"]
    salt = row["salt"]
    computed_hash = _hash_password(password, salt)

    if secrets.compare_digest(expected_hash, computed_hash):
        user = {
            "id": row["id"],
            "username": row["username"],
            "email": row["email"],
            "full_name": row["full_name"],
        }
        return True, "Login successful!", user
    else:
        return False, "Invalid username/email or password.", None


# ============================================================
# STUDENT PROFILE CRUD
# ============================================================

def save_student_profile(user_id: int, profile_data: Dict[str, Any]) -> bool:
    """Saves or updates user's student assessment input profile."""
    field = profile_data.get("field") or profile_data.get("target_field", "")
    
    # Handle both nested and flat dictionaries
    hobbies = profile_data.get("hobbies")
    if hobbies is None or not isinstance(hobbies, dict):
        hobbies = {k: v for k, v in profile_data.items() if k.startswith("hobby_") or k.startswith("Hobby_") or k.startswith("H_")}
    
    grades = profile_data.get("grades")
    if grades is None or not isinstance(grades, dict):
        grades = {k: v for k, v in profile_data.items() if k.startswith("grade_") or k.startswith("Grade_") or k.startswith("G_")}
        
    aptitudes = profile_data.get("aptitudes")
    if aptitudes is None or not isinstance(aptitudes, dict):
        aptitudes = {k: v for k, v in profile_data.items() if k.startswith("score_") or k.startswith("Aptitude_") or k.startswith("A_")}

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO student_profiles (user_id, target_field, hobbies_json, grades_json, aptitudes_json, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                target_field = excluded.target_field,
                hobbies_json = excluded.hobbies_json,
                grades_json = excluded.grades_json,
                aptitudes_json = excluded.aptitudes_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                user_id,
                field,
                json.dumps(hobbies),
                json.dumps(grades),
                json.dumps(aptitudes),
            ),
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving profile: {e}")
        return False
    finally:
        conn.close()


def load_student_profile(user_id: int) -> Optional[Dict[str, Any]]:
    """Loads student profile dictionary for a user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT target_field, hobbies_json, grades_json, aptitudes_json FROM student_profiles WHERE user_id = ?",
        (user_id,),
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    hobbies = json.loads(row["hobbies_json"]) if row["hobbies_json"] else {}
    grades = json.loads(row["grades_json"]) if row["grades_json"] else {}
    aptitudes = json.loads(row["aptitudes_json"]) if row["aptitudes_json"] else {}

    return {
        "field": row["target_field"] or "Computer Science & IT",
        "hobbies": hobbies,
        "grades": grades,
        "aptitudes": aptitudes,
    }


# ============================================================
# CAREER & COURSE ASSESSMENTS HISTORY
# ============================================================

def save_career_assessment(
    user_id: int,
    careers_df_or_list: Any,
    courses_list: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """Saves a career & course assessment record to SQLite."""
    careers_data = []
    if hasattr(careers_df_or_list, "to_dict"):
        careers_data = careers_df_or_list.to_dict(orient="records")
    elif isinstance(careers_df_or_list, list):
        careers_data = careers_df_or_list

    top_career = careers_data[0].get("Career", "Unknown") if careers_data else "Unknown"
    top_confidence = careers_data[0].get("Confidence (%)", 0.0) if careers_data else 0.0
    top_course = courses_list[0].get("course", "") if courses_list else ""

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO career_assessments (user_id, top_career, top_confidence, top_course, careers_json, courses_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            top_career,
            float(top_confidence),
            top_course,
            json.dumps(careers_data),
            json.dumps(courses_list or []),
        ),
    )
    assessment_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return assessment_id


def get_user_career_assessments(user_id: int, limit: int = 20) -> List[Dict[str, Any]]:
    """Retrieves past career assessments for a given user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, top_career, top_confidence, top_course, careers_json, courses_json, created_at
        FROM career_assessments
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    )
    rows = cursor.fetchall()
    conn.close()

    results = []
    for r in rows:
        results.append({
            "id": r["id"],
            "top_career": r["top_career"],
            "top_confidence": r["top_confidence"],
            "top_course": r["top_course"],
            "careers": json.loads(r["careers_json"]) if r["careers_json"] else [],
            "courses": json.loads(r["courses_json"]) if r["courses_json"] else [],
            "created_at": r["created_at"],
        })
    return results


def delete_career_assessment(assessment_id: int, user_id: int) -> bool:
    """Deletes a career assessment record belonging to the user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM career_assessments WHERE id = ? AND user_id = ?",
        (assessment_id, user_id),
    )
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0


# ============================================================
# RESUME & ATS SCANS HISTORY
# ============================================================

def save_resume_scan(
    user_id: int,
    resume_name: str,
    ats_score: float,
    match_score: Optional[float] = None,
    target_job: Optional[str] = None,
    extracted_skills: Optional[List[str]] = None,
    issues: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """Saves a resume ATS scan report to SQLite."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO resume_scans (user_id, resume_name, target_job, ats_score, match_score, skills_json, issues_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            resume_name,
            target_job or "",
            float(ats_score),
            float(match_score) if match_score is not None else None,
            json.dumps(extracted_skills or []),
            json.dumps(issues or []),
        ),
    )
    scan_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return scan_id


def get_user_resume_scans(user_id: int, limit: int = 20) -> List[Dict[str, Any]]:
    """Retrieves resume scan history for a user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, resume_name, target_job, ats_score, match_score, skills_json, issues_json, created_at
        FROM resume_scans
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    )
    rows = cursor.fetchall()
    conn.close()

    results = []
    for r in rows:
        results.append({
            "id": r["id"],
            "resume_name": r["resume_name"],
            "target_job": r["target_job"],
            "ats_score": r["ats_score"],
            "match_score": r["match_score"],
            "skills": json.loads(r["skills_json"]) if r["skills_json"] else [],
            "issues": json.loads(r["issues_json"]) if r["issues_json"] else [],
            "created_at": r["created_at"],
        })
    return results


def delete_resume_scan(scan_id: int, user_id: int) -> bool:
    """Deletes a resume scan record belonging to the user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM resume_scans WHERE id = ? AND user_id = ?",
        (scan_id, user_id),
    )
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0


# ============================================================
# CHAT LOGS PERSISTENCE
# ============================================================

def save_chat_message(user_id: int, role: str, content: str) -> None:
    """Saves a single chat message for an authenticated user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_logs (user_id, role, content) VALUES (?, ?, ?)",
        (user_id, role, content),
    )
    conn.commit()
    conn.close()


def load_user_chat_history(user_id: int, limit: int = 50) -> List[Dict[str, str]]:
    """Loads previous chat messages for a user in chronological order."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT role, content
        FROM (
            SELECT role, content, created_at
            FROM chat_logs
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        ) ORDER BY created_at ASC
        """,
        (user_id, limit),
    )
    rows = cursor.fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def clear_user_chat_history(user_id: int) -> None:
    """Clears all chat logs for a user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_logs WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
