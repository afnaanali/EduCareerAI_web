import re
from pypdf import PdfReader

ATS_WEIGHTS = {
    "Contact Information": 10,
    "Sections": 15,
    "Skills": 15,
    "Experience/Projects": 15,
    "Keywords": 15,
    "Achievements": 10,
    "Formatting": 10,
    "Education": 5,
    "Links": 5,
}

SECTION_ALIASES = {
    "contact": ["contact", "contact information", "personal details"],
    "summary": ["summary", "professional summary", "profile", "about me"],
    "objective": ["objective", "career objective"],
    "education": ["education", "academic background"],
    "skills": ["skills", "technical skills", "core skills", "competencies"],
    "experience": ["experience", "work experience", "professional experience", "employment"],
    "projects": ["projects", "personal projects", "academic projects"],
    "certifications": ["certifications", "certificates", "licenses"],
    "achievements": ["achievements", "awards", "honors"],
    "internships": ["internships", "internship"],
    "publications": ["publications", "research"],
    "languages": ["languages", "language proficiency"],
}

SKILL_TERMS = [
    "python", "java", "javascript", "typescript", "c++", "sql", "excel",
    "power bi", "powerbi", "tableau", "pandas", "numpy", "tensorflow", "pytorch",
    "scikit-learn", "machine learning", "deep learning", "data analysis",
    "data visualization", "statistics", "etl", "aws", "azure", "gcp",
    "docker", "kubernetes", "linux", "git", "github", "jenkins", "terraform",
    "html", "css", "react", "node.js", "django", "flask", "selenium",
    "cybersecurity", "networking", "mongodb", "postgresql", "mysql", "agile",
]

ANN_8_SKILLS = [
    "Python",
    "SQL",
    "AWS",
    "Docker",
    "Linux",
    "Git",
    "PowerBI",
    "Excel"
]

CAREER_PROFILES = {
    "Data Analyst": ["python", "sql", "excel", "power bi", "tableau", "data analysis", "statistics"],
    "Data Scientist": ["python", "sql", "pandas", "numpy", "machine learning", "statistics", "tensorflow"],
    "Software Developer": ["python", "java", "javascript", "git", "sql", "testing"],
    "Python Developer": ["python", "django", "flask", "sql", "git", "api"],
    "Web Developer": ["html", "css", "javascript", "react", "node.js", "git"],
    "Cloud Engineer": ["aws", "azure", "gcp", "linux", "docker", "kubernetes", "terraform"],
    "DevOps Engineer": ["linux", "docker", "kubernetes", "jenkins", "terraform", "aws", "git"],
    "Machine Learning Engineer": ["python", "machine learning", "tensorflow", "pytorch", "docker", "git"],
    "Cybersecurity Analyst": ["cybersecurity", "linux", "networking", "python", "sql"],
    "Database Administrator": ["sql", "mysql", "postgresql", "mongodb", "linux"],
    "Business Analyst": ["sql", "excel", "power bi", "tableau", "data analysis", "agile"],
    "Power BI Developer": ["power bi", "sql", "excel", "data visualization", "etl"],
    "QA Engineer": ["selenium", "python", "javascript", "testing", "sql", "git"],
    "Network Engineer": ["networking", "linux", "aws", "azure", "cybersecurity"],
}


def clean_resume_text(text: str) -> str:
    text = text.replace("\u2022", "- ").replace("\u00a0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(line for line in lines if line)).strip()


def extract_resume_text(pdf_file) -> str:
    """Extracts text content from uploaded PDF stream."""
    reader = PdfReader(pdf_file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return clean_resume_text("\n".join(pages))


def detect_resume_sections(text: str) -> dict:
    lines = text.splitlines()
    section_names = {alias.lower(): key for key, aliases in SECTION_ALIASES.items() for alias in aliases}
    found = {key: "" for key in SECTION_ALIASES}
    current = "contact"
    chunks = {key: [] for key in SECTION_ALIASES}
    for line in lines:
        normalized = re.sub(r"[^a-z ]", "", line.lower()).strip()
        matched = section_names.get(normalized)
        if matched:
            current = matched
        else:
            chunks[current].append(line)
    for key, values in chunks.items():
        found[key] = "\n".join(values).strip()
    return found


def extract_skills(text: str) -> list:
    lowered = text.lower()
    found = set()
    for skill in SKILL_TERMS:
        pattern = r"(?<![a-z])" + re.escape(skill) + r"(?![a-z])"
        if re.search(pattern, lowered):
            found.add(skill.title() if skill != "power bi" else "Power BI")
    return sorted(found)


def extract_ann_skill_vector(resume_skills: list) -> list:
    """Maps extracted skills to the 8 binary indicators required by the ANN."""
    lowered_skills = {s.lower().replace(" ", "") for s in resume_skills}
    vector = []
    for skill in ANN_8_SKILLS:
        match = skill.lower().replace(" ", "") in lowered_skills
        vector.append(1 if match else 0)
    return vector


def calculate_job_match(resume_skills: list, job_skills: list):
    res_set = {s.lower() for s in resume_skills}
    job_set = {s.lower() for s in job_skills}
    matched = sorted({s.title() for s in res_set & job_set})
    missing = sorted({s.title() for s in job_set - res_set})
    score = round(len(matched) / len(job_set) * 100) if job_set else 0
    return score, matched, missing


def calculate_ats_score(resume_text: str, sections: dict):
    skills = extract_skills(resume_text)
    lines = resume_text.splitlines()
    bullets = [line for line in lines if line.startswith(("-", "*"))]
    has_contact = bool(re.search(r"@|\+?\d[\d ()-]{7,}|linkedin\.com|github\.com", resume_text, re.I))
    measurable = len(re.findall(r"\b\d+(?:%|\+|\s*(?:years?|users?|projects?|sales|revenue))\b", resume_text, re.I))
    section_count = sum(bool(value) for key, value in sections.items() if key != "contact")
    section_points = round(ATS_WEIGHTS["Sections"] * min(section_count / 6, 1))
    
    scores = {
        "Contact Information": ATS_WEIGHTS["Contact Information"] if has_contact else 0,
        "Sections": section_points,
        "Skills": round(ATS_WEIGHTS["Skills"] * min(len(skills) / 8, 1)),
        "Experience/Projects": round(ATS_WEIGHTS["Experience/Projects"] * min((bool(sections["experience"]) + bool(sections["projects"]) + min(len(bullets), 4) / 4) / 3, 1)),
        "Keywords": round(ATS_WEIGHTS["Keywords"] * min(len(skills) / 10, 1)),
        "Achievements": min(ATS_WEIGHTS["Achievements"], measurable * 2),
        "Formatting": min(ATS_WEIGHTS["Formatting"], 5 + (3 if bullets else 0) + (2 if len(lines) <= 120 else 0)),
        "Education": ATS_WEIGHTS["Education"] if sections["education"] or sections["certifications"] else 0,
        "Links": ATS_WEIGHTS["Links"] if re.search(r"linkedin\.com|github\.com|portfolio", resume_text, re.I) else 0,
    }
    return sum(scores.values()), scores


def generate_rule_based_careers(resume_skills: list) -> list:
    lowered_skills = {s.lower() for s in resume_skills}
    recommendations = []
    for career, required in CAREER_PROFILES.items():
        matched = [s for s in required if s in lowered_skills]
        if matched:
            score = round(len(matched) / len(required) * 100)
            recommendations.append({"career": career, "match": score, "why": [s.title() for s in matched]})
    return sorted(recommendations, key=lambda item: item["match"], reverse=True)[:5]


def build_resume_issues(text: str, sections: dict) -> list:
    issues = []
    if not re.search(r"@|\+?\d[\d ()-]{7,}", text, re.I):
        issues.append(("Contact information is incomplete", "Recruiters need a direct way to reach you.", "Add an email address and phone number."))
    if not sections.get("summary") and not sections.get("objective"):
        issues.append(("Professional summary is missing", "A concise summary quickly establishes your fit.", "Add a 2-3 sentence summary tailored to the target role."))
    if not sections.get("skills"):
        issues.append(("Skills section is missing", "ATS parsers and recruiters use a clear skills section to scan fit.", "Add a focused technical skills section."))
    if not sections.get("projects") and not sections.get("experience"):
        issues.append(("Experience or projects are missing", "Evidence of applied ability is important for career matching.", "Add projects or work experience with outcomes."))
    if text and not re.search(r"\b\d+(?:%|\+|\s*(?:years?|users?|projects?|sales|revenue))\b", text, re.I):
        issues.append(("No measurable achievements found", "Numbers make impact easier to assess.", "Add scale, time, performance, users, or other truthful outcomes."))
    if any(len(paragraph.split()) > 80 for paragraph in text.split("\n\n")):
        issues.append(("A paragraph is unusually long", "Dense blocks are harder to scan and parse.", "Break it into concise bullet points."))
    return issues


def generate_skill_gap(resume_skills: list, job_skills: list, recommendations: list) -> list:
    res_set = {s.lower() for s in resume_skills}
    if job_skills:
        job_set = {s.lower() for s in job_skills}
        return sorted({s.title() for s in job_set - res_set})
    target_skills = []
    for item in recommendations[:2]:
        for s in CAREER_PROFILES.get(item["career"], []):
            target_skills.append(s)
    return sorted({s.title() for s in target_skills if s not in res_set})[:8]
