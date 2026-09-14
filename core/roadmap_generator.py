"""
Career Roadmap and Training Generator.
Provides structured, actionable milestone roadmaps for career paths.
"""

CAREER_ROADMAPS = {
    "Data Analyst": {
        "title": "Data Analyst Career Pathway",
        "description": "Transform raw numbers into actionable business insights using SQL, BI tools, and Python.",
        "duration": "4 - 6 Months",
        "phases": [
            {
                "phase": "Phase 1: Foundations",
                "topics": ["Advanced Excel (VLOOKUP, Pivot Tables, Power Query)", "Statistics & Descriptive Analytics", "Relational Database Basics"],
                "project": "Sales Performance Dashboard in Excel"
            },
            {
                "phase": "Phase 2: Database & SQL Mastery",
                "topics": ["SQL Joins, Aggregations, Window Functions", "CTEs and Subqueries", "Database Schema Design"],
                "project": "Customer Churn Analysis in PostgreSQL"
            },
            {
                "phase": "Phase 3: Business Intelligence & Dashboards",
                "topics": ["Power BI / Tableau Visualizations", "DAX Formulas & Data Modeling", "Storytelling with Data"],
                "project": "Executive Business KPIs Dashboard in Power BI"
            },
            {
                "phase": "Phase 4: Python & Automation",
                "topics": ["Pandas & NumPy for Data Wrangling", "Matplotlib & Seaborn Visualizations", "Automated ETL reporting scripts"],
                "project": "End-to-End E-Commerce Data Pipeline"
            }
        ],
        "certifications": ["Google Data Analytics Professional Certificate", "Microsoft Certified: Power BI Data Analyst (PL-300)"],
        "key_skills": ["SQL", "Power BI", "Excel", "Python", "Tableau", "Statistics"]
    },
    "Data Scientist": {
        "title": "Data Scientist Career Pathway",
        "description": "Harness mathematical modeling, statistical inference, and machine learning to predict outcomes.",
        "duration": "6 - 9 Months",
        "phases": [
            {
                "phase": "Phase 1: Python, Mathematics & Linear Algebra",
                "topics": ["Python for Data Science (NumPy, SciPy, Pandas)", "Multivariable Calculus & Linear Algebra", "Probability & Hypothesis Testing"],
                "project": "Exploratory Data Analysis on Housing Market Data"
            },
            {
                "phase": "Phase 2: Classical Machine Learning",
                "topics": ["Scikit-Learn Algorithms (Regression, Trees, SVM, Ensembles)", "Cross-Validation & Hyperparameter Tuning", "Feature Engineering"],
                "project": "Credit Default Risk Predictive Model"
            },
            {
                "phase": "Phase 3: Deep Learning & NLP/CV",
                "topics": ["Neural Networks in TensorFlow/PyTorch", "Text Processing & Embeddings", "CNNs & Image Processing"],
                "project": "Customer Sentiment Classifier with LSTM/Transformers"
            },
            {
                "phase": "Phase 4: Model Deployment & MLOps",
                "topics": ["FastAPI Model Serving", "Docker Containerization", "Streamlit Dashboard Deployment", "Model Monitoring"],
                "project": "Production ML Inference Web App"
            }
        ],
        "certifications": ["IBM Data Science Professional Certificate", "AWS Certified Machine Learning - Specialty"],
        "key_skills": ["Python", "Machine Learning", "TensorFlow", "Scikit-Learn", "SQL", "Statistics"]
    },
    "Software Developer": {
        "title": "Software Developer Career Pathway",
        "description": "Design, build, and maintain scalable software applications and robust backends.",
        "duration": "6 Months",
        "phases": [
            {
                "phase": "Phase 1: Core Programming & Algorithms",
                "topics": ["Python / Java / C++ Fundamentals", "Object-Oriented Programming (OOP)", "Data Structures (Trees, Graphs, HashMaps) & Big-O"],
                "project": "CLI Task Management & File Indexing System"
            },
            {
                "phase": "Phase 2: Database Systems & Backend APIs",
                "topics": ["RESTful API Architecture", "FastAPI / Django / Spring Boot", "SQL & NoSQL Database Integration", "Authentication (JWT, OAuth)"],
                "project": "Secure Multi-tenant REST API with Auth"
            },
            {
                "phase": "Phase 3: Frontend Basics & Integration",
                "topics": ["HTML5, Modern CSS & JavaScript (ES6+)", "React or Frontend Frameworks", "API Consumption & State Management"],
                "project": "Full-Stack Web Portal"
            },
            {
                "phase": "Phase 4: Testing, CI/CD & DevOps Basics",
                "topics": ["Unit Testing with PyTest/JUnit", "Git branching & GitHub Actions", "Dockerizing Applications"],
                "project": "Automated CI/CD Pipeline for Web Application"
            }
        ],
        "certifications": ["Meta Back-End Developer Certificate", "Oracle Certified Associate Java Programmer"],
        "key_skills": ["Python", "Java", "Git", "SQL", "Docker", "REST APIs", "Data Structures"]
    },
    "Cloud Engineer": {
        "title": "Cloud Engineer Career Pathway",
        "description": "Architect, provision, and operate resilient cloud infrastructure and microservices.",
        "duration": "5 - 7 Months",
        "phases": [
            {
                "phase": "Phase 1: Linux & Networking Fundamentals",
                "topics": ["Linux Shell Scripting & Administration", "TCP/IP, DNS, Subnets, VPC, VPN", "SSH & System Security"],
                "project": "Automated Linux Server Provisioning Script"
            },
            {
                "phase": "Phase 2: Public Cloud Mastery (AWS / Azure / GCP)",
                "topics": ["Compute (EC2, Lambda)", "Storage & Databases (S3, RDS, DynamoDB)", "Identity & Access Management (IAM)", "VPC Networking"],
                "project": "High-Availability Multi-Tier Cloud Architecture"
            },
            {
                "phase": "Phase 3: Infrastructure as Code (IaC) & Containers",
                "topics": ["Docker Containerization", "Terraform & CloudFormation", "Kubernetes Cluster Management (EKS/AKS)"],
                "project": "Terraform-Automated Cloud Infrastructure Setup"
            },
            {
                "phase": "Phase 4: Observability & Security",
                "topics": ["CloudWatch, Prometheus & Grafana", "Cloud Cost Optimization", "Zero-Trust Security Policies"],
                "project": "Centralized Cloud Monitoring & Alerting System"
            }
        ],
        "certifications": ["AWS Solutions Architect - Associate", "Microsoft Certified: Azure Fundamentals (AZ-900)"],
        "key_skills": ["AWS", "Azure", "Linux", "Docker", "Terraform", "Kubernetes", "Networking"]
    },
    "DevOps Engineer": {
        "title": "DevOps Engineer Career Pathway",
        "description": "Bridge development and IT operations through continuous integration, automation, and infrastructure.",
        "duration": "6 Months",
        "phases": [
            {
                "phase": "Phase 1: Linux, Git & Scripting",
                "topics": ["Advanced Linux Systems Administration", "Bash & Python Automation Scripting", "Git Workflows & Trunk-Based Development"],
                "project": "Automated Backup & Log Rotation Script"
            },
            {
                "phase": "Phase 2: Containerization & Orchestration",
                "topics": ["Docker Image Optimization & Multi-stage Builds", "Kubernetes Architecture (Pods, Services, Ingress)", "Helm Package Manager"],
                "project": "Containerized Microservices on Kubernetes"
            },
            {
                "phase": "Phase 3: CI/CD Pipelines",
                "topics": ["GitHub Actions & GitLab CI", "Jenkins Pipeline-as-Code", "Automated Testing & Security Scans (SAST/DAST)"],
                "project": "Zero-Downtime Blue/Green Deployment Pipeline"
            },
            {
                "phase": "Phase 4: IaC & Observability",
                "topics": ["Terraform Modular Configs", "Ansible Configuration Management", "ELK / Prometheus / Grafana Monitoring"],
                "project": "Self-Healing Cloud Infrastructure Pipeline"
            }
        ],
        "certifications": ["Certified Kubernetes Administrator (CKA)", "AWS Certified DevOps Engineer"],
        "key_skills": ["Linux", "Docker", "Kubernetes", "Git", "Jenkins", "Terraform", "AWS"]
    }
}

DEFAULT_ROADMAP = {
    "title": "Professional Career Development Pathway",
    "description": "Systematic progression plan covering technical foundations, applied projects, and industry readiness.",
    "duration": "4 - 6 Months",
    "phases": [
        {
            "phase": "Phase 1: Foundational Theory & Tools",
            "topics": ["Core Domain Knowledge", "Key Industry Software & Tools", "Fundamental Methodologies"],
            "project": "Initial Hands-On Baseline Project"
        },
        {
            "phase": "Phase 2: Intermediate Practical Applications",
            "topics": ["Applied Workflows", "Problem-Solving Frameworks", "Standard Operating Procedures"],
            "project": "Intermediate Case Study Implementation"
        },
        {
            "phase": "Phase 3: Advanced Portfolio Building",
            "topics": ["Industry Best Practices", "Optimization & Collaboration", "Real-World Scenario Handling"],
            "project": "Comprehensive Capstone Portfolio Project"
        },
        {
            "phase": "Phase 4: Career Launch & Interview Readiness",
            "topics": ["ATS Resume Tailoring", "Portfolio Presentation", "Technical & Behavioral Interview Prep"],
            "project": "Published Professional Portfolio & Case Studies"
        }
    ],
    "certifications": ["Industry Standard Professional Accreditation"],
    "key_skills": ["Domain Fundamentals", "Problem Solving", "Communication", "Practical Execution"]
}


def get_roadmap_for_career(career_name: str) -> dict:
    """Returns a tailored roadmap or structured default roadmap."""
    for key, data in CAREER_ROADMAPS.items():
        if key.lower() in career_name.lower() or career_name.lower() in key.lower():
            return data
    custom_roadmap = dict(DEFAULT_ROADMAP)
    custom_roadmap["title"] = f"{career_name} Career Pathway"
    return custom_roadmap
