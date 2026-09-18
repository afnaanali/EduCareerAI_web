"""
core/interview_generator.py — AI Interview Question Generation Engine
Generates tailored behavioral, technical, situational, and leadership interview questions
with STAR hints and gold-standard model answers based on user-selected roles,
seniority levels, and custom focus areas.
"""

import random
import uuid
from typing import List, Dict, Any, Optional

# ============================================================
# COMPREHENSIVE CURATED QUESTION BANK ACROSS 15+ ROLES
# ============================================================

CURATED_ROLE_QUESTIONS: Dict[str, List[Dict[str, Any]]] = {
    "Data Scientist / ML Engineer": [
        {
            "category": "behavioral",
            "difficulty": "Mid-Level",
            "question": "Describe a machine learning or data science project where your model initially performed poorly. How did you diagnose and fix it?",
            "hint": "Use STAR: Explain the business metric, diagnosis (e.g. data leakage, class imbalance, overfitting), the action taken, and quantifiable improvement.",
            "modelAnswer": "In my previous role, our customer churn prediction model suffered from low recall (48%) on high-value accounts. I investigated the feature distributions and identified heavy class imbalance and data leakage in timestamp features. I engineered rolling-window temporal features, applied SMOTE with focal loss, and tuned a Gradient Boosting classifier. This increased recall to 86% and helped our retention team save $320K in quarterly revenue.",
            "skills": ["Machine Learning", "Diagnostics", "Feature Engineering", "Gradient Boosting"]
        },
        {
            "category": "technical",
            "difficulty": "Mid-Level",
            "question": "How do you prevent overfitting in deep neural networks when working with limited training data?",
            "hint": "Mention data augmentation, dropout, weight decay (L2), transfer learning/pretrained weights, and early stopping.",
            "modelAnswer": "When training neural networks on scarce data, I apply a multi-layered regularization strategy: First, domain-specific data augmentation expands effective sample diversity. Second, I leverage transfer learning using pretrained foundation weights. Third, I incorporate spatial dropout (0.3–0.5) and AdamW with weight decay. Finally, I monitor validation loss with early stopping to prevent memorization.",
            "skills": ["Deep Learning", "Regularization", "Transfer Learning", "PyTorch"]
        },
        {
            "category": "leadership",
            "difficulty": "Senior / Lead",
            "question": "How do you explain complex neural network predictions to non-technical executive stakeholders?",
            "hint": "Discuss SHAP / LIME explainability, translation to business KPIs, and visual decision trees rather than raw math.",
            "modelAnswer": "I bridge the technical gap by translating model weights into business outcomes. Rather than discussing loss gradients, I use SHAP value waterfall plots to show which business drivers influenced decisions. I focus on actionable metrics—such as cost per false positive—and present interactive scenario simulators that allow leaders to test strategic hypotheses.",
            "skills": ["Explainable AI (XAI)", "SHAP", "Stakeholder Communication", "KPIs"]
        },
        {
            "category": "situational",
            "difficulty": "Senior / Lead",
            "question": "Your production recommendation system starts exhibiting concept drift after a seasonal product catalog update. How do you respond?",
            "hint": "Explain real-time data drift monitoring (Evidently/Whylogs), automated shadow deployment, and scheduled retraining pipelines.",
            "modelAnswer": "I detect concept drift by tracking Kolmogorov-Smirnov statistical divergence on input embeddings and monitoring degradation in live Click-Through Rates. Upon triggering an alert, our automated MLOps pipeline spins up a shadow model trained on the last 14 days of seasonal interactions. After A/B testing confirms a 12% lift in NDCG over the baseline, we execute a zero-downtime blue/green swap.",
            "skills": ["MLOps", "Model Monitoring", "Concept Drift", "A/B Testing"]
        },
        {
            "category": "technical",
            "difficulty": "Entry-Level",
            "question": "Explain the trade-off between Precision and Recall. In what business scenario would you optimize for Recall over Precision?",
            "hint": "Define both metrics mathematically, discuss the F1/PR curve trade-off, and give a high-stakes false-negative example (e.g., cancer detection or fraud screening).",
            "modelAnswer": "Precision measures the proportion of predicted positives that are truly positive, whereas Recall measures the proportion of actual positives successfully captured. In critical detection tasks—such as malignant tumor diagnosis or anti-money laundering triage—a False Negative is catastrophic. Therefore, we deliberately tune classification thresholds to achieve 98%+ Recall, accepting lower Precision that can be filtered by secondary human review.",
            "skills": ["Evaluation Metrics", "Precision vs Recall", "Threshold Tuning"]
        },
    ],

    "Software Engineer / Full Stack": [
        {
            "category": "behavioral",
            "difficulty": "Mid-Level",
            "question": "Tell me about a time you had to resolve a critical production bug under intense time pressure.",
            "hint": "Focus on triage, root-cause isolation, mitigation vs permanent fix, post-mortem, and automated regression guards.",
            "modelAnswer": "During a peak sales event, our payment microservice experienced thread pool exhaustion and dropped 18% of checkout requests. I immediately triaged APM traces, identified a deadlocked connection pool caused by unindexed DB queries, and deployed a connection limiter rollback within 12 minutes. I then indexed the query tables, implemented circuit breakers, and authored an incident post-mortem with automated load test gates.",
            "skills": ["Incident Management", "Root Cause Analysis", "Database Optimization", "Post-Mortem"]
        },
        {
            "category": "technical",
            "difficulty": "Senior / Lead",
            "question": "How do you design a scalable distributed system to handle 100,000 concurrent websocket connections with sub-50ms latency?",
            "hint": "Discuss stateless edge nodes, Redis Pub/Sub or Kafka message brokers, horizontal autoscaling, and connection pooling.",
            "modelAnswer": "To handle 100k concurrent WebSocket connections, I decouple connection state from business logic. Stateless WebSocket gateway pods run behind an ALB using epoll/event-loop runtimes. Message broadcasting uses Redis Pub/Sub or Kafka clusters. Client heartbeats prevent zombie sockets, and horizontal pod autoscalers scale gateway nodes based on active socket count.",
            "skills": ["System Design", "WebSockets", "Distributed Systems", "Redis / Kafka"]
        },
        {
            "category": "leadership",
            "difficulty": "Senior / Lead",
            "question": "How do you handle technical debt when product managers are pushing heavily for immediate feature delivery?",
            "hint": "Advocate for allocating a constant percentage (e.g., 20%) to tech debt, quantify debt cost in developer velocity / downtime risk, and bundle refactoring into related features.",
            "modelAnswer": "I quantify technical debt by showing its direct drag on sprint velocity and incident rates. I negotiated a 20% dedicated capacity allocation in each sprint for debt reduction and architectural hardening. Additionally, I enforce the 'Boy Scout Rule'—requiring developers to leave touched modules cleaner than they found them—and link refactoring milestones directly to upcoming feature roadmaps.",
            "skills": ["Technical Leadership", "Agile Estimation", "Stakeholder Negotiation", "Architecture"]
        },
        {
            "category": "technical",
            "difficulty": "Entry-Level",
            "question": "Explain the difference between SQL and NoSQL databases. When would you choose PostgreSQL over MongoDB?",
            "hint": "Cover ACID compliance, schema flexibility, relational normalization vs document embedding, and horizontal vs vertical scalability.",
            "modelAnswer": "PostgreSQL is a relational database adhering strictly to ACID guarantees, optimal for structured schemas requiring multi-table joins, transactions, and relational integrity like fintech or billing systems. MongoDB is a document-oriented NoSQL database optimized for rapid schema iteration, nested hierarchical payloads, and horizontal sharding. I select PostgreSQL when data consistency is non-negotiable.",
            "skills": ["Database Architecture", "PostgreSQL", "MongoDB", "ACID Transactions"]
        },
        {
            "category": "situational",
            "difficulty": "Mid-Level",
            "question": "A third-party API your service depends on suddenly experiences intermittent 504 Gateway Timeouts. How do you protect your application?",
            "hint": "Discuss timeout limits, circuit breakers (e.g. Resilience4j/Polly), exponential backoff retries with jitter, and caching fallbacks.",
            "modelAnswer": "I implement an immediate circuit breaker pattern with a 3-second timeout limit. If the upstream failure rate exceeds 20%, the breaker trips to open state, serving cached stale data or fallback defaults instead of stalling backend worker threads. For asynchronous operations, I queue tasks into a dead-letter queue with exponential backoff and randomized jitter.",
            "skills": ["Resilience Patterns", "Circuit Breakers", "Fault Tolerance", "API Design"]
        },
    ],

    "Cloud & DevOps Engineer": [
        {
            "category": "situational",
            "difficulty": "Mid-Level",
            "question": "Walk me through how you would architect a zero-downtime multi-region CI/CD deployment pipeline on AWS/Kubernetes.",
            "hint": "Explain GitOps (ArgoCD), canary/blue-green releases, automated rollback triggers, and cross-region traffic routing via Route 53.",
            "modelAnswer": "I architect multi-region CI/CD pipelines using GitOps with ArgoCD and Terraform. Code merges trigger automated container security scanning and unit testing. Deployments utilize Canary rollouts—routing 5% of live traffic to the new revision while monitoring Prometheus error rates and latency SLAs. If anomalies occur, automatic rollbacks fire in under 10 seconds without dropping user sessions.",
            "skills": ["GitOps", "ArgoCD", "Kubernetes", "AWS Route 53", "Canary Deployment"]
        },
        {
            "category": "technical",
            "difficulty": "Senior / Lead",
            "question": "How do you design an Infrastructure as Code (IaC) modular framework in Terraform to manage multi-account AWS landing zones securely?",
            "hint": "Discuss Terraform remote state locking (S3/DynamoDB), AWS Organizations, IAM role assumption, and automated drift detection.",
            "modelAnswer": "I structure Terraform into reusable root and child modules separated by environment (dev, staging, prod) and account boundaries. State is centralized in encrypted S3 buckets with DynamoDB locking. CI pipelines run 'terraform plan' on pull requests, executing via cross-account assumed IAM roles with least-privilege policies. Scheduled Atlantis/Terraform Cloud workflows continuously detect and report configuration drift.",
            "skills": ["Terraform", "AWS Organizations", "IaC Security", "Drift Detection"]
        },
        {
            "category": "behavioral",
            "difficulty": "Mid-Level",
            "question": "Describe an incident where a faulty deployment caused an outage. How did you restore service and what permanent safeguards did you implement?",
            "hint": "Detail fast rollback procedures, communication during the outage, blameless post-mortem, and automated gating.",
            "modelAnswer": "A faulty Kubernetes Helm upgrade introduced a configuration typo that caused CrashLoopBackOff on our ingress proxies, causing a 7-minute customer outage. I executed an immediate 'helm rollback' within 90 seconds to restore availability. In the post-mortem, I integrated strict Helm schema validation, pre-commit kubeconform linting, and automated staging smoke tests into our GitHub Actions workflow.",
            "skills": ["Kubernetes", "Helm", "Incident Response", "CI/CD Gating"]
        },
        {
            "category": "technical",
            "difficulty": "Entry-Level",
            "question": "What is the difference between Docker containers and Virtual Machines (VMs), and how does Linux cgroups/namespaces make containers lightweight?",
            "hint": "Highlight OS kernel sharing vs hypervisor virtualization, startup latency, and memory footprint.",
            "modelAnswer": "VMs run full guest operating systems on top of a hypervisor with dedicated virtualized hardware, consuming gigabytes of memory and requiring minutes to boot. Docker containers share the host Linux OS kernel, using Namespaces for process/network isolation and Cgroups for resource allocation (CPU/RAM caps). This enables sub-second boot times and minimal overhead.",
            "skills": ["Docker", "Linux Kernel", "Cgroups", "Namespaces", "Virtualization"]
        },
    ],

    "Data Analyst & BI Specialist": [
        {
            "category": "behavioral",
            "difficulty": "Mid-Level",
            "question": "Describe an instance where your analytical insights directly influenced a major business decision.",
            "hint": "State the business problem, your data analysis approach (SQL/BI dashboards), the counter-intuitive insight, and resulting ROI.",
            "modelAnswer": "Our marketing team was spending 40% of their ad budget on an underperforming channel. I conducted cohort retention analysis across 500,000 users using SQL and PowerBI, discovering that organic referral users had a 3x higher 90-day LTV. I presented these findings to the VP of Growth, leading to a reallocation of $150K into customer referral programs, which increased overall CAC efficiency by 32%.",
            "skills": ["SQL", "Cohort Analysis", "PowerBI", "Executive Storytelling", "LTV/CAC"]
        },
        {
            "category": "technical",
            "difficulty": "Mid-Level",
            "question": "How do you optimize an expensive SQL query involving multiple joins and aggregations across tables with 50+ million rows?",
            "hint": "Discuss EXPLAIN ANALYZE execution plans, composite indexing, avoiding SELECT *, partitioning, and CTE materialization.",
            "modelAnswer": "I start by inspecting the query execution plan with EXPLAIN ANALYZE to identify sequential table scans and nested loop bottlenecks. I add composite indexes on foreign key join predicates and filter columns, rewrite correlated subqueries into CTEs or window functions, partition high-volume transaction tables by date, and pre-aggregate reporting metrics into indexed materialized views.",
            "skills": ["SQL Optimization", "EXPLAIN ANALYZE", "Database Indexing", "Materialized Views"]
        },
        {
            "category": "situational",
            "difficulty": "Entry-Level",
            "question": "Two different department heads come to you with conflicting definitions of 'Monthly Active Users' (MAU). How do you resolve this?",
            "hint": "Explain data governance, facilitating alignment meetings, documenting metric dictionaries, and establishing a single source of truth.",
            "modelAnswer": "I organized a alignment session between both department leaders to understand their specific reporting use cases. We established a standardized data dictionary in our data catalog (dbt/Alation) defining core Platform MAU vs Paid Active Users. I updated central warehouse semantic models so all dashboards query the verified single source of truth.",
            "skills": ["Data Governance", "Metric Standardization", "Stakeholder Alignment", "dbt"]
        },
    ],

    "Cybersecurity & SOC Analyst": [
        {
            "category": "situational",
            "difficulty": "Mid-Level",
            "question": "A high-severity alert triggers indicating possible ransomware activity on an endpoint inside the corporate LAN. What are your immediate containment steps?",
            "hint": "Follow NIST incident response: network isolation, memory dump capture, process termination, hash extraction, and credential revocation.",
            "modelAnswer": "I follow the NIST Incident Response Framework: First, I immediately isolate the infected host from the corporate network via EDR agent to prevent lateral movement. Second, I take a live volatile memory dump for forensic analysis before terminating malicious processes. Third, I extract file hashes (SHA256) to block across our perimeter firewalls and revoke all Active Directory session tokens for compromised user accounts.",
            "skills": ["Incident Response", "EDR Containment", "NIST Framework", "Malware Forensics"]
        },
        {
            "category": "technical",
            "difficulty": "Senior / Lead",
            "question": "Explain how an attacker executes a Cross-Site Request Forgery (CSRF) vs Cross-Site Scripting (XSS) attack, and how you defend against both.",
            "hint": "Explain malicious script injection vs unauthorized state-changing command execution, SameSite cookies, CSP headers, and anti-CSRF tokens.",
            "modelAnswer": "XSS occurs when malicious JavaScript is injected into a trusted web app, executing in victim browsers to steal session storage or DOM data. Defense requires context-aware output encoding and strict Content Security Policy (CSP). CSRF tricks authenticated users into submitting unwanted commands; defense requires SameSite=Strict cookies, custom anti-CSRF token verification, and checking Origin/Referer headers.",
            "skills": ["AppSec", "OWASP Top 10", "XSS Remediation", "CSRF Defense", "CSP"]
        },
        {
            "category": "behavioral",
            "difficulty": "Mid-Level",
            "question": "Describe how you conducted a threat hunting exercise using SIEM log analysis that identified an unpatched vulnerability or unauthorized access.",
            "hint": "Explain hypothesis creation, log source correlation (Windows Event 4624/Sysmon/Zeek), MITRE ATT&CK mapping, and remediation.",
            "modelAnswer": "Hypothesizing that living-off-the-land binaries (LOLBins) were executing PowerShell obfuscation, I queried Splunk for anomalous parent-child process relationships (Word spawning PowerShell with EncodedCommand flags). I detected an unauthorized persistence script in an unpatched staging server, mapped it to MITRE T1059.001, evicted the attacker, and patched the vulnerable service.",
            "skills": ["Threat Hunting", "SIEM (Splunk)", "Sysmon", "MITRE ATT&CK", "PowerShell Security"]
        },
    ],

    "AI / Deep Learning Researcher": [
        {
            "category": "technical",
            "difficulty": "Senior / Lead",
            "question": "Explain the mathematical formulation of the Scaled Dot-Product Attention in Transformers and why the scaling factor sqrt(d_k) is necessary.",
            "hint": "Write Softmax((Q*K^T)/sqrt(d_k))*V and explain how large dimensions cause dot products to grow large, pushing softmax into tiny gradient regions.",
            "modelAnswer": "Attention is calculated as Attention(Q,K,V) = Softmax((Q K^T) / sqrt(d_k)) V. When the key dimension d_k is large, dot products grow substantially in magnitude. Without the scaling factor 1/sqrt(d_k), the softmax function enters regions with extremely small gradients (vanishing gradient problem), impairing backpropagation during training.",
            "skills": ["Transformer Architecture", "Self-Attention", "Gradient Dynamics", "PyTorch"]
        },
        {
            "category": "situational",
            "difficulty": "Senior / Lead",
            "question": "You are fine-tuning an 8B parameter LLM for domain-specific medical QA on limited GPU hardware. What PEFT and quantization strategy do you choose?",
            "hint": "Discuss QLoRA (4-bit NF4 quantization), rank selection (r=16, alpha=32), FlashAttention-2, and gradient checkpointing.",
            "modelAnswer": "I would implement QLoRA using 4-bit NormalFloat (NF4) base model quantization and Double Quantization to reduce GPU memory footprint to ~6 GB. I would attach low-rank adapter matrices (r=16, lora_alpha=32) to all linear projection layers, enable FlashAttention-2 for fast kernel execution, and use gradient checkpointing with paged AdamW to fit fine-tuning comfortably on a single 24GB RTX 4090/A10G GPU.",
            "skills": ["LLM Fine-Tuning", "QLoRA", "PEFT", "Quantization", "FlashAttention"]
        },
    ],

    "Backend Engineer (Python / Go / Java)": [
        {
            "category": "technical",
            "difficulty": "Mid-Level",
            "question": "How do Goroutines in Go differ from OS threads, and how does the Go runtime scheduler (GMP model) manage concurrent execution?",
            "hint": "Explain lightweight user-space scheduling, starting stack size (2KB vs 1-8MB), and M:N multiplexing across Goroutines (G), OS threads (M), and Processors (P).",
            "modelAnswer": "Goroutines are lightweight user-space green threads managed entirely by the Go runtime rather than the OS kernel. They start with a tiny 2KB dynamic stack (compared to 1-8MB for OS threads). The GMP scheduler multiplexes N goroutines (G) onto M OS threads across P logical processors, utilizing work-stealing and asynchronous system call handoffs to maintain near-zero context-switching overhead.",
            "skills": ["Go Concurrency", "Goroutines", "GMP Scheduler", "Low-Level Performance"]
        },
        {
            "category": "technical",
            "difficulty": "Senior / Lead",
            "question": "How do you implement distributed idempotency in a payment checkout API to prevent duplicate charges caused by network retries?",
            "hint": "Explain client-provided Idempotency-Key headers, Redis atomic SETNX with TTL, database unique constraints, and transaction isolation.",
            "modelAnswer": "I enforce an 'Idempotency-Key' header on all checkout endpoints. The API performs an atomic Redis SETNX with a 120-second lease to lock the key. If duplicate requests arrive during processing, they receive a 409 Conflict or await the cached final response. Once the database transaction completes, the finalized HTTP payload is stored in Redis under the key for instant deterministic replay.",
            "skills": ["API Design", "Distributed Idempotency", "Redis Locking", "Transaction Safety"]
        },
    ],

    "Frontend / UI Engineer (React / TypeScript)": [
        {
            "category": "technical",
            "difficulty": "Mid-Level",
            "question": "Explain how React 19 / 18 Concurrent Rendering and Fiber reconciliation work under the hood. How does React avoid blocking the main UI thread?",
            "hint": "Discuss fiber tree node structure, cooperative multitasking (requestIdleCallback / MessageChannel scheduler), work interruption, and useTransition.",
            "modelAnswer": "React Fiber represents the virtual DOM as a doubly linked tree of work units. Under Concurrent Mode, React breaks rendering into interruptible chunks executed via its internal Scheduler using MessageChannel. High-priority user events (keystrokes, clicks) can pause low-priority render transitions (filtering lists with useTransition), preventing frame drops and ensuring responsive 60 FPS interactions.",
            "skills": ["React Fiber", "Concurrent Rendering", "useTransition", "DOM Performance"]
        },
        {
            "category": "technical",
            "difficulty": "Mid-Level",
            "question": "How do you optimize Core Web Vitals (LCP, INP, CLS) for a high-traffic e-commerce web application?",
            "hint": "Discuss image optimization (WebP/AVIF, responsive srcset, aspect-ratio reserving), font display optional, code splitting with React.lazy, and memoization.",
            "modelAnswer": "To optimize LCP, I preload the hero banner image in modern AVIF format and self-host critical fonts with 'font-display: swap'. To eliminate CLS, all dynamic image and video elements have explicit width/height aspect ratios and skeleton loaders reserve layout dimensions. For INP, I break long CPU tasks into Web Workers and use React useDeferredValue for heavy search filtering.",
            "skills": ["Core Web Vitals", "LCP / INP / CLS", "Web Performance", "TypeScript"]
        },
    ],

    "Product / Technical Project Manager": [
        {
            "category": "leadership",
            "difficulty": "Senior / Lead",
            "question": "How do you prioritize a product roadmap when engineering, sales, and executive leadership have completely conflicting feature requests?",
            "hint": "Use frameworks like RICE (Reach, Impact, Confidence, Effort), customer discovery evidence, revenue projections, and roadmap transparency.",
            "modelAnswer": "I evaluate conflicting feature requests using the RICE scoring framework augmented with qualitative customer discovery interviews. I translate engineering debt into reliability impact and tie sales requests to verified pipeline ARR. By publishing an objective scoring matrix and aligning every feature to top-level company OKRs, stakeholders understand trade-offs transparently.",
            "skills": ["Product Strategy", "RICE Scoring", "Stakeholder Alignment", "Roadmapping"]
        },
        {
            "category": "situational",
            "difficulty": "Mid-Level",
            "question": "A key feature is 2 weeks behind schedule with only 3 days left before a publicly announced client launch. What action do you take?",
            "hint": "De-scope non-critical requirements to deliver an MVP, communicate early with transparent alternatives, and plan an immediate v1.1 fast-follow.",
            "modelAnswer": "I immediately convene a triage session with engineering leads to assess core user flows vs secondary bells and whistles. We aggressively de-scope non-critical UI extras while safeguarding core transaction security and stability. I proactively communicate an updated staged-rollout plan to executives, launching the verified core on time followed by a v1.1 update 7 days later.",
            "skills": ["Scope Management", "Agile Triage", "Crisis Communication", "MVP Delivery"]
        },
    ],
}

# Default Supported Roles list
SUPPORTED_ROLES = [
    "Data Scientist / ML Engineer",
    "Software Engineer / Full Stack",
    "Cloud & DevOps Engineer",
    "Data Analyst & BI Specialist",
    "Cybersecurity & SOC Analyst",
    "AI / Deep Learning Researcher",
    "Backend Engineer (Python / Go / Java)",
    "Frontend / UI Engineer (React / TypeScript)",
    "Product / Technical Project Manager",
]

# ============================================================
# DYNAMIC QUESTION SYNTHESIZER (SYNTHESIZES TAILORED QUESTIONS FOR ANY ROLE)
# ============================================================

def _generate_synthetic_question(
    role: str,
    category: str,
    difficulty: str,
    focus_topic: Optional[str] = None
) -> Dict[str, Any]:
    """Generates a high-quality, realistic STAR interview question for custom roles or focus topics."""
    
    clean_role = role.strip()
    clean_topic = focus_topic.strip() if focus_topic else clean_role

    if category == "behavioral":
        templates = [
            (
                f"Describe a complex {clean_topic} project where you faced unforeseen technical constraints or timeline bottlenecks. How did you navigate the challenge to deliver successful outcomes?",
                f"Use STAR: Detail the project context, the specific roadblock with {clean_topic}, your tactical intervention, and quantifiable deliverables (e.g. latency, revenue, efficiency).",
                f"In my previous project as a {clean_role}, we were tasked with implementing {clean_topic} within a tight 6-week deadline. We encountered severe performance bottlenecks during stress testing. I restructured the core workflow into parallelized stages and introduced automated caching. This reduced execution time by 44% and allowed our team to ship on schedule with zero customer-facing defects."
            ),
            (
                f"Tell me about a disagreement you had with an engineering peer or product stakeholder regarding the architectural approach for {clean_topic}. How did you reach consensus?",
                f"Use STAR: Highlight empathy, data-driven benchmarking or POC comparison, mutual compromise, and the final production result.",
                f"During an architecture review for {clean_topic}, a teammate proposed an approach that would have increased operational maintenance overhead. Instead of debating in the abstract, I built a quick prototype benchmark demonstrating a 3x throughput advantage and lower latency with the alternative. We aligned on the data-backed solution, which later supported a 200% traffic surge seamlessly."
            ),
        ]
    elif category == "technical":
        templates = [
            (
                f"What are the core design principles and industry best practices you follow when architecting high-reliability solutions in {clean_topic}?",
                f"Discuss modular decoupling, defensive error handling, scalability bottlenecks, observability/logging, and automated test coverage.",
                f"When building robust systems with {clean_topic}, I apply four core tenets: First, strict separation of concerns and interface abstraction. Second, comprehensive automated testing (unit, integration, and contract tests). Third, proactive observability using structured metrics, traces, and automated alerting. Finally, implementing graceful degradation and circuit breakers to guarantee high availability."
            ),
            (
                f"How do you optimize throughput, memory footprint, and latency when scaling {clean_topic} in production environments?",
                f"Explain profiling tools, asynchronous concurrency, connection reuse, caching layers, and database query/computation pruning.",
                f"To optimize {clean_topic}, I begin with APM profiling to locate the exact hot paths. I eliminate redundant I/O operations by introducing Redis/in-memory caching, utilize non-blocking asynchronous event loops for concurrent workloads, and tune connection pooling. In benchmark tests, this strategy consistently reduces p99 latency by over 50% while slashing resource utilization."
            ),
        ]
    elif category == "leadership":
        templates = [
            (
                f"As a {clean_role}, how do you mentor junior engineers and foster technical excellence and code quality across your organization in {clean_topic}?",
                f"Mention pair programming, comprehensive code review standards, internal tech talks/documentation, and establishing blameless post-mortem cultures.",
                f"I foster engineering excellence through structured mentorship and continuous knowledge sharing. I instituted standard code review checklists, lead bi-weekly architectural deep-dive sessions, and pair program with junior developers on complex tickets. This reduced PR revision cycles by 30% and accelerated new engineer onboarding time from 5 weeks to under 2 weeks."
            ),
        ]
    else:  # situational
        templates = [
            (
                f"You discover a severe degradation in production affecting {clean_topic} right during a major customer release. What is your immediate incident triage process?",
                f"Explain mitigation before root cause (rollback/kill switch), cross-team status communication, blameless post-mortem, and permanent automated guards.",
                f"My first priority is containment and customer protection: I trigger the pre-configured feature flag or initiate a quick rollback to restore the last known stable state within minutes. Once live traffic is safe, I investigate logs and distributed traces to pinpoint the defect, release a hotfix under full test coverage, and publish a blameless post-mortem detailing prevention mechanisms."
            ),
        ]

    selected = random.choice(templates)
    return {
        "id": f"ai-{uuid.uuid4().hex[:8]}",
        "role": clean_role,
        "category": category,
        "difficulty": difficulty,
        "question": selected[0],
        "hint": selected[1],
        "modelAnswer": selected[2],
        "skills": [clean_topic, "Problem Solving", "STAR Communication"],
        "isAiGenerated": True,
    }


# ============================================================
# MAIN GENERATION ENTRYPOINT
# ============================================================

def generate_interview_questions(
    role: str = "Data Scientist / ML Engineer",
    category: str = "all",
    difficulty: str = "all",
    count: int = 5,
    focus_topics: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generates tailored interview questions based on role, category, and difficulty.
    Blends curated expert questions with AI dynamic synthesis for guaranteed coverage.
    """
    count = max(1, min(count, 15))
    results: List[Dict[str, Any]] = []

    # 1. Match from Curated Bank if role matches closely
    matched_role_key = None
    for supported in CURATED_ROLE_QUESTIONS.keys():
        if supported.lower() == role.lower() or supported.lower().startswith(role.lower()[:8]):
            matched_role_key = supported
            break

    candidate_pool = []
    if matched_role_key and not focus_topics:
        for q in CURATED_ROLE_QUESTIONS[matched_role_key]:
            cat_match = (category == "all" or q["category"] == category)
            diff_match = (difficulty == "all" or q.get("difficulty") == difficulty)
            if cat_match and diff_match:
                candidate_pool.append({
                    "id": f"bank-{uuid.uuid4().hex[:6]}",
                    "role": role,
                    "category": q["category"],
                    "difficulty": q.get("difficulty", "Mid-Level"),
                    "question": q["question"],
                    "hint": q["hint"],
                    "modelAnswer": q["modelAnswer"],
                    "skills": q.get("skills", []),
                    "isAiGenerated": False,
                })

    # Shuffle candidate bank questions
    random.shuffle(candidate_pool)
    results.extend(candidate_pool[:count])

    # 2. If we need more questions or custom role/focus topics, generate dynamically
    categories_cycle = ["behavioral", "technical", "situational", "leadership"]
    if category != "all":
        categories_cycle = [category]

    difficulty_cycle = ["Mid-Level", "Senior / Lead", "Entry-Level"]
    if difficulty != "all":
        difficulty_cycle = [difficulty]

    while len(results) < count:
        curr_cat = categories_cycle[len(results) % len(categories_cycle)]
        curr_diff = difficulty_cycle[len(results) % len(difficulty_cycle)]
        synthetic = _generate_synthetic_question(
            role=role,
            category=curr_cat,
            difficulty=curr_diff,
            focus_topic=focus_topics
        )
        results.append(synthetic)

    return results[:count]


# ============================================================
# INTERVIEW ANSWER & TIPS GENERATOR
# ============================================================

def generate_answer_and_tips(
    question: str,
    role: str = "Data Scientist / ML Engineer",
    category: str = "behavioral",
    difficulty: str = "Mid-Level",
    hint: Optional[str] = None,
    model_answer: Optional[str] = None,
    user_response: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generates or formats a gold-standard recommended answer along with
    strategic tips, STAR breakdown, pitfalls to avoid, keywords to mention,
    and comparative feedback if a candidate response is provided.
    """
    clean_question = question.strip() if question else "Interview Question"
    clean_role = role.strip() if role else "Software & Data Professional"
    cat_lower = category.lower() if category else "behavioral"
    diff = difficulty if difficulty else "Mid-Level"

    # 1. Resolve Recommended Answer
    final_model_answer = model_answer.strip() if model_answer else ""
    final_hint = hint.strip() if hint else ""

    if not final_model_answer:
        # Search curated banks for matching question
        for r_key, q_list in CURATED_ROLE_QUESTIONS.items():
            for item in q_list:
                if item["question"].lower() in clean_question.lower() or clean_question.lower() in item["question"].lower():
                    final_model_answer = item.get("modelAnswer", "")
                    if not final_hint:
                        final_hint = item.get("hint", "")
                    break
            if final_model_answer:
                break

    # If still not found, synthesize a high-impact structured answer
    if not final_model_answer:
        if "technical" in cat_lower or "design" in clean_question.lower() or "how do you" in clean_question.lower():
            final_model_answer = (
                f"To address this in a production {clean_role} environment, I follow a 4-step architectural approach: "
                f"1) Requirements & Trade-offs: I clarify latency, scalability, and consistency constraints. "
                f"2) Core Implementation: I leverage industry-standard frameworks and decouple state from compute. "
                f"3) Resilience & Observability: I introduce circuit breakers, distributed tracing, and automated telemetry. "
                f"4) Verification: I validate performance under stress testing and establish automated regression guards."
            )
        else:
            final_model_answer = (
                f"In my previous project as a {clean_role}, our team encountered a critical challenge regarding this exact scenario. "
                f"Situation & Task: We faced tight delivery timelines with ambiguous requirements affecting our core workflow. "
                f"Action: I took ownership by conducting root-cause telemetry analysis, orchestrating stakeholder alignment, and engineering an automated pipeline fix. "
                f"Result: This mitigated the issue within 48 hours, boosted operational efficiency by 28%, and established automated regression tests for future deployments."
            )

    if not final_hint:
        final_hint = "Use STAR: Detail the Situation, Task, Action you specifically executed, and quantifiable business Result."

    # 2. Build Category-Tailored Strategic Tips
    key_tips = []
    if "technical" in cat_lower:
        key_tips = [
            "Structure your answer from High-Level Architecture down to Low-Level Implementation details.",
            "Always state the Trade-offs (e.g. Memory vs CPU, Latency vs Throughput, Consistency vs Availability).",
            "Mention concrete technologies, libraries, and observability tools relevant to " + clean_role + ".",
            "Conclude by mentioning automated testing, edge-case handling, and monitoring."
        ]
        framework = "Technical Architecture & Trade-off Analysis"
    elif "leadership" in cat_lower:
        key_tips = [
            "Highlight proactive communication and alignment across non-technical and executive stakeholders.",
            "Focus on active listening, empathy, and framing solutions around business KPIs.",
            "Demonstrate how you empower team members and resolve conflicting priorities constructively.",
            "State the lasting organizational process or cultural improvement that resulted."
        ]
        framework = "Leadership & Stakeholder Alignment Framework"
    elif "situational" in cat_lower:
        key_tips = [
            "Clarify immediate triage vs long-term preventative fixes.",
            "Show systematic root-cause isolation rather than guessing or reactive patching.",
            "Explain communication protocols during incidents (SLAs, status updates, post-mortems).",
            "Quantify the risk avoided and post-incident safeguards implemented."
        ]
        framework = "Incident Triage & Systematic Problem Solving"
    else: # behavioral / STAR
        key_tips = [
            "Use the STAR Method: 15% Situation, 15% Task, 50% Action, and 20% Result.",
            "Focus on 'I' (your individual contribution and technical decisions) rather than just 'we'.",
            "Include quantifiable metrics in your Result (e.g., '% improvement', '$ revenue saved', 'latency reduced by X ms').",
            "Highlight reflection: what you learned and how it influenced subsequent projects."
        ]
        framework = "STAR Method (Situation, Task, Action, Result)"

    # 3. Common Pitfalls to Avoid
    common_pitfalls = [
        "Speaking in vague generalities without anchoring your response in a concrete, real-world project.",
        "Over-indexing on team efforts ('we did...') without clarifying your personal architectural and coding contributions.",
        "Failing to quantify the business outcome or metric improvements in the conclusion."
    ]

    # 4. Extract Recommended Keywords
    keywords = []
    for word in ["STAR Method", "Root Cause Analysis", "Scalability", "Optimization", "Latency", "Throughput", "Data Pipeline", "A/B Testing", "CI/CD", "Post-Mortem", "Stakeholder Alignment", "KPIs", "Architecture"]:
        if word.lower() in final_model_answer.lower() or word.lower() in clean_question.lower():
            keywords.append(word)
    if not keywords:
        keywords = ["STAR Framework", "Metrics & KPIs", "Problem Solving", "Collaboration", "Technical Rigor"]

    # 5. Comparative Evaluation if User Answer is Provided
    comparative_feedback = []
    strengths = []
    improvement_areas = []

    if user_response and user_response.strip():
        u_lower = user_response.lower()
        words = u_lower.split()
        w_count = len(words)

        has_metrics = any(c in user_response for c in ["%", "$"]) or any(w.isdigit() for w in words)
        has_situation = any(w in u_lower for w in ["when", "during", "project", "scenario", "context", "faced", "company", "team"])
        has_action = any(w in u_lower for w in ["implemented", "built", "engineered", "designed", "created", "analyzed", "developed", "led", "optimized"])
        has_result = any(w in u_lower for w in ["result", "achieved", "improved", "reduced", "increased", "%", "saved", "delivered"])

        if w_count >= 50:
            strengths.append(f"Good response length ({w_count} words) with thorough explanatory context.")
        else:
            improvement_areas.append(f"Answer is relatively brief ({w_count} words). Aim for 80–150 words to demonstrate depth.")

        if has_metrics:
            strengths.append("Great job incorporating quantifiable metrics and measurable outcomes.")
        else:
            improvement_areas.append("Missing quantifiable metrics (e.g., % performance increase, latency reduction, or dollar value saved).")

        if has_action:
            strengths.append("Strong technical action verbs demonstrating direct ownership.")
        else:
            improvement_areas.append("Highlight more concrete technical actions you executed (e.g. 'engineered', 'optimized', 'diagnosed').")

        if has_result:
            strengths.append("Clear conclusion stating the outcome and resolution.")
        else:
            improvement_areas.append("Conclude with an explicit Result section summarizing business impact and team takeaway.")

        comparative_feedback = [
            f"Comparison with AI Gold Standard: The recommended answer provides a tight STAR structure with specific metrics and technical depth.",
            *strengths,
            *([f"Targeted Improvement: {tip}" for tip in improvement_areas] if improvement_areas else ["Excellent alignment with gold-standard interview criteria."])
        ]

    return {
        "success": True,
        "question": clean_question,
        "role": clean_role,
        "category": cat_lower,
        "difficulty": diff,
        "framework": framework,
        "hint": final_hint,
        "recommended_answer": final_model_answer,
        "key_tips": key_tips,
        "common_pitfalls": common_pitfalls,
        "essential_keywords": keywords,
        "comparative_feedback": comparative_feedback,
        "strengths": strengths,
        "improvement_areas": improvement_areas,
    }
