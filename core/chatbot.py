"""
core/chatbot.py — Intelligent Educational & Career AI Assistant
Supports context-aware follow-up queries, ordinal references ("explain the 4th one"),
deep conceptual deep-dives, learning roadmaps, interview prep, and code examples.
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple

_LLM_INSTANCE = None


def get_llm_model():
    """Lazily loads and caches local GGUF model if llama_cpp is available."""
    global _LLM_INSTANCE
    if _LLM_INSTANCE is not None:
        return _LLM_INSTANCE
    try:
        from llama_cpp import Llama
        _LLM_INSTANCE = Llama.from_pretrained(
            repo_id="unsloth/SmolLM2-135M-Instruct-GGUF",
            filename="SmolLM2-135M-Instruct-Q4_K_M.gguf",
            n_ctx=2048,
            verbose=False,
        )
        return _LLM_INSTANCE
    except Exception:
        return None


# ============================================================
# DEEP DIVE EXPLAINERS FOR SUBTOPICS & NUMBERED POINTS
# ============================================================

SUBTOPIC_EXPLAINERS = {
    # ------------------ MACHINE LEARNING ------------------
    "reinforcement learning": r"""### 🎮 Deep Dive: Reinforcement Learning (RL)

**Reinforcement Learning (RL)** is an advanced branch of Machine Learning where an autonomous software **Agent** learns to make a sequence of optimal decisions in an **Environment** to maximize a cumulative numerical **Reward** through continuous trial and error.

Unlike Supervised Learning (which requires explicit labels) or Unsupervised Learning (which finds passive patterns), RL is **active and goal-directed**.

---

#### 🔑 The 5 Core Components of Reinforcement Learning:
1. **Agent:** The decision maker or learner (e.g., self-driving car software, chess bot, robotic arm).
2. **Environment:** The world with which the agent interacts (e.g., a road simulation, chessboard, stock market).
3. **State (S):** The current situation/snapshot of the agent within the environment.
4. **Action (A):** The set of possible moves or decisions the agent can take.
5. **Reward (R):** Numerical feedback signal returned by the environment (Positive reward for good actions, Negative penalty for errors).
6. **Policy ($\pi$):** The agent's internal strategy or brain that maps given States to the best Actions: $\pi(s) \rightarrow a$.


---

#### ⚙️ How Reinforcement Learning Works (The Agent-Environment Loop):
```
        ┌─────────────────────────────────────────┐
        │               ENVIRONMENT               │
        └───────┬─────────────────────────▲───────┘
                │                         │
      Current State (S)             Action (A)
      & Reward (R)                        │
                │                         │
        ┌───────▼─────────────────────────┴───────┐
        │                  AGENT                  │
        │       (Learns Policy via Rewards)       │
        └─────────────────────────────────────────┘
```

1. The agent observes the current **State ($S_t$)**.
2. Based on its **Policy**, it selects an **Action ($A_t$)**.
3. The environment changes to a new **State ($S_{t+1}$)** and returns a **Reward ($R_{t+1}$)**.
4. The agent updates its mathematical value function (e.g., Q-Value table or Deep Neural Network weights) using the Bellman Equation to favor actions that yield higher long-term cumulative return.

---

#### 🧠 Key Algorithms & Architectures:
* **Model-Free Q-Learning:** Tabular algorithm calculating expected future rewards for state-action pairs.
* **Deep Q-Networks (DQN):** Combines Convolutional/Dense Neural Networks with Q-Learning (used by DeepMind for Atari games).
* **Policy Gradient / PPO (Proximal Policy Optimization):** Directly optimizes policy probabilities (standard algorithm used in **RLHF** to train ChatGPT & Claude).
* **Actor-Critic (A2C/A3C):** Uses two neural networks — an *Actor* that takes actions and a *Critic* that evaluates how good the action was.

---

#### 🚀 Major Real-World Applications:
* **Large Language Models (RLHF):** Reinforcement Learning from Human Feedback aligns AI responses with human preferences.
* **Autonomous Driving:** Lane centering, adaptive cruise control, and pedestrian avoidance pathing (Tesla, Waymo).
* **Game AI:** Superhuman game playing in Chess (AlphaZero), Go (AlphaGo), Dota 2 (OpenAI Five).
* **Robotics & Industrial Automation:** Robotic arms learning warehouse object sorting and robotic walking control.
* **Algorithmic Quantitative Finance:** Automated dynamic portfolio rebalancing and trade execution.
""",

    "supervised learning": """### 🏷️ Deep Dive: Supervised Learning

**Supervised Learning** is the most widely adopted paradigm of Machine Learning. It trains mathematical algorithms on **labeled datasets**, meaning every training input $X$ is paired with the known correct ground truth output $y$.

---

#### 🧩 The Two Primary Categories:
1. **Regression (Predicting Continuous Numerical Values):**
   * *Objective:* Fit a curve or hyperplane that minimizes prediction error (Loss function: Mean Squared Error / MSE, MAE).
   * *Examples:* Predicting real estate market valuations, stock prices, temperature forecasting.
   * *Key Algorithms:* Linear Regression, Ridge/Lasso, Support Vector Regressor (SVR), Random Forest Regressor, XGBoost.

2. **Classification (Predicting Discrete Categories / Labels):**
   * *Objective:* Draw decision boundaries separating distinct target classes (Loss function: Binary/Categorical Cross-Entropy).
   * *Examples:* Email spam detection (Spam / Not Spam), Medical disease diagnosis (Positive / Negative), Credit card fraud detection.
   * *Key Algorithms:* Logistic Regression, Decision Trees, Random Forests, Support Vector Machines (SVM), Naive Bayes, LightGBM.

---

#### 🔄 Supervised Learning Workflow:
1. **Data Splitting:** Divide labeled dataset into **Training Set (70-80%)** and **Testing/Validation Set (20-30%)**.
2. **Feature Scaling:** Normalize/Standardize numerical inputs (`StandardScaler`, `MinMaxScaler`).
3. **Model Training:** Optimization algorithms (e.g., Gradient Descent) adjust weights to minimize prediction error.
4. **Evaluation:** Measure performance using Precision, Recall, F1-Score, ROC-AUC (for classification) or $R^2$ / RMSE (for regression).
""",

    "unsupervised learning": """### 🔍 Deep Dive: Unsupervised Learning

**Unsupervised Learning** trains algorithms on **unlabeled datasets** with no predefined target output $y$. The algorithm independently discovers hidden geometric distributions, patterns, and natural groupings within the data $X$.

---

#### 🧩 Key Types & Subfields:
1. **Clustering:**
   * Grouping similar observations together based on distance metrics (Euclidean, Manhattan, Cosine similarity).
   * *Algorithms:* **K-Means** (centroid-based), **Hierarchical Clustering** (dendrograms), **DBSCAN** (density-based spatial clustering).
   * *Use Cases:* Customer market segmentation, document clustering, genetic DNA clustering.

2. **Dimensionality Reduction (Feature Compression):**
   * Reducing the number of input features while preserving maximum statistical variance.
   * *Algorithms:* **PCA (Principal Component Analysis)**, **t-SNE**, **UMAP**, Autoencoders.
   * *Use Cases:* High-dimensional data visualization, data compression, mitigating the "Curse of Dimensionality".

3. **Anomaly & Outlier Detection:**
   * Identifying rare data points that deviate significantly from standard normal patterns.
   * *Algorithms:* Isolation Forests, One-Class SVM, Local Outlier Factor (LOF).
   * *Use Cases:* Credit card fraud detection, industrial equipment sensor failure prediction.
""",

    "semi-supervised learning": """### ⚖️ Deep Dive: Semi-Supervised Learning

**Semi-Supervised Learning** bridges the gap between Supervised and Unsupervised learning by combining a **small amount of labeled data** with a **large pool of unlabeled data**.

---

#### 💡 Why It Is Crucial:
In many modern industries, raw data is abundant and cheap (e.g., millions of medical CT scans, voice recordings), but acquiring expert human annotations (e.g., certified radiologists, linguists) is extremely expensive and time-consuming.

#### ⚙️ Common Techniques:
1. **Pseudo-Labeling / Self-Training:** A model is initially trained on the small labeled set, then predicts labels for the unlabeled data with high confidence, and retrains on the combined set.
2. **Consistency Regularization:** Adding noise/augmentations to unlabeled data and forcing the model to produce consistent predictions.
3. **Graph-Based Methods:** Constructing similarity graphs connecting labeled and unlabeled nodes to propagate label information.
""",

    # ------------------ DEVOPS ------------------
    "ci/cd": """### 🔄 Deep Dive: CI/CD (Continuous Integration & Continuous Deployment)

**CI/CD** forms the automated backbone of modern software engineering and DevOps, replacing manual releases with automated pipelines.

---

#### 1. 🏗️ Continuous Integration (CI):
* Developers frequently push small, incremental code commits to a shared Git branch (main/trunk).
* Automated CI runners immediately execute:
  1. Syntax linting & static code analysis (`flake8`, `ESLint`, `SonarQube`).
  2. Unit and integration test suites (`pytest`, `Jest`).
  3. Security vulnerability scans (`Snyk`, `Trivy`).
* *Goal:* Catch bugs and merge conflicts within seconds of writing code.

#### 2. 🚀 Continuous Delivery vs. Continuous Deployment (CD):
* **Continuous Delivery:** Code passes all automated tests and is packaged into production-ready release artifacts (Docker images); deployment to production requires a one-click manual approval.
* **Continuous Deployment:** Every commit that passes the automated pipeline is automatically deployed straight to live production servers with zero human intervention.

*Tools:* **GitHub Actions**, **GitLab CI**, **Jenkins**, **ArgoCD**, **CircleCI**.
""",

    "infrastructure as code": """### 📜 Deep Dive: Infrastructure as Code (IaC)

**Infrastructure as Code (IaC)** is the practice of provisioning and managing cloud infrastructure (Virtual Machines, Kubernetes clusters, VPC networks, RDS databases, S3 buckets) using **declarative machine-readable code configuration files** rather than clicking through manual cloud console dashboards.

---

#### 🔑 Core Principles & Benefits:
1. **Declarative Configuration:** You define the *desired end state* (e.g., "I need 3 EC2 instances in a private subnet"), and the IaC engine automatically figures out how to provision them.
2. **Version Controlled Infrastructure:** Your cloud architecture is committed to Git, enabling code reviews, history tracking, and instant rollback.
3. **Idempotence & Consistency:** Applying the code 1 time or 100 times produces identical environments without configuration drift between Dev, Staging, and Production.

*Industry Standard Tools:* **Terraform (HashiCorp)**, **AWS CloudFormation**, **Ansible**, **Pulumi**.
""",

    "containerization": """### 🐳 Deep Dive: Containerization & Kubernetes Orchestration

**Containerization** packages application code, runtime, system libraries, and dependencies into lightweight, standalone executable units called **Containers**.

---

#### 💡 Containers vs. Virtual Machines (VMs):
* **Virtual Machines:** Include a full Guest OS on top of a hypervisor, consuming gigabytes of RAM and taking minutes to boot.
* **Containers (Docker):** Share the host OS kernel and run as isolated processes, consuming megabytes and booting in milliseconds.

#### ☸️ Kubernetes (K8s) — Container Orchestration:
When managing hundreds of microservices across multiple cloud servers, Kubernetes automates:
* **Auto-Scaling (HPA):** Dynamically spinning up more container Pods when web traffic spikes.
* **Self-Healing:** Automatically restarting crashed containers and rescheduling on healthy nodes.
* **Zero-Downtime Rolling Updates:** Gradually replacing old container versions with new ones.
* **Load Balancing & Service Discovery:** Routing user requests efficiently across healthy pods.
""",

    "devsecops": """### 🛡️ Deep Dive: DevSecOps (Security-First DevOps)

**DevSecOps** is the integration of cybersecurity practices into every stage of the DevOps software delivery lifecycle ("Shifting Security Left").

---

#### 🔑 Core Pillars:
1. **SAST (Static Application Security Testing):** Scanning source code for vulnerabilities (e.g., SQL injections, buffer overflows) during CI builds.
2. **DAST (Dynamic Application Security Testing):** Testing running applications for exploitable endpoints.
3. **SCA (Software Composition Analysis):** Scanning third-party dependencies (`npm`, `pip`) for known CVE vulnerabilities.
4. **Secret Detection:** Preventing developers from accidentally committing API keys or database passwords to Git repositories (GitGuardian, TruffleHog).
5. **IAM & Zero Trust:** Enforcing least-privilege permissions and secret rotation (HashiCorp Vault).
""",

    "mlops": """### 🤖 Deep Dive: MLOps (Machine Learning Operations)

**MLOps** is the discipline of applying DevOps principles (automation, testing, CI/CD, monitoring) to Machine Learning systems in production.

---

#### 🔑 The 3 Pillars of MLOps:
1. **Data Versioning & Management:** Tracking dataset versions and schemas as data evolves (DVC, Feast Feature Store).
2. **Model Training & Registry Pipelines:** Tracking training experiments, metrics, hyperparameter runs, and model artifacts (MLflow, Weights & Biases).
3. **Production Serving & Drift Monitoring:**
   * **Data Drift:** Statistical change in incoming user input distributions over time.
   * **Concept Drift:** Change in the real-world statistical relationship between features and target labels (requiring automated retraining triggers).
""",

    "sre": r"""### 🌐 Deep Dive: Site Reliability Engineering (SRE)

**Site Reliability Engineering (SRE)** is an engineering discipline pioneered by Google that applies software engineering approaches to IT operations and system reliability.

---

#### 🔑 Core SRE Concepts:
1. **SLI (Service Level Indicator):** A quantifiable metric of performance (e.g., Request Latency < 200ms, Error Rate < 0.01%).
2. **SLO (Service Level Objective):** The target goal agreed upon by the engineering team (e.g., 99.9% uptime over 30 days).
3. **SLA (Service Level Agreement):** The contractual commitment to external customers with financial penalties if breached.
4. **Error Budget:** The acceptable margin of unreliability ($100\% - SLO$). If the error budget is healthy, teams can deploy new experimental features quickly; if exhausted, deployments pause to focus on stability.
"""
}



# ============================================================
# KNOWLEDGE BASE DICTIONARY
# ============================================================

KNOWLEDGE_BASE = {
    "machine learning": {
        "key": "machine learning",
        "title": "Machine Learning (ML)",
        "definition": (
            "**Machine Learning (ML)** is a subfield of Artificial Intelligence (AI) focused on building algorithms "
            "and mathematical models that allow computers to learn patterns directly from data and make predictions "
            "or decisions without being explicitly programmed."
        ),
        "types_detail": """### 🧩 The 4 Primary Types of Machine Learning

Machine Learning is broadly categorized into 4 core paradigms based on how the learning algorithm interacts with training data:

---

#### 1. 🏷️ **Supervised Learning (Learning with Labels)**
* **How It Works:** The algorithm is trained on labeled input-output pairs $(X, y)$ to learn a mapping function $f(X) \\rightarrow y$.
* **Sub-Types:**
  - **Regression (Continuous Outputs):** Predicting numerical values (e.g., house prices, stock trends, test scores).
    * *Algorithms:* Linear Regression, Ridge/Lasso, Support Vector Regression (SVR), Random Forest Regressor, XGBoost.
  - **Classification (Discrete Outputs):** Categorizing data into classes (e.g., Spam vs. Not Spam, Disease Diagnosis).
    * *Algorithms:* Logistic Regression, Decision Trees, Random Forests, Naive Bayes, Support Vector Machines (SVM), HistGradientBoosting.
* **Industry Examples:** Loan default prediction, medical image pathology detection, customer churn forecasting.

---

#### 2. 🔍 **Unsupervised Learning (Learning without Labels)**
* **How It Works:** The algorithm analyzes unlabeled data $(X)$ to uncover hidden structures, groupings, or geometric distributions.
* **Sub-Types:**
  - **Clustering:** Partitioning similar items into clusters (e.g., K-Means, Hierarchical Clustering, DBSCAN).
  - **Dimensionality Reduction:** Compressing high-dimensional features while preserving variance (e.g., PCA, t-SNE, UMAP).
  - **Anomaly Detection:** Identifying rare outliers (e.g., Isolation Forests, One-Class SVM).
  - **Association Rule Learning:** Discovering item relationships (e.g., Apriori, Market Basket Analysis).
* **Industry Examples:** Customer segmentation, genomic clustering, fraud detection, recommendation engines.

---

#### 3. ⚖️ **Semi-Supervised Learning (Hybrid Learning)**
* **How It Works:** Combines a small amount of labeled data with a large volume of unlabeled data to improve accuracy without expensive manual annotation.
* **Sub-Types:** Pseudo-Labeling, Self-Training, Graph-Based Semi-Supervised Learning.
* **Industry Examples:** Medical image datasets where only a fraction are annotated by certified radiologists, speech recognition.

---

#### 4. 🎮 **Reinforcement Learning (RL — Learning via Rewards)**
* **How It Works:** An autonomous agent interacts with an environment, taking actions to maximize cumulative numerical rewards and minimize penalties through trial-and-error.
* **Key Components:** Agent, Environment, State ($S$), Action ($A$), Reward ($R$), Policy ($\\pi$).
* **Sub-Types:** Model-Free (Q-Learning, Deep Q-Networks/DQN, PPO) vs. Model-Based RL.
* **Industry Examples:** Self-driving vehicle path planning, game playing (AlphaGo, OpenAI Five), robotics control, algorithmic trading.
""",
        "ordered_subtopics": [
            "supervised learning",
            "unsupervised learning",
            "semi-supervised learning",
            "reinforcement learning"
        ],
        "workflow": """### 🔄 The End-to-End Machine Learning Pipeline

1. 📥 **Problem Definition & Data Collection:** Formulating business questions and gathering historical structured/unstructured data.
2. 🧹 **Data Preprocessing & EDA:** Imputing missing values, removing outliers, encoding categoricals (One-Hot, Ordinal), and standardizing scales.
3. ⚙️ **Feature Engineering:** Creating domain-specific features, polynomial interactions, and dimensionality reduction.
4. 🧠 **Model Selection & Training:** Training baseline algorithms vs. ensemble models (Random Forests, Gradient Boosting) or neural nets.
5. 📊 **Evaluation & Validation:** Stratified K-Fold cross-validation, measuring Precision, Recall, F1-Score, ROC-AUC, and RMSE.
6. 🚀 **Hyperparameter Tuning:** Grid search, RandomizedSearchCV, or Bayesian Optimization (Optuna).
7. 🌐 **Deployment & Monitoring (MLOps):** Containerizing models with Docker/FastAPI, serving predictions, and monitoring for feature/data drift.
""",
        "advantages": """### ⚖️ Advantages & Disadvantages of Machine Learning

#### ✅ Key Advantages:
- **Automation of Complex Decisions:** Handles millions of variables impossible for humans to code with static `if-else` logic.
- **Continuous Improvement:** Models can retrain dynamically on newly collected data to adapt to changing trends.
- **Pattern Recognition in Big Data:** Discovers non-linear relationships across hundreds of features.

#### ⚠️ Challenges & Limitations:
- **Data Quality Dependency ("Garbage In, Garbage Out"):** Requires clean, representative, and unbiased training datasets.
- **Interpretability / "Black Box" Problem:** Deep complex ensembles can be difficult to explain to regulators without SHAP/LIME.
- **Compute & Maintenance Overhead:** Retraining, GPU hardware, and monitoring data drift require ongoing engineering.
""",
        "pillars": [
            "**Supervised Learning:** Regression & Classification on labeled data.",
            "**Unsupervised Learning:** Clustering, PCA, and anomaly detection on unlabeled data.",
            "**Reinforcement Learning:** Goal-directed learning through reward maximization.",
            "**Deep Learning:** Multi-layer neural networks (CNN, RNN, Transformers)."
        ],
        "tools": ["Python", "Scikit-Learn", "PyTorch / TensorFlow", "Pandas & NumPy", "MLflow", "Jupyter"],
        "examples": "Spam filtering, recommendation systems (Netflix/Spotify), credit scoring, medical imaging diagnostics, and self-driving perception.",
        "pathway": "1. Master Python & Math (Linear Algebra, Calculus, Stats) -> 2. Scikit-Learn Tabular ML -> 3. PyTorch Deep Learning -> 4. MLOps & Deployment."
    },

    "devops": {
        "key": "devops",
        "title": "DevOps (Development & Operations)",
        "definition": (
            "**DevOps** is a set of cultural philosophies, automated workflows, and tools that unifies software development (Dev) "
            "with IT operations (Ops) to shorten the systems development life cycle and reliably deliver high-quality software continuously."
        ),
        "types_detail": """### 🧩 Core Disciplines & Sub-Types of DevOps

DevOps encompasses several specialized modern paradigms:

---

#### 1. 🔄 **CI/CD (Continuous Integration & Continuous Deployment)**
* **Continuous Integration:** Developers merge code commits into a central repository multiple times a day; automated tests run to catch bugs immediately.
* **Continuous Delivery/Deployment:** Automated deployment pipelines test and release code to staging and production without manual friction.
* *Tools:* GitHub Actions, GitLab CI, Jenkins, ArgoCD.

---

#### 2. 📜 **Infrastructure as Code (IaC)**
* Provisioning and managing computing infrastructure (servers, VPCs, subnets, databases) using declarative code files rather than manual dashboard clicks.
* *Tools:* Terraform, AWS CloudFormation, Ansible, Pulumi.

---

#### 3. 🐳 **Containerization & Cloud Orchestration**
* Packaging software code, runtimes, and dependencies into lightweight portable containers.
* *Tools:* Docker, Podman, Kubernetes (K8s), Amazon EKS, Helm.

---

#### 4. 🛡️ **DevSecOps (Security-First DevOps)**
* Shifting security left by embedding vulnerability scanning, SAST/DAST, secrets detection, and compliance directly into CI/CD build pipelines.
* *Tools:* Snyk, SonarQube, Trivy, HashiCorp Vault.

---

#### 5. 🤖 **MLOps (Machine Learning Operations)**
* Adapting DevOps practices to machine learning pipelines (data versioning, model artifact registries, automated retraining, inference telemetry).
* *Tools:* MLflow, Kubeflow, DVC, Feast.

---

#### 6. 🌐 **Site Reliability Engineering (SRE)**
* Applying software engineering principles to infrastructure and operations to ensure high availability, uptime, SLOs/SLAs, and error budgets.
* *Tools:* Prometheus, Grafana, Datadog, PagerDuty.
""",
        "ordered_subtopics": [
            "ci/cd",
            "infrastructure as code",
            "containerization",
            "devsecops",
            "mlops",
            "sre"
        ],
        "workflow": """### 🔄 The Standard DevOps Lifecycle (The Infinity Loop)

1. 📝 **Plan:** Defining project milestones, backlog items, and sprint goals (Jira, GitHub Projects).
2. 💻 **Code:** Writing modular code with Git version control and branch protection rules.
3. 🏗️ **Build:** Compiling code and packaging dependencies into Docker container images.
4. 🧪 **Test:** Automated unit tests, integration tests, and security scans in CI pipelines.
5. 📦 **Release:** Tagging release artifacts and publishing Docker images to container registries.
6. 🚀 **Deploy:** Rolling out updates with zero downtime using Kubernetes / blue-green deployments.
7. ⚙️ **Operate:** Managing cloud server infrastructure, scaling auto-scaling groups, and DNS routing.
8. 📊 **Monitor:** Real-time metrics visualization (Grafana), logging (ELK), and alerting (PagerDuty).
""",
        "advantages": """### ⚖️ Benefits & Challenges of DevOps

#### ✅ Key Benefits:
- **High Deployment Velocity:** Deploying updates multiple times daily instead of quarterly releases.
- **Rapid Bug Resolution:** Automated testing catches regressions before reaching production.
- **Scalability & Reliability:** IaC and Kubernetes provide automated failover, load balancing, and self-healing.

#### ⚠️ Key Challenges:
- **Cultural Shift:** Requires breaking silos between developers and system administrators.
- **Tooling Complexity:** Managing distributed microservices, Kubernetes clusters, and cloud IAM security requires strong engineering discipline.
""",
        "pillars": [
            "**CI/CD:** Automated building, testing, and continuous deployment.",
            "**Infrastructure as Code (IaC):** Automated cloud infrastructure via code (Terraform).",
            "**Containerization:** Packaging apps into Docker & Kubernetes clusters.",
            "**Observability & Telemetry:** Real-time monitoring with Prometheus and Grafana."
        ],
        "tools": ["Docker", "Kubernetes", "GitHub Actions", "Terraform", "Linux (Bash)", "AWS / GCP", "Prometheus & Grafana"],
        "examples": "Zero-downtime rolling microservice updates, auto-scaling web traffic during high-load events, and automated test pipelines.",
        "pathway": "1. Master Linux & Shell Scripting -> 2. Git & CI/CD Pipelines -> 3. Docker Containers -> 4. Kubernetes Orchestration -> 5. Cloud & IaC."
    },

    "data science": {
        "key": "data science",
        "title": "Data Science",
        "definition": (
            "**Data Science** is an interdisciplinary field combining domain expertise, programming skills, "
            "and mathematics/statistics to extract actionable insights and predictive intelligence from structured and unstructured data."
        ),
        "types_detail": """### 🧩 The 4 Core Types of Data Analytics & Data Science

1. 📊 **Descriptive Analytics ("What happened?"):** Historical reporting, KPI dashboards, summary aggregations (SQL, Tableau).
2. 🔍 **Diagnostic Analytics ("Why did it happen?"):** Root cause analysis, correlation studies, hypothesis testing ($p$-values).
3. 🔮 **Predictive Analytics ("What will happen next?"):** Forecasting future outcomes with ML algorithms (Regression, Time-Series).
4. 🎯 **Prescriptive Analytics ("What should we do about it?"):** Optimization models and simulation (Linear Programming, RL).
""",
        "ordered_subtopics": [
            "supervised learning",
            "unsupervised learning",
            "semi-supervised learning",
            "reinforcement learning"
        ],
        "pillars": [
            "**Exploratory Data Analysis (EDA):** Cleaning and visualizing data.",
            "**Statistical Modeling:** Hypothesis testing and significance.",
            "**Predictive Modeling:** Supervised and unsupervised ML models.",
            "**Business Intelligence:** Data storytelling with Tableau/Power BI."
        ],
        "tools": ["Python / R", "SQL", "Pandas", "Tableau / Power BI", "Scikit-Learn", "Seaborn / Plotly"],
        "examples": "Customer lifetime value prediction, churn early warning systems, fraud anomaly detection, dynamic pricing algorithms.",
        "pathway": "1. SQL & Advanced Excel -> 2. Python (Pandas, NumPy, Matplotlib) -> 3. Statistics & Probability -> 4. Machine Learning & Dashboarding."
    }
}


# ============================================================
# CONVERSATIONAL CONTEXT RESOLUTION
# ============================================================

def _detect_topic_from_text(text: str) -> Optional[str]:
    """Detects primary knowledge topic key from a piece of text."""
    t_lower = text.lower()
    for key in KNOWLEDGE_BASE:
        if key in t_lower or key.replace(" ", "") in t_lower:
            return key
    if re.search(r"\bml\b", t_lower):
        return "machine learning"
    if re.search(r"\bai\b", t_lower):
        return "machine learning"
    if "dev ops" in t_lower:
        return "devops"
    if "cyber" in t_lower or "security" in t_lower:
        return "cyber security"
    if "cloud" in t_lower or "aws" in t_lower or "azure" in t_lower:
        return "cloud computing"
    if "data" in t_lower and ("analyst" in t_lower or "science" in t_lower or "analysis" in t_lower):
        return "data science"
    return None


def _extract_ordinal_index(text: str) -> Optional[int]:
    """
    Parses ordinal references like '1st', 'the 4th one', 'point 3', 'second', 'number 2'.
    Returns 0-based index (e.g. 4th -> index 3).
    """
    t = text.lower()
    ordinal_map = {
        "1st": 0, "first": 0, "point 1": 0, "number 1": 0, "#1": 0, "the 1st": 0,
        "2nd": 1, "second": 1, "point 2": 1, "number 2": 1, "#2": 1, "the 2nd": 1,
        "3rd": 2, "third": 2, "point 3": 2, "number 3": 2, "#3": 2, "the 3rd": 2,
        "4th": 3, "fourth": 3, "point 4": 3, "number 4": 3, "#4": 3, "the 4th": 3,
        "5th": 4, "fifth": 4, "point 5": 4, "number 5": 4, "#5": 4, "the 5th": 4,
        "6th": 5, "sixth": 5, "point 6": 5, "number 6": 5, "#6": 5, "the 6th": 5,
        "7th": 6, "seventh": 6, "point 7": 6, "number 7": 6, "#7": 6, "the 7th": 6,
    }
    for phrase, idx in ordinal_map.items():
        if phrase in t:
            return idx

    # Match patterns like "explain the 4th", "explain the 4 th one", "tell me about 4th"
    m = re.search(r"\b(\d+)\s*(?:st|nd|rd|th)?\s*(?:one|point|item)?\b", t)
    if m and ("explain" in t or "tell" in t or "what is" in t or "about" in t or "more" in t):
        try:
            num = int(m.group(1))
            if 1 <= num <= 10:
                return num - 1
        except ValueError:
            pass

    return None


def resolve_conversational_context(messages: List[Dict[str, str]]) -> Tuple[str, Optional[str], str, Optional[int]]:
    """
    Analyzes current user query and conversation history to determine:
    1. user_query: Latest clean query string
    2. active_topic: Current subject (e.g. 'machine learning', 'devops')
    3. intent: Query intent ('types', 'how_it_works', 'advantages', 'tools', 'roadmap', 'code', 'ordinal', 'general')
    4. ordinal_idx: If ordinal reference like '4th one', returns index (e.g. 3)
    """
    user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            user_query = msg.get("content", "").strip()
            break
        elif isinstance(msg, str) and not user_query:
            user_query = msg.strip()

    query_lower = user_query.lower()

    # 1. First, check for ordinal reference (e.g. "explain the 4th one", "tell me about the 2nd")
    ordinal_idx = _extract_ordinal_index(query_lower)

    # 2. Check if query mentions a specific subtopic directly (e.g. "explain reinforcement learning", "explain CI/CD")
    for sub_key in SUBTOPIC_EXPLAINERS:
        if sub_key in query_lower or (sub_key == "reinforcement learning" and ("rl" in query_lower.split() or "reinforcement" in query_lower)):
            return user_query, None, f"subtopic_{sub_key}", ordinal_idx

    # 3. Topic detection — only inherit from history for genuine follow-up queries
    topic = _detect_topic_from_text(query_lower)
    if not topic:
        # Check if this query is a genuine follow-up vs a standalone new question
        words = query_lower.split()
        has_pronoun = any(p in words for p in ["it", "its", "this", "that", "they", "them", "their", "those", "these"])
        starts_with_continuation = query_lower.startswith((
            "explain ", "tell me more", "go deeper", "what about", "and what", "and how",
            "more about", "elaborate", "continue", "give more", "details on"
        ))
        has_ordinal_cue = any(w in query_lower for w in [
            "the 1st", "the 2nd", "the 3rd", "the 4th", "the 5th",
            "first one", "second one", "third one", "fourth one", "fifth one", "last one"
        ])
        is_isolated_aspect = query_lower.strip("?! ") in [
            "types", "what are the types", "what are its types", "give types",
            "how it works", "how does it work", "workflow", "lifecycle",
            "advantages", "disadvantages", "pros and cons", "benefits",
            "tools", "tech stack", "tools used", "roadmap", "how to learn",
            "code example", "code", "example", "examples", "interview questions"
        ]

        # Explicitly avoid inheriting if it's a standalone "what is / define / who is / how to <noun>"
        # unless it contains pronouns
        is_standalone_inquiry = (
            query_lower.startswith(("what is ", "what are ", "who is ", "define ", "meaning of ", "explain what "))
            and not has_pronoun
        )

        _is_followup = (has_pronoun or starts_with_continuation or has_ordinal_cue or is_isolated_aspect) and not is_standalone_inquiry

        if _is_followup:
            for msg in reversed(messages[:-1]):
                content = msg.get("content", "")
                found = _detect_topic_from_text(content)
                if found:
                    topic = found
                    break

    # 4. Intent detection
    intent = "general"
    if ordinal_idx is not None:
        intent = "ordinal"
    elif any(w in query_lower for w in ["type", "types", "kinds", "categories", "classification", "classify", "subfield", "subfields", "branches"]):
        intent = "types"
    elif any(w in query_lower for w in ["how it works", "how does it work", "workflow", "lifecycle", "pipeline", "process", "architecture", "mechanism"]):
        intent = "workflow"
    elif any(w in query_lower for w in ["advantage", "advantages", "benefit", "benefits", "pros and cons", "disadvantage", "limitations", "drawbacks"]):
        intent = "advantages"
    elif any(w in query_lower for w in ["tool", "tools", "technology", "technologies", "tech stack", "framework", "frameworks", "libraries"]):
        intent = "tools"
    elif any(w in query_lower for w in ["roadmap", "how to learn", "how to start", "study plan", "pathway", "learn", "prerequisite", "prerequisites"]):
        intent = "roadmap"
    elif any(w in query_lower for w in ["code", "example", "examples", "snippet", "implementation", "program", "syntax"]):
        intent = "code"
    elif any(w in query_lower for w in ["interview", "questions", "prepare", "salary", "job", "career"]):
        intent = "interview"

    return user_query, topic, intent, ordinal_idx


# ============================================================
# MAIN LLM INFERENCE & RESPONSE GENERATION
# ============================================================

def generate_llm_response(
    messages: List[Dict[str, str]],
    max_tokens: int = 400,
    temperature: float = 0.7,
) -> str:
    """
    Context-aware response generator. Resolves ordinals ('explain the 4th one'),
    pronouns, follow-ups, and subtopic queries.

    When document context is present in the system prompt (RAG mode),
    the knowledge base lookup is bypassed and answers are extracted/synthesized
    directly from document content.
    """
    # 0. Check if RAG document context is present in system prompt
    doc_context_text = ""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system" and "DOCUMENT CONTEXT" in (msg.get("content") or ""):
            doc_context_text = msg.get("content", "")
            break
    _has_doc_context = bool(doc_context_text)

    # 1. Try local GGUF if initialized
    model = get_llm_model()
    if model is not None:
        try:
            result = model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
            )
            content = result["choices"][0]["message"]["content"].strip()
            if content and len(content) > 30:
                return content
        except Exception:
            pass

    # 2. Resolve Conversational Context & Intent
    user_query, active_topic, intent, ordinal_idx = resolve_conversational_context(messages)
    query_lower = user_query.lower()

    # 3. Direct Subtopic Match — skip if RAG doc context is active
    if not _has_doc_context and intent.startswith("subtopic_"):
        sub_key = intent.replace("subtopic_", "")
        if sub_key in SUBTOPIC_EXPLAINERS:
            return SUBTOPIC_EXPLAINERS[sub_key]

    # 4. Ordinal Reference Match — skip if RAG doc context is active
    if not _has_doc_context and intent == "ordinal" and ordinal_idx is not None:
        topic_key = active_topic or "machine learning"
        kb_entry = KNOWLEDGE_BASE.get(topic_key, KNOWLEDGE_BASE["machine learning"])
        ordered = kb_entry.get("ordered_subtopics", [])

        if 0 <= ordinal_idx < len(ordered):
            target_subtopic = ordered[ordinal_idx]
            if target_subtopic in SUBTOPIC_EXPLAINERS:
                return SUBTOPIC_EXPLAINERS[target_subtopic]

    # 5. Handle Python Decorators — skip if RAG doc context is active
    if not _has_doc_context and "decorator" in query_lower:
        return """### 🐍 Understanding Python Decorators

A **Decorator** in Python is a design pattern that allows you to dynamically modify or extend the behavior of a function or class **without permanently modifying its source code**.

In Python, functions are **first-class citizens**—meaning they can be passed as arguments, returned from other functions, and assigned to variables.

---

#### 💡 Core Mechanism (How It Works Under the Hood)
A decorator is essentially a function that accepts another function as an input argument, wraps additional logic around it, and returns the modified wrapper function:

```python
import time

def timing_decorator(func):
    \"\"\"Decorator that measures the execution time of any function.\"\"\"
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"⏱️ '{func.__name__}' executed in {end_time - start_time:.4f} seconds.")
        return result
    return wrapper

# Using the decorator with the @ syntax
@timing_decorator
def process_data(n):
    total = sum(i * i for i in range(n))
    return total

# Execution
result = process_data(1_000_000)
```

---

#### 🔑 Common Practical Use Cases in Industry:
1. **Authentication & Authorization:** Verifying user tokens before granting access to web routes (e.g., `@login_required` in Flask/Django).
2. **Logging & Auditing:** Automatically recording function inputs, outputs, and timestamps.
3. **Caching / Memoization:** Storing repeated expensive function outputs (e.g., `@functools.lru_cache`).
4. **Input Validation & Rate Limiting:** Enforcing schema parameters and API request quotas.
"""

    # 6. Handle Contextual Queries with Known Topic — skip entirely if RAG doc context is active
    if not _has_doc_context and active_topic and active_topic in KNOWLEDGE_BASE:
        kb = KNOWLEDGE_BASE[active_topic]

        # Case: "What are its types?" / Types
        if intent == "types" or "type" in query_lower or "category" in query_lower or "branches" in query_lower:
            if "types_detail" in kb:
                return kb["types_detail"]

        # Case: "How does it work?" / Workflow
        if intent == "workflow":
            if "workflow" in kb:
                return kb["workflow"]

        # Case: Advantages / Disadvantages
        if intent == "advantages":
            if "advantages" in kb:
                return kb["advantages"]

        # Case: Tools / Tech Stack
        if intent == "tools":
            tools_list = "\n".join([f"- ⚡ **{t}**" for t in kb["tools"]])
            return f"""### 🛠️ Industry Standard Tech Stack for **{kb['title']}**

Here are the essential tools, libraries, and frameworks used by professionals:

{tools_list}

#### 🚀 Recommended Getting Started Tooling:
Begin by installing **Python 3.11+**, **VS Code / Jupyter Notebooks**, and the core foundational libraries.
"""

        # Case: Code Example
        if intent == "code":
            if active_topic == "machine learning":
                return """### 🐍 Hands-on Python Machine Learning Example (Scikit-Learn)

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Dataset
data = load_iris()
X, y = data.data, data.target

# 2. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Model Training (Random Forest Ensemble)
clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
clf.fit(X_train_scaled, y_train)

# 5. Inference & Evaluation
y_pred = clf.predict(X_test_scaled)
print(f"🎯 Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=data.target_names))
```
"""
            elif active_topic == "devops":
                return """### 🐳 Production Docker & CI/CD Pipeline Example (DevOps)

#### 1. `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 2. `.github/workflows/ci.yml` (GitHub Actions CI):
```yaml
name: CI Test Pipeline
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Run Tests
        run: |
          pip install pytest
          pytest tests/
```
"""

        # Case: General Definition / Overview
        return f"""### 💡 {kb['title']}

{kb['definition']}

---

#### 🔑 Core Pillars & Concepts:
{chr(10).join(['- ' + p for p in kb['pillars']])}

#### 🛠️ Industry Standard Tools & Tech Stack:
{chr(10).join(['- ⚡ `' + t + '`' for t in kb['tools']])}

#### 🚀 Real-World Applications:
{kb['examples']}

#### 🎓 Recommended Learning & Career Roadmap:
{kb['pathway']}

*💬 Tip: You can ask me follow-up questions like **"What are its types?"**, **"Explain the 4th one"**, **"How does it work?"**, or **"Show me a code example"**!*
"""

    # 7. RAG Mode Response Generation (when document chunks are present)
    if _has_doc_context:
        pattern = r"\[Source \d+: ([^\|]+) \| Page (\d+)\]\s*([\s\S]*?)(?=(?:\[Source \d+:|\-\-\- END DOCUMENT CONTEXT|$))"
        matches = re.findall(pattern, doc_context_text)
        if matches:
            from core.rag_engine import synthesize_document_answer
            parsed_chunks = [
                {"filename": fname.strip(), "page": page_num.strip(), "text": content.strip()}
                for fname, page_num, content in matches
            ]
            doc_ans = synthesize_document_answer(user_query, parsed_chunks)
            if doc_ans:
                return doc_ans

    # 8. General Knowledge Fallback
    return (
        f"### 💡 {user_query.strip('?! ').title()}\n\n"
        f"{_general_knowledge_answer(query_lower)}"
    )


def _general_knowledge_answer(query_lower: str) -> str:
    """
    Returns a structured, accurate technical answer for common computing,
    engineering, AI, career, and degree questions.
    Covers: degrees, software engineering, CS, AI, web dev, databases,
    cybersecurity, cloud, DSA, networking, programming languages, and more.
    """
    q = query_lower.strip("?! ")

    # ── Higher Education & Academic Degrees ─────────────────────────────────
    if any(w in q for w in ["phd", "ph.d", "doctorate", "doctoral", "doctor of philosophy"]):
        return (
            "A **Ph.D. in Computer Science (Doctor of Philosophy in CS)** is the highest terminal academic and research degree in the field. "
            "Unlike undergraduate or standard master's degrees that focus on learning existing concepts through coursework, a Ph.D. is dedicated "
            "to **creating original knowledge**, conducting foundational research, and solving previously unsolved computational problems.\n\n"
            "---\n\n"
            "#### ⏱️ Typical Timeline & Program Structure (4–6 Years):\n"
            "1. **Years 1–2 (Foundation & Qualification):**\n"
            "   - Advanced graduate coursework in specialized computer science areas.\n"
            "   - **Qualifying Exams (Quals) / Comprehensive Exams:** Rigorous evaluations testing broad CS and mathematical competence.\n"
            "   - Selecting a Faculty **Research Advisor (PI)** and joining a specialized research lab.\n\n"
            "2. **Years 2–3 (Thesis Proposal & Candidacy):**\n"
            "   - Formulating a unique research hypothesis and conducting in-depth literature review.\n"
            "   - Presenting the formal **Thesis Proposal** to a faculty dissertation committee to achieve **Ph.D. Candidacy (ABD - All But Dissertation)**.\n\n"
            "3. **Years 3–5+ (Research, Experimentation & Publishing):**\n"
            "   - Performing deep experiments, mathematical proofs, software architecture prototypes, or theoretical theorems.\n"
            "   - Publishing peer-reviewed papers in tier-1 conferences (e.g., *NeurIPS, ICML, CVPR, ACL, SOSP, OSDI, STOC, FOCS, IEEE S&P*).\n\n"
            "4. **Final Stage (Dissertation Defense):**\n"
            "   - Authoring a comprehensive book-length doctoral dissertation.\n"
            "   - Public oral defense before a committee of expert examiners.\n\n"
            "---\n\n"
            "#### 🔬 Major Ph.D. Research Specializations:\n"
            "- **Artificial Intelligence & Deep Learning:** Foundation models, Reinforcement Learning, Computer Vision, NLP.\n"
            "- **Systems & Distributed Computing:** Cloud architectures, OS kernels, high-performance computing, distributed databases.\n"
            "- **Cybersecurity & Cryptography:** Zero-knowledge proofs, post-quantum encryption, protocol verification.\n"
            "- **Theoretical Computer Science:** Computational complexity (P vs NP), algorithm design, graph theory.\n"
            "- **Quantum Computing:** Quantum algorithms, quantum error correction, qubit architecture simulation.\n"
            "- **Human-Computer Interaction (HCI) & Robotics:** Autonomous navigation, brain-computer interfaces, assistive robotics.\n\n"
            "---\n\n"
            "#### 💼 Career Opportunities Post-Ph.D.:\n"
            "1. **Academia:** Tenure-track Assistant Professor, Associate Professor, Principal Investigator (PI) leading university research labs.\n"
            "2. **Industrial Research Labs:** Principal Research Scientist at elite R&D labs (*Google DeepMind, OpenAI, Meta FAIR, Microsoft Research, Apple AI*).\n"
            "3. **Deep-Tech Entrepreneurship:** Founding AI/hardware startups or serving as Chief Technology Officer (CTO) / Chief AI Officer (CAIO).\n"
            "4. **Quantitative Finance:** Lead Quantitative Researcher designing algorithmic models at premier hedge funds (Jane Street, Citadel, Two Sigma).\n\n"
            "---\n\n"
            "#### 💰 Funding Note:\n"
            "At most reputable universities globally, Computer Science Ph.D. programs are **fully funded**—meaning admitted students receive a **full tuition waiver** plus a monthly living stipend in exchange for serving as a Graduate Research Assistant (GRA) or Teaching Assistant (TA)."
        )

    if any(w in q for w in ["master in cs", "masters in cs", "ms in cs", "m.s. in computer science", "mtech in cs", "master's in computer science"]):
        return (
            "A **Master's in Computer Science (MS / M.Tech / MSc in CS)** is a 1.5 to 2-year graduate degree designed to bridge undergraduate fundamentals "
            "with advanced specialization and high-level engineering leadership.\n\n"
            "#### 🧩 Two Primary Tracks:\n"
            "1. **Thesis Track (MS with Thesis):** Focuses on research coursework and authoring an academic thesis. Ideal stepping stone toward a Ph.D. or Research Scientist roles.\n"
            "2. **Non-Thesis / Professional Track (MCS / M.Eng):** Coursework-intensive with practical industry capstones. Focuses on advanced software engineering, distributed systems, and ML engineering for immediate industry placement.\n\n"
            "#### 🚀 Key Specialization Domains:\n"
            "- Machine Learning & Data Science\n"
            "- Cloud & Distributed Systems\n"
            "- Cybersecurity & Network Architecture\n"
            "- Software Engineering Leadership & System Design"
        )

    if any(w in q for w in ["bachelor in cs", "bachelors in cs", "bs in cs", "b.tech in cs", "be in cs", "btech in computer science"]):
        return (
            "A **Bachelor's in Computer Science (BS / B.Tech / BE in CS)** is a 3 to 4-year undergraduate degree establishing core theoretical and applied foundations in computing.\n\n"
            "#### 🧱 Core Curriculum Pillars:\n"
            "1. **Programming Fundamentals:** Python, C++, Java, Object-Oriented Programming (OOP).\n"
            "2. **Data Structures & Algorithms (DSA):** Arrays, Trees, Graphs, Sorting, Dynamic Programming, Big O time/space complexity.\n"
            "3. **Computer Systems:** Computer Architecture, Operating Systems (OS), Computer Networks, Database Management Systems (DBMS).\n"
            "4. **Mathematics:** Discrete Mathematics, Linear Algebra, Multivariable Calculus, Probability & Statistics.\n"
            "5. **Capstones & Internships:** Industry projects, open-source contributions, and technical internships."
        )

    if any(w in q for w in ["phd vs master", "ms vs phd", "master vs phd", "difference between ms and phd"]):
        return (
            "### ⚖️ Master's (MS) vs Ph.D. in Computer Science\n\n"
            "| Dimension | Master's in CS (MS/M.Tech) | Ph.D. in Computer Science |\n"
            "| :--- | :--- | :--- |\n"
            "| **Primary Goal** | Deepen technical mastery & applied engineering skills | Discover and create **new, original scientific knowledge** |\n"
            "| **Typical Duration** | 1.5 – 2 Years | 4 – 6 Years |\n"
            "| **Funding** | Usually self-funded / student loans (some TA/RA) | **Fully Funded** (Full tuition waiver + monthly stipend) |\n"
            "| **Key Output** | Course credits + Capstone / Master's Project | Doctoral Dissertation + Tier-1 Conference Publications |\n"
            "| **Best For** | Senior Software Engineers, ML Engineers, Tech Leads | University Professors, Principal Research Scientists, AI Lab Leads |"
        )

    if any(w in q for w in ["deep learning", "neural network", "neural networks"]):
        return (
            "**Deep Learning** is a specialized subfield of Machine Learning based on **Artificial Neural Networks** with multiple hidden layers (hence 'deep').\n\n"
            "#### 🧠 Core Architectures:\n"
            "1. **Feedforward Neural Networks (ANN / Multi-Layer Perceptrons):** Tabular predictions and baseline classification.\n"
            "2. **Convolutional Neural Networks (CNN):** Spatial feature hierarchies for computer vision, image classification, and medical scans.\n"
            "3. **Recurrent Neural Networks (RNN / LSTM / GRU):** Sequential memory processing for time-series forecasting and text/audio streams.\n"
            "4. **Transformers & Self-Attention:** Parallelized sequence models powering modern LLMs (GPT-4, Gemini, Claude, LLaMA) and Vision Transformers (ViT).\n\n"
            "#### 🛠️ Core Frameworks:\n"
            "- **PyTorch** (Industry standard for research & production)\n"
            "- **TensorFlow / Keras** (Enterprise & edge deployment)\n"
            "- **Hugging Face Transformers** (Pretrained NLP & vision models)"
        )

    if any(w in q for w in ["large language model", "llm", "generative ai", "genai", "transformer"]):
        return (
            "**Large Language Models (LLMs)** are foundation deep learning models trained on trillions of words of text to understand, generate, and reason with human language.\n\n"
            "#### ⚙️ How LLMs Work:\n"
            "1. **Pre-training (Next-Token Prediction):** Self-supervised learning predicting the next word across massive web datasets.\n"
            "2. **Instruction Fine-Tuning (SFT):** Training on curated question-answer and task demonstration pairs.\n"
            "3. **Alignment (RLHF / DPO):** Reinforcement Learning from Human Feedback ensuring helpfulness, accuracy, and safety.\n"
            "4. **Inference & RAG:** Augmenting model generation with verified vector databases using Retrieval-Augmented Generation (RAG)."
        )

    # ── Software Engineering ─────────────────────────────────────────────────
    if any(w in q for w in ["software engineering", "software engineer", "beng software", "b.eng software",
                             "be software", "bsc software", "bachelor of software"]):
        return (
            "**Software Engineering** is the systematic, disciplined, and quantifiable application of engineering principles to the "
            "design, development, testing, deployment, and maintenance of software systems.\n\n"
            "---\n\n"
            "#### 🎓 BEng / BSc Software Engineering Degree (3–4 Years):\n"
            "**Core Modules:**\n"
            "1. **Programming Fundamentals:** Python, Java, C++ and Object-Oriented Design.\n"
            "2. **Data Structures & Algorithms (DSA):** Arrays, Linked Lists, Trees, Graphs, Sorting, Dynamic Programming.\n"
            "3. **Software Design Patterns:** MVC, Observer, Factory, Repository — the Gang of Four patterns.\n"
            "4. **Database Systems:** SQL (PostgreSQL, MySQL), NoSQL (MongoDB), ORM frameworks.\n"
            "5. **Software Testing & QA:** Unit testing (pytest, JUnit), TDD, BDD.\n"
            "6. **Agile Project Management:** Scrum sprints, Kanban boards, CI/CD pipelines.\n"
            "7. **Software Architecture:** Monolithic vs. Microservices, REST APIs.\n"
            "8. **Web & App Development:** Frontend (React/Vue), Backend (Node.js/FastAPI/Django).\n\n"
            "---\n\n"
            "#### ✅ Prerequisites for Admission:\n"
            "- Strong Mathematics background (Calculus, Discrete Math)\n"
            "- Basic programming experience (Python or any language is ideal)\n"
            "- Problem-solving aptitude and logical thinking\n\n"
            "---\n\n"
            "#### 💼 Career Prospects & Roles:\n"
            "| Role | Average Salary (Global) |\n"
            "| :--- | :--- |\n"
            "| **Software Engineer / Developer** | $80K – $150K/yr |\n"
            "| **Full-Stack Web Developer** | $75K – $140K/yr |\n"
            "| **DevOps / Site Reliability Engineer** | $90K – $160K/yr |\n"
            "| **Machine Learning Engineer** | $110K – $180K/yr |\n"
            "| **Cloud Solutions Architect** | $120K – $200K/yr |\n"
            "| **Engineering Manager / CTO** | $150K – $300K+/yr |\n\n"
            "#### 🚀 Recommended Learning Roadmap:\n"
            "1. Master Python / Java fundamentals + DSA (LeetCode)\n"
            "2. Learn Git, Linux, and command-line tools\n"
            "3. Build full-stack web apps (React + FastAPI or Node.js)\n"
            "4. Learn Docker, CI/CD, and cloud deployment (AWS/GCP)\n"
            "5. Contribute to open-source and build a strong GitHub portfolio"
        )

    # ── Computer Science ─────────────────────────────────────────────────────
    if any(w in q for w in ["computer science", "cs degree", "bsc cs", "bsc computer", "what is cs"]):
        return (
            "**Computer Science (CS)** is the theoretical and practical study of computation, algorithms, data structures, "
            "artificial intelligence, software systems, and the mathematical foundations of computing.\n\n"
            "---\n\n"
            "#### 🧩 Core CS Disciplines:\n"
            "1. **Algorithms & Data Structures:** The mathematical backbone of efficient programming.\n"
            "2. **Artificial Intelligence & Machine Learning:** Teaching computers to learn and reason.\n"
            "3. **Operating Systems:** Process scheduling, memory management, and system calls.\n"
            "4. **Computer Networks:** TCP/IP, DNS, HTTP/HTTPS, routing protocols.\n"
            "5. **Database Systems:** Relational (SQL) and NoSQL data modeling and query optimization.\n"
            "6. **Software Engineering:** Design patterns, testing, agile methodology, DevOps.\n"
            "7. **Computer Architecture:** CPU pipelines, cache hierarchies, memory models.\n"
            "8. **Theory of Computation:** Turing machines, finite automata, NP-completeness.\n\n"
            "---\n\n"
            "#### 💼 Career Paths:\n"
            "- Software Engineer → Senior SWE → Principal Engineer → Engineering Director\n"
            "- Data Scientist → ML Engineer → AI Researcher → Research Scientist\n"
            "- Cloud Architect → Solutions Architect → VP of Engineering → CTO\n"
            "- Security Engineer → Penetration Tester → CISO"
        )

    # ── Web Development ──────────────────────────────────────────────────────
    if any(w in q for w in ["web development", "web dev", "frontend", "front-end", "backend", "back-end",
                              "full stack", "fullstack", "react", "node.js", "django", "flask", "fastapi"]):
        return (
            "**Web Development** encompasses building and maintaining websites and web applications — from visual interfaces to server-side logic and databases.\n\n"
            "---\n\n"
            "#### 🗂️ The Three Layers of Web Development:\n\n"
            "**1. 🎨 Frontend (Client-Side):**\n"
            "- Languages: HTML5, CSS3, JavaScript/TypeScript\n"
            "- Frameworks: React.js, Vue.js, Next.js, Svelte\n\n"
            "**2. ⚙️ Backend (Server-Side):**\n"
            "- Python: FastAPI (high-performance), Django (full-featured), Flask (lightweight)\n"
            "- JavaScript: Node.js + Express.js\n\n"
            "**3. 🗄️ Databases & Storage:**\n"
            "- SQL: PostgreSQL, MySQL, SQLite\n"
            "- NoSQL: MongoDB, Redis, DynamoDB\n\n"
            "---\n\n"
            "#### 🚀 Full-Stack Roadmap (6–12 Months):\n"
            "1. HTML + CSS + JavaScript fundamentals\n"
            "2. React.js for modern UI development\n"
            "3. FastAPI or Node.js for REST API backends\n"
            "4. PostgreSQL + ORM (SQLAlchemy / Prisma)\n"
            "5. Docker, CI/CD, deploy to Vercel / Railway / AWS"
        )

    # ── Cybersecurity ────────────────────────────────────────────────────────
    if any(w in q for w in ["cybersecurity", "cyber security", "information security", "infosec",
                              "ethical hacking", "penetration testing", "pentesting"]):
        return (
            "**Cybersecurity** is the practice of protecting digital systems, networks, data, and critical infrastructure from unauthorized access, "
            "attacks, damage, or theft.\n\n"
            "---\n\n"
            "#### 🛡️ Core Cybersecurity Domains:\n"
            "1. **Network Security:** Firewalls, IDS/IPS, VPNs, zero-trust architecture.\n"
            "2. **Application Security (AppSec):** OWASP Top 10, SAST/DAST scanning.\n"
            "3. **Ethical Hacking & Penetration Testing:** Simulated attacks to discover vulnerabilities.\n"
            "4. **Cryptography:** AES, RSA, TLS/SSL, post-quantum cryptography.\n"
            "5. **Cloud Security:** IAM policies, AWS GuardDuty, Azure Sentinel.\n"
            "6. **Incident Response:** SIEM systems (Splunk), log analysis, threat hunting.\n\n"
            "#### 💼 Career Roles & Salaries:\n"
            "| Role | Avg. Salary |\n"
            "| :--- | :--- |\n"
            "| **Security Analyst (SOC)** | $65K–$100K |\n"
            "| **Penetration Tester / Ethical Hacker** | $85K–$140K |\n"
            "| **Cloud Security Architect** | $130K–$200K |\n"
            "| **CISO** | $200K–$400K+ |\n\n"
            "#### 🎓 Top Certifications:\n"
            "CompTIA Security+ → CEH → OSCP → CISSP → CISM"
        )

    # ── Cloud Computing ──────────────────────────────────────────────────────
    if any(w in q for w in ["cloud computing", "cloud services", "aws", "azure", "google cloud", "gcp",
                              "cloud architecture"]):
        return (
            "**Cloud Computing** delivers on-demand computing resources — servers, storage, databases, networking, and AI — "
            "over the internet with pay-as-you-go pricing.\n\n"
            "---\n\n"
            "#### ☁️ The Three Service Models:\n"
            "1. **IaaS:** Virtual machines, storage, networking. (AWS EC2, Azure VM, GCP Compute Engine)\n"
            "2. **PaaS:** Managed runtime environments. (Heroku, Google App Engine, Azure App Service)\n"
            "3. **SaaS:** Full software via browser. (Gmail, Salesforce, Office 365)\n\n"
            "#### 🌐 The Big 3 Cloud Providers:\n"
            "| Provider | Market Share | Flagship Services |\n"
            "| :--- | :--- | :--- |\n"
            "| **AWS (Amazon)** | ~32% | EC2, S3, Lambda, RDS, SageMaker |\n"
            "| **Microsoft Azure** | ~23% | Azure AD, AKS, Cosmos DB, Azure OpenAI |\n"
            "| **Google Cloud (GCP)** | ~12% | BigQuery, Vertex AI, GKE, Cloud Run |\n\n"
            "#### 🎓 Cloud Career Roadmap:\n"
            "Linux → Networking → AWS Cloud Practitioner → Solutions Architect → Docker/Kubernetes (CKA) → Terraform"
        )

    # ── Data Structures & Algorithms ─────────────────────────────────────────
    if any(w in q for w in ["data structure", "data structures", "algorithm", "algorithms", "dsa",
                              "big o", "time complexity", "sorting"]):
        return (
            "**Data Structures & Algorithms (DSA)** form the foundational pillars of computer science and software engineering.\n\n"
            "---\n\n"
            "#### 📦 Essential Data Structures:\n"
            "| Structure | Use Case | Time Complexity |\n"
            "| :--- | :--- | :--- |\n"
            "| **Array** | Random access | O(1) read |\n"
            "| **Hash Table** | Key-value lookups | O(1) avg |\n"
            "| **Binary Search Tree** | Sorted data | O(log n) avg |\n"
            "| **Heap** | Priority queues | O(log n) insert |\n"
            "| **Graph** | Networks, pathfinding | O(V+E) BFS/DFS |\n\n"
            "#### ⚙️ Core Algorithm Categories:\n"
            "1. **Sorting:** QuickSort O(n log n), MergeSort O(n log n)\n"
            "2. **Searching:** Binary Search O(log n), BFS/DFS O(V+E)\n"
            "3. **Dynamic Programming:** Memoization for overlapping subproblems\n"
            "4. **Greedy Algorithms:** Huffman coding, Kruskal's MST\n\n"
            "#### 🎯 Interview Prep:\n"
            "Practice LeetCode: Arrays → Hashing → Two Pointers → Sliding Window → Binary Search → Trees → Graphs → DP"
        )

    # ── Database / SQL ───────────────────────────────────────────────────────
    if any(w in q for w in ["database", "sql", "nosql", "mongodb", "postgresql", "mysql", "rdbms"]):
        return (
            "**Databases** are organized collections of data managed by a Database Management System (DBMS).\n\n"
            "---\n\n"
            "#### 📊 Relational (SQL) vs NoSQL:\n"
            "| Feature | Relational (SQL) | NoSQL |\n"
            "| :--- | :--- | :--- |\n"
            "| **Schema** | Fixed, table-based | Flexible, schema-less |\n"
            "| **ACID Compliance** | ✅ Fully ACID | Varies |\n"
            "| **Best For** | Banking, ERP, e-commerce | Real-time apps, IoT |\n"
            "| **Examples** | PostgreSQL, MySQL, SQLite | MongoDB, Redis, Cassandra |\n\n"
            "#### 🛠️ Core SQL:\n"
            "```sql\n"
            "SELECT name, AVG(salary) FROM employees\n"
            "GROUP BY department HAVING AVG(salary) > 80000\n"
            "ORDER BY AVG(salary) DESC;\n"
            "```"
        )

    # ── Artificial Intelligence ──────────────────────────────────────────────
    if any(w in q for w in ["artificial intelligence", "what is ai", "ai overview", "ai basics"]):
        return (
            "**Artificial Intelligence (AI)** is the broad science of engineering intelligent machines that can simulate human cognitive capabilities "
            "such as learning, reasoning, problem-solving, perception, and language understanding.\n\n"
            "---\n\n"
            "#### 🧠 The AI Hierarchy:\n"
            "- **AI** → Machine Learning → Deep Learning → Computer Vision / NLP / Generative AI\n\n"
            "#### 🔑 Major AI Domains:\n"
            "1. **Machine Learning:** Statistical algorithms that learn from data.\n"
            "2. **Computer Vision:** Understanding images (object detection, OCR, segmentation).\n"
            "3. **Natural Language Processing (NLP):** Language translation, summarization, chatbots.\n"
            "4. **Generative AI:** LLMs (GPT-4, Gemini, Claude), Diffusion Models (DALL-E, Stable Diffusion).\n"
            "5. **Robotics:** Autonomous navigation, manipulation, and physical AI.\n\n"
            "#### 💼 Top AI Career Roles:\n"
            "AI/ML Engineer | Research Scientist | MLOps Engineer | AI Product Manager | Prompt Engineer"
        )

    # ── Python ───────────────────────────────────────────────────────────────
    if "python" in q and any(w in q for w in ["what is", "explain", "about", "language", "programming"]):
        return (
            "**Python** is a high-level, dynamically-typed, interpreted programming language renowned for clean syntax and versatility.\n\n"
            "---\n\n"
            "#### 🌟 Why Python Dominates Modern Tech:\n"
            "- **#1 Language for AI/ML:** NumPy, Pandas, Scikit-Learn, PyTorch, TensorFlow, Hugging Face.\n"
            "- **Web Development:** Django, FastAPI, Flask (used by Instagram, Spotify, Dropbox).\n"
            "- **Data Engineering:** PySpark, Airflow, Dask for big data pipelines.\n"
            "- **Automation & Scripting:** Web scraping (BeautifulSoup, Selenium), system automation.\n"
            "- **Scientific Computing:** SciPy, Matplotlib, Jupyter Notebooks.\n\n"
            "#### 📚 Python Learning Roadmap:\n"
            "1. Basics: variables, data types, functions, OOP\n"
            "2. Intermediate: decorators, generators, file I/O, exception handling\n"
            "3. Libraries: NumPy → Pandas → Matplotlib → Scikit-learn → PyTorch"
        )

    # ── JavaScript / TypeScript ──────────────────────────────────────────────
    if any(w in q for w in ["javascript", "typescript", "what is js", "node.js"]):
        return (
            "**JavaScript (JS)** is the native programming language of web browsers and enables interactive, dynamic web experiences.\n"
            "**TypeScript** adds static type checking on top of JavaScript.\n\n"
            "**Frontend:** Vanilla JS → React.js → Next.js\n"
            "**Backend:** Node.js + Express.js / Fastify / NestJS\n"
            "**Tools:** npm/yarn, Webpack, Vite, ESLint, Jest\n"
            "**Used by:** Meta, Google, Netflix, Airbnb, Twitter."
        )

    # ── Operating Systems ────────────────────────────────────────────────────
    if any(w in q for w in ["operating system", "linux", "what is os", "unix", "os kernel"]):
        return (
            "An **Operating System (OS)** is the fundamental system software managing computer hardware resources.\n\n"
            "#### 🔑 Core OS Functions:\n"
            "1. **Process Management:** Scheduling CPU time (Round Robin, Priority-based)\n"
            "2. **Memory Management:** Virtual memory, paging, heap/stack allocation\n"
            "3. **File System:** Organizing and accessing files (ext4, NTFS, APFS)\n"
            "4. **I/O Device Management:** Device drivers and interrupt handling\n"
            "5. **Security & Access Control:** User permissions, privilege rings\n\n"
            "#### 🐧 Linux — The Developer's OS:\n"
            "- Powers 96% of web servers, Android, and cloud infrastructure\n"
            "- Distros: Ubuntu (beginner), Debian (stable), Arch (advanced)\n"
            "- Key tools: bash, vim, grep, awk, sed, systemd, ssh, Docker"
        )

    # ── Networking ───────────────────────────────────────────────────────────
    if any(w in q for w in ["networking", "computer network", "tcp/ip", "what is tcp", "dns"]):
        return (
            "**Computer Networking** connects computers and devices to share data across local and global networks.\n\n"
            "#### 🌐 OSI 7-Layer Model (simplified):\n"
            "| Layer | Example Protocols |\n"
            "| :--- | :--- |\n"
            "| Application (7) | HTTP, FTP, SMTP, DNS |\n"
            "| Transport (4) | **TCP, UDP** |\n"
            "| Network (3) | **IP, ICMP, BGP** |\n"
            "| Data Link (2) | Ethernet, Wi-Fi |\n\n"
            "#### 🔑 Key Concepts:\n"
            "- **TCP vs UDP:** TCP is reliable/ordered (web, email). UDP is fast/unreliable (video, gaming, DNS).\n"
            "- **HTTP/HTTPS:** Request-response protocol of the web. HTTPS adds TLS encryption.\n"
            "- **DNS:** Translates domain names (google.com) to IP addresses.\n"
            "- **Load Balancers:** Distribute traffic across servers (Nginx, AWS ALB)."
        )

    # ── Agile / Scrum ────────────────────────────────────────────────────────
    if any(w in q for w in ["agile", "scrum", "kanban", "sprint", "project management"]):
        return (
            "**Agile** is an iterative software development methodology delivering working software in short cycles called **Sprints** (1–4 weeks).\n\n"
            "#### 🔄 Scrum Framework:\n"
            "- **Roles:** Product Owner, Scrum Master, Development Team\n"
            "- **Ceremonies:** Sprint Planning → Daily Standup → Sprint Review → Retrospective\n"
            "- **Artifacts:** Product Backlog, Sprint Backlog, Increment\n\n"
            "#### 📋 Kanban:\n"
            "- Visual workflow: **To Do → In Progress → Review → Done**\n"
            "- Limit Work-In-Progress (WIP) to maximize flow\n\n"
            "#### 🛠️ Tools: Jira, Trello, Linear, GitHub Projects, Notion"
        )

    # ── Generic Intelligent Fallback ─────────────────────────────────────────
    clean_topic = query_lower.strip("?! ").title()
    return (
        f"**{clean_topic}** is an important topic in the world of computing, software engineering, and technology.\n\n"
        f"---\n\n"
        f"#### 🔑 Conceptual Overview:\n"
        f"This domain applies systematic principles of computation and software development to solve real-world problems. "
        f"Professionals in this area combine theoretical knowledge with practical engineering skills.\n\n"
        f"#### 🛠️ Key Skills & Technologies Typically Involved:\n"
        f"- **Programming:** Python, Java, JavaScript, or domain-specific languages\n"
        f"- **Version Control:** Git (branches, pull requests, code reviews)\n"
        f"- **System Design:** Scalable architecture, APIs, and data modeling\n"
        f"- **Testing & CI/CD:** Unit tests, integration tests, automated pipelines\n"
        f"- **Deployment:** Docker, cloud platforms (AWS/GCP/Azure)\n\n"
        f"#### 🎯 Ask Me More Specific Questions:\n"
        f"- *\"What are the types of {clean_topic}?\"*\n"
        f"- *\"How does {clean_topic} work?\"*\n"
        f"- *\"What tools are used in {clean_topic}?\"*\n"
        f"- *\"Give me a code example of {clean_topic}\"*\n"
        f"- *\"What is the career roadmap for {clean_topic}?\"*\n"
        f"- *\"What jobs are available in {clean_topic}?\"*\n"
    )



def build_system_prompt(
    mode: str = "💼 Career Assistant",
    resume_text: str = None,
    ats_score: int = None,
    career_recs: list = None,
    course_recs: list = None,
) -> str:
    """Constructs enriched context-aware system prompts without hallucinated data."""
    if mode == "📚 Study Assistant":
        prompt = """You are EduCareer AI, an advanced educational intelligence assistant.
Your mission is to help students understand complex engineering, data science, AI, and computer science concepts clearly.
Break difficult topics into steps, provide structured explanations, real-world examples, and code blocks.
"""
    else:
        prompt = """You are EduCareer AI, an expert career counselor and technical mentor.
Your mission is to guide students and professionals on learning roadmaps, career transitions, technical interview preparation, resume optimization, and industry skill requirements.
Give practical, structured, and beginner-friendly advice.
"""

    context_parts = []
    if career_recs:
        top_careers_str = ", ".join([f"{c['Career']} ({c.get('Suitability_Percent', 0):.1f}%)" if isinstance(c, dict) else str(c) for c in career_recs[:3]])
        context_parts.append(f"CURRENT TOP ML RECOMMENDED CAREERS: {top_careers_str}")

    if course_recs:
        top_courses_str = ", ".join([c['course'] if isinstance(c, dict) else str(c) for c in course_recs[:3]])
        context_parts.append(f"CURRENT TOP ML RECOMMENDED COURSES: {top_courses_str}")

    if resume_text:
        score_str = f"Estimated ATS Score: {ats_score}/100\n" if ats_score is not None else ""
        context_parts.append(
            f"ACTIVE RESUME CONTEXT (Facts Only):\n{score_str}Resume excerpt: {resume_text[:2000]}"
        )

    if context_parts:
        prompt += "\n--- VERIFIED CONTEXT ---\n" + "\n\n".join(context_parts) + "\n------------------------\n"

    return prompt

