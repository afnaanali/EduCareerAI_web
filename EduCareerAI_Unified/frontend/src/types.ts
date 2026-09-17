export type User = {
  id: number;
  username: string;
  email: string;
  full_name?: string;
};

export type CareerOption = {
  fields: string[];
  hobbies: { id: string; label: string }[];
  grades: { id: string; label: string }[];
  aptitudes: { id: string; label: string }[];
};

export type CareerPrediction = {
  Rank: number;
  Career: string;
  Probability: number;
  Suitability_Percent: number;
};

export type CourseRecommendation = {
  course: string;
  field: string;
  difficulty?: string;
  score?: number;
};

export type CareerPredictionResult = {
  success: boolean;
  careers: CareerPrediction[];
  courses: CourseRecommendation[];
  input_summary: {
    field: string;
    academic_avg: number;
    aptitude_avg: number;
    active_hobbies_count: number;
  };
};

export type RoadmapPhase = {
  phase: string;
  topics: string[];
  project: string;
};

export type RoadmapData = {
  title: string;
  description: string;
  duration: string;
  phases: RoadmapPhase[];
  certifications: string[];
  key_skills: string[];
};

export type ResumeAnalysisResult = {
  success: boolean;
  resume_name: string;
  ats_score: number;
  breakdown: Record<string, number>;
  extracted_skills: string[];
  ann_vector: {
    skills: string[];
    vector: number[];
  };
  target_job?: string;
  match_score?: number;
  matched_job_skills: string[];
  missing_job_skills: string[];
  skill_gap: string[];
  matched_careers: { career: string; match: number; why: string[] }[];
  issues: { title: string; why: string; fix: string }[];
  sections_detected: Record<string, boolean>;
  word_count: number;
};

export type ChatMessage = {
  role: 'user' | 'assistant';
  content: string;
  citations?: string;
  has_rag_context?: boolean;
};

export type DeepLearningANNResult = {
  success: boolean;
  skill_vector: number[];
  top_career?: string;
  confidence?: number;
  predictions: { Career: string; Confidence: number; Probability?: number }[];
};

export type DigitCNNResult = {
  success: boolean;
  predicted_digit: number;
  confidence: number;
  probabilities: number[];
};

export type SentimentResult = {
  success: boolean;
  comparison: {
    lstm: { sentiment: string; confidence: number; raw_score?: number };
    rnn: { sentiment: string; confidence: number; raw_score?: number };
  };
  coach: {
    impact_score?: number;
    tone_label?: string;
    tone_badge?: string;
    tone_color?: string;
    word_count?: number;
    found_action_verbs?: string[];
    found_weak_phrases?: { phrase: string; fix: string }[];
    star_breakdown?: Record<string, boolean>;
    star_score?: number;
    rephrased_preview?: string;
    tone_feedback: string[];
    actionable_tips: string[];
  };
};

export type InterviewQuestionItem = {
  id: string;
  role: string;
  category: 'behavioral' | 'technical' | 'leadership' | 'situational';
  difficulty?: 'Entry-Level' | 'Mid-Level' | 'Senior / Lead';
  question: string;
  hint: string;
  modelAnswer: string;
  skills?: string[];
  isAiGenerated?: boolean;
};

export type InterviewGenerateRequest = {
  role: string;
  category?: string;
  difficulty?: string;
  count?: number;
  focus_topics?: string;
};

export type InterviewGenerateResponse = {
  success: boolean;
  role: string;
  count: number;
  questions: InterviewQuestionItem[];
};
