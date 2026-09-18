import axios from 'axios';
import type {
  CareerOption,
  CareerPredictionResult,
  RoadmapData,
  ResumeAnalysisResult,
  DeepLearningANNResult,
  DigitCNNResult,
  SentimentResult,
  InterviewGenerateRequest,
  InterviewGenerateResponse,
  InterviewAnswerTipsRequest,
  InterviewAnswerTipsResponse,
} from './types';

const API_BASE =
  import.meta.env.VITE_API_URL ||
  (typeof window !== 'undefined' &&
  (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') &&
  window.location.port !== '8000'
    ? 'http://127.0.0.1:8000'
    : '');

const client = axios.create({
  baseURL: API_BASE,
});

// Interceptor: let browser/axios attach the proper multipart boundary for FormData payloads
client.interceptors.request.use((config) => {
  if (config.data instanceof FormData) {
    if (config.headers) {
      delete config.headers['Content-Type'];
      delete config.headers['content-type'];
    }
  } else {
    if (!config.headers['Content-Type'] && !config.headers['content-type']) {
      config.headers['Content-Type'] = 'application/json';
    }
  }
  return config;
});

export const api = {
  // Health & Info
  checkHealth: async () => {
    const res = await client.get('/api/health');
    return res.data;
  },

  getSystemInfo: async () => {
    const res = await client.get('/api/info');
    return res.data;
  },

  // Auth & Profile
  register: async (data: { username: string; email: string; password: string; full_name?: string }) => {
    const res = await client.post('/api/auth/register', data);
    return res.data;
  },

  login: async (data: { identifier: string; password: string }) => {
    const res = await client.post('/api/auth/login', data);
    return res.data;
  },

  getProfile: async (userId: number) => {
    const res = await client.get(`/api/auth/profile/${userId}`);
    return res.data;
  },

  saveProfile: async (userId: number, profile: any) => {
    const res = await client.post('/api/auth/profile', { user_id: userId, profile });
    return res.data;
  },

  getUserHistory: async (userId: number) => {
    const res = await client.get(`/api/auth/history/${userId}`);
    return res.data;
  },

  deleteAssessment: async (userId: number, assessmentId: number) => {
    const res = await client.delete(`/api/auth/history/assessment/${userId}/${assessmentId}`);
    return res.data;
  },

  deleteResumeScan: async (userId: number, scanId: number) => {
    const res = await client.delete(`/api/auth/history/resume/${userId}/${scanId}`);
    return res.data;
  },

  // Career & Courses
  getCareerOptions: async (): Promise<CareerOption> => {
    const res = await client.get('/api/career/options');
    return res.data;
  },

  predictCareers: async (payload: {
    user_id?: number;
    field: string;
    hobbies: Record<string, number>;
    grades: Record<string, number>;
    aptitudes: Record<string, number>;
    top_n?: number;
  }): Promise<CareerPredictionResult> => {
    const res = await client.post('/api/career/predict', payload);
    return res.data;
  },

  // Roadmaps
  listRoadmapCareers: async (): Promise<{ careers: string[]; total: number }> => {
    const res = await client.get('/api/roadmap/careers');
    return res.data;
  },

  getRoadmap: async (careerName: string): Promise<{ career: string; roadmap: RoadmapData }> => {
    const res = await client.get(`/api/roadmap/${encodeURIComponent(careerName)}`);
    return res.data;
  },

  // Resume & ATS
  analyzeResumeText: async (payload: {
    user_id?: number;
    resume_text: string;
    resume_name?: string;
    target_job?: string;
    job_description?: string;
  }): Promise<ResumeAnalysisResult> => {
    const res = await client.post('/api/resume/analyze-text', payload);
    return res.data;
  },

  analyzeResumeFile: async (formData: FormData): Promise<ResumeAnalysisResult> => {
    const res = await client.post('/api/resume/analyze-file', formData);
    return res.data;
  },

  getTargetRoles: async (): Promise<{ roles: string[] }> => {
    const res = await client.get('/api/resume/target-roles');
    return res.data;
  },

  // Chat & RAG
  sendChatMessage: async (payload: {
    session_id?: string;
    user_id?: number;
    message: string;
    mode?: string;
    resume_context?: string;
    ats_score?: number;
    career_recs?: any[];
    course_recs?: any[];
    history?: { role: string; content: string }[];
  }): Promise<{ success: boolean; response: string; citations?: string; has_rag_context: boolean }> => {
    const res = await client.post('/api/chat/message', payload);
    return res.data;
  },

  uploadRagDocument: async (formData: FormData) => {
    const res = await client.post('/api/chat/rag/upload', formData);
    return res.data;
  },

  getRagStatus: async (sessionId: string) => {
    const res = await client.get(`/api/chat/rag/status/${sessionId}`);
    return res.data;
  },

  clearRag: async (sessionId: string) => {
    const res = await client.delete(`/api/chat/rag/clear/${sessionId}`);
    return res.data;
  },

  getChatHistory: async (userId: number) => {
    const res = await client.get(`/api/chat/history/${userId}`);
    return res.data;
  },

  clearChatHistory: async (userId: number) => {
    const res = await client.delete(`/api/chat/history/${userId}`);
    return res.data;
  },

  // Deep Learning Lab
  getAnnSkills: async (): Promise<{ skills: string[] }> => {
    const res = await client.get('/api/lab/ann-skills');
    return res.data;
  },

  predictAnn: async (skillVector: number[]): Promise<DeepLearningANNResult> => {
    const res = await client.post('/api/lab/ann-predict', { skill_vector: skillVector });
    return res.data;
  },

  predictDigit: async (formData: FormData): Promise<DigitCNNResult> => {
    const res = await client.post('/api/lab/cnn-digit', formData);
    return res.data;
  },

  analyzeMarksheet: async (formData: FormData) => {
    const res = await client.post('/api/lab/cnn-marksheet', formData);
    return res.data;
  },

  analyzeSentimentTone: async (text: string): Promise<SentimentResult> => {
    const res = await client.post('/api/lab/sentiment-tone', { text });
    return res.data;
  },

  // AI Interview Question Generator
  getInterviewRoles: async (): Promise<{ success: boolean; roles: string[] }> => {
    const res = await client.get('/api/interview/roles');
    return res.data;
  },

  generateInterviewQuestions: async (
    payload: InterviewGenerateRequest
  ): Promise<InterviewGenerateResponse> => {
    const res = await client.post('/api/interview/generate', payload);
    return res.data;
  },

  getInterviewAnswerAndTips: async (
    payload: InterviewAnswerTipsRequest
  ): Promise<InterviewAnswerTipsResponse> => {
    const res = await client.post('/api/interview/answer-and-tips', payload);
    return res.data;
  },
};
