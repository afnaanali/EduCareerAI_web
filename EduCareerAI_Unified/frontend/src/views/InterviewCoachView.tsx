import React, { useState, useEffect, useRef } from 'react';
import {
  Mic,
  MicOff,
  Sparkles,
  TrendingUp,
  RotateCcw,
  CheckCircle2,
  AlertTriangle,
  Lightbulb,
  ArrowRight,
  Award,
  BookOpen,
  Volume2,
  RefreshCw,
  HelpCircle,
  Copy,
  Check,
} from 'lucide-react';
import { api } from '../api';
import type { SentimentResult, InterviewQuestionItem, InterviewAnswerTipsResponse } from '../types';

const INITIAL_QUESTION_BANK: InterviewQuestionItem[] = [
  // ── DATA SCIENTIST & ML ENGINEER ──
  {
    id: 'ds-1',
    role: 'Data Scientist / ML Engineer',
    category: 'behavioral',
    difficulty: 'Mid-Level',
    question: 'Describe a machine learning or data science project where your model initially performed poorly. How did you diagnose and fix it?',
    hint: 'Use STAR: Explain the business metric, diagnosis (e.g. data leakage, class imbalance, overfitting), the action taken, and quantifiable improvement.',
    modelAnswer: 'In my previous role, our customer churn prediction model suffered from low recall (48%) on high-value accounts. I investigated the feature distributions and identified heavy class imbalance and data leakage in timestamp features. I engineered rolling-window temporal features, applied SMOTE with focal loss, and tuned a Gradient Boosting classifier. This increased recall to 86% and helped our retention team save $320K in quarterly revenue.',
    skills: ['Machine Learning', 'Diagnostics', 'SMOTE', 'Gradient Boosting'],
  },
  {
    id: 'ds-2',
    role: 'Data Scientist / ML Engineer',
    category: 'technical',
    difficulty: 'Mid-Level',
    question: 'How do you prevent overfitting in deep neural networks when working with limited training data?',
    hint: 'Mention data augmentation, dropout, weight decay (L2), transfer learning/pretrained weights, and early stopping.',
    modelAnswer: 'When training neural networks on scarce data, I apply a multi-layered regularization strategy: First, domain-specific data augmentation expands effective sample diversity. Second, I leverage transfer learning using pretrained foundation weights. Third, I incorporate spatial dropout (0.3–0.5) and AdamW with weight decay. Finally, I monitor validation loss with early stopping to prevent memorization.',
    skills: ['Deep Learning', 'Regularization', 'Transfer Learning'],
  },
  {
    id: 'ds-3',
    role: 'Data Scientist / ML Engineer',
    category: 'leadership',
    difficulty: 'Senior / Lead',
    question: 'How do you explain complex neural network predictions to non-technical executive stakeholders?',
    hint: 'Discuss SHAP / LIME explainability, translation to business KPIs, and visual decision trees rather than raw math.',
    modelAnswer: 'I bridge the technical gap by translating model weights into business outcomes. Rather than discussing loss gradients, I use SHAP value waterfall plots to show which business drivers influenced decisions. I focus on actionable metrics—such as cost per false positive—and present interactive scenario simulators that allow leaders to test strategic hypotheses.',
    skills: ['SHAP', 'Explainability', 'Stakeholder Communication'],
  },

  // ── SOFTWARE & FULL STACK DEVELOPER ──
  {
    id: 'swe-1',
    role: 'Software Engineer / Full Stack',
    category: 'behavioral',
    difficulty: 'Mid-Level',
    question: 'Tell me about a time you had to resolve a critical production bug under intense time pressure.',
    hint: 'Focus on triage, root-cause isolation, mitigation vs permanent fix, post-mortem, and automated regression guards.',
    modelAnswer: 'During a peak sales event, our payment microservice experienced thread pool exhaustion and dropped 18% of checkout requests. I immediately triaged APM traces, identified a deadlocked connection pool caused by unindexed DB queries, and deployed a connection limiter rollback within 12 minutes. I then indexed the query tables, implemented circuit breakers, and authored an incident post-mortem with automated load test gates.',
    skills: ['Incident Management', 'Database Optimization', 'Post-Mortem'],
  },
  {
    id: 'swe-2',
    role: 'Software Engineer / Full Stack',
    category: 'technical',
    difficulty: 'Senior / Lead',
    question: 'How do you design a scalable distributed system to handle 100,000 concurrent websocket connections?',
    hint: 'Discuss stateless edge nodes, Redis Pub/Sub or Kafka message brokers, horizontal autoscaling, and connection pooling.',
    modelAnswer: 'To handle 100k concurrent WebSocket connections, I decouple connection state from business logic. Stateless WebSocket gateway pods run behind an ALB using epoll/event-loop runtimes. Message broadcasting uses Redis Pub/Sub or Kafka clusters. Client heartbeats prevent zombie sockets, and horizontal pod autoscalers scale gateway nodes based on active socket count.',
    skills: ['Distributed Systems', 'WebSockets', 'Redis'],
  },

  // ── CLOUD & DEVOPS ENGINEER ──
  {
    id: 'devops-1',
    role: 'Cloud & DevOps Engineer',
    category: 'situational',
    difficulty: 'Mid-Level',
    question: 'Walk me through how you would architect a zero-downtime multi-region CI/CD deployment pipeline on AWS/Kubernetes.',
    hint: 'Explain GitOps (ArgoCD), canary/blue-green releases, automated rollback triggers, and cross-region traffic routing via Route 53.',
    modelAnswer: 'I architect multi-region CI/CD pipelines using GitOps with ArgoCD and Terraform. Code merges trigger automated container security scanning and unit testing. Deployments utilize Canary rollouts—routing 5% of live traffic to the new revision while monitoring Prometheus error rates and latency SLAs. If anomalies occur, automatic rollbacks fire in under 10 seconds without dropping user sessions.',
    skills: ['GitOps', 'ArgoCD', 'Kubernetes', 'Canary Rollout'],
  },

  // ── DATA ANALYST & BI SPECIALIST ──
  {
    id: 'da-1',
    role: 'Data Analyst & BI Specialist',
    category: 'behavioral',
    difficulty: 'Mid-Level',
    question: 'Describe an instance where your analytical insights directly influenced a major business decision.',
    hint: 'State the business problem, your data analysis approach (SQL/BI dashboards), the counter-intuitive insight, and resulting ROI.',
    modelAnswer: 'Our marketing team was spending 40% of their ad budget on an underperforming channel. I conducted cohort retention analysis across 500,000 users using SQL and PowerBI, discovering that organic referral users had a 3x higher 90-day LTV. I presented these findings to the VP of Growth, leading to a reallocation of $150K into customer referral programs, which increased overall CAC efficiency by 32%.',
    skills: ['SQL', 'PowerBI', 'Cohort Analysis'],
  },
];

const PRESET_ROLES = [
  'Data Scientist / ML Engineer',
  'Software Engineer / Full Stack',
  'Cloud & DevOps Engineer',
  'Data Analyst & BI Specialist',
  'Cybersecurity & SOC Analyst',
  'AI / Deep Learning Researcher',
  'Backend Engineer (Python / Go / Java)',
  'Frontend / UI Engineer (React / TypeScript)',
  'Product / Technical Project Manager',
];

export const InterviewCoachView: React.FC = () => {
  const [questionBank, setQuestionBank] = useState<InterviewQuestionItem[]>(INITIAL_QUESTION_BANK);
  const [selectedRole, setSelectedRole] = useState<string>('All Roles');
  const [isCustomRole, setIsCustomRole] = useState<boolean>(false);
  const [customRoleInput, setCustomRoleInput] = useState<string>('');
  const [activeCategory, setActiveCategory] = useState<string>('all');
  const [selectedDifficulty, setSelectedDifficulty] = useState<string>('all');
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState<number>(0);
  const [candidateResponse, setCandidateResponse] = useState<string>('');
  const [showModelAnswer, setShowModelAnswer] = useState<boolean>(false);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisResult, setAnalysisResult] = useState<SentimentResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // AI Generator Panel States
  const [showGeneratorPanel, setShowGeneratorPanel] = useState<boolean>(false);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [generateCount, setGenerateCount] = useState<number>(5);
  const [focusTopicsInput, setFocusTopicsInput] = useState<string>('');
  const [generatorSuccessMsg, setGeneratorSuccessMsg] = useState<string | null>(null);

  // AI Answer & Strategy Tips Generator State
  const [isGeneratingAnswerTips, setIsGeneratingAnswerTips] = useState<boolean>(false);
  const [answerTipsResult, setAnswerTipsResult] = useState<InterviewAnswerTipsResponse | null>(null);
  const [showAnswerTipsModal, setShowAnswerTipsModal] = useState<boolean>(false);
  const [copiedToast, setCopiedToast] = useState<boolean>(false);
  const [scorecardRecommendedAnswer, setScorecardRecommendedAnswer] = useState<InterviewAnswerTipsResponse | null>(null);

  // Practice Timer / Stopwatch
  const [timerSeconds, setTimerSeconds] = useState<number>(0);
  const [timerRunning, setTimerRunning] = useState<boolean>(false);
  const timerRef = useRef<any>(null);

  // Speech Recognition (Dictation)
  const [isListening, setIsListening] = useState<boolean>(false);
  const recognitionRef = useRef<any>(null);

  // Filtered Questions
  const filteredQuestions = questionBank.filter((q) => {
    const effectiveRole = isCustomRole && customRoleInput.trim() ? customRoleInput.trim() : selectedRole;
    const roleMatch = effectiveRole === 'All Roles' || q.role.toLowerCase() === effectiveRole.toLowerCase() || (isCustomRole && q.role.toLowerCase().includes(effectiveRole.toLowerCase()));
    const catMatch = activeCategory === 'all' || q.category === activeCategory;
    const diffMatch = selectedDifficulty === 'all' || q.difficulty === selectedDifficulty;
    return roleMatch && catMatch && diffMatch;
  });

  const activeQuestion = filteredQuestions[currentQuestionIndex] || filteredQuestions[0] || questionBank[0];

  // Timer Effect
  useEffect(() => {
    if (timerRunning) {
      timerRef.current = setInterval(() => {
        setTimerSeconds((prev) => prev + 1);
      }, 1000);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [timerRunning]);


  // Speech Recognition Setup
  useEffect(() => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';

      recognition.onresult = (event: any) => {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          transcript += event.results[i][0].transcript;
        }
        setCandidateResponse((prev) => prev + ' ' + transcript);
      };

      recognition.onerror = (e: any) => {
        console.warn('Speech recognition error', e);
        setIsListening(false);
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      recognitionRef.current = recognition;
    }
  }, []);

  const handleGenerateQuestions = async () => {
    const targetRole = isCustomRole && customRoleInput.trim()
      ? customRoleInput.trim()
      : (selectedRole === 'All Roles' ? 'Data Scientist / ML Engineer' : selectedRole);

    setIsGenerating(true);
    setErrorMessage(null);
    setGeneratorSuccessMsg(null);

    try {
      const res = await api.generateInterviewQuestions({
        role: targetRole,
        category: activeCategory === 'all' ? undefined : activeCategory,
        difficulty: selectedDifficulty === 'all' ? undefined : selectedDifficulty,
        count: generateCount,
        focus_topics: focusTopicsInput.trim() || undefined,
      });

      if (res && res.questions && res.questions.length > 0) {
        // Append new questions and select the first new one
        setQuestionBank((prev) => {
          const newQuestions = res.questions.filter(
            (nq) => !prev.some((pq) => pq.question.toLowerCase() === nq.question.toLowerCase())
          );
          return [...newQuestions, ...prev];
        });

        if (isCustomRole && customRoleInput.trim()) {
          setSelectedRole(customRoleInput.trim());
        } else if (selectedRole === 'All Roles') {
          setSelectedRole(targetRole);
        }

        setCurrentQuestionIndex(0);
        setCandidateResponse('');
        setAnalysisResult(null);
        setShowModelAnswer(false);
        setShowAnswerTipsModal(false);
        setAnswerTipsResult(null);
        setScorecardRecommendedAnswer(null);
        setGeneratorSuccessMsg(`Successfully generated ${res.questions.length} AI interview questions for "${targetRole}"!`);
        setShowGeneratorPanel(false);
      }
    } catch (err: any) {
      console.error('AI question generation error', err);
      setErrorMessage(err.response?.data?.detail || err.message || 'Failed to generate interview questions with AI');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleResetBank = () => {
    setQuestionBank(INITIAL_QUESTION_BANK);
    setSelectedRole('All Roles');
    setIsCustomRole(false);
    setCustomRoleInput('');
    setActiveCategory('all');
    setSelectedDifficulty('all');
    setCurrentQuestionIndex(0);
    setCandidateResponse('');
    setAnalysisResult(null);
    setShowModelAnswer(false);
    setShowAnswerTipsModal(false);
    setAnswerTipsResult(null);
    setScorecardRecommendedAnswer(null);
    setGeneratorSuccessMsg('Question bank reset to default curated templates.');
  };

  const toggleListening = () => {
    if (!recognitionRef.current) {
      alert('Speech recognition is not supported in this browser. Please type your response.');
      return;
    }

    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    } else {
      try {
        recognitionRef.current.start();
        setIsListening(true);
        if (!timerRunning) setTimerRunning(true);
      } catch (err) {
        console.error('Speech recognition error', err);
      }
    }
  };

  const handleNextQuestion = () => {
    const nextIdx = (currentQuestionIndex + 1) % filteredQuestions.length;
    setCurrentQuestionIndex(nextIdx);
    setShowModelAnswer(false);
    setShowAnswerTipsModal(false);
    setAnswerTipsResult(null);
    setScorecardRecommendedAnswer(null);
    setCandidateResponse('');
    setAnalysisResult(null);
    setTimerSeconds(0);
    setTimerRunning(false);
  };

  const handleRandomQuestion = () => {
    const randomIdx = Math.floor(Math.random() * filteredQuestions.length);
    setCurrentQuestionIndex(randomIdx);
    setShowModelAnswer(false);
    setShowAnswerTipsModal(false);
    setAnswerTipsResult(null);
    setScorecardRecommendedAnswer(null);
    setCandidateResponse('');
    setAnalysisResult(null);
    setTimerSeconds(0);
    setTimerRunning(false);
  };

  const handleGetAnswerAndTips = async () => {
    if (!activeQuestion) return;
    setIsGeneratingAnswerTips(true);
    setShowAnswerTipsModal(true);
    setErrorMessage(null);

    try {
      const res = await api.getInterviewAnswerAndTips({
        question: activeQuestion.question,
        role: activeQuestion.role,
        category: activeQuestion.category,
        difficulty: activeQuestion.difficulty,
        hint: activeQuestion.hint,
        model_answer: activeQuestion.modelAnswer,
        user_response: candidateResponse.trim() || undefined,
      });
      setAnswerTipsResult(res);
    } catch (err: any) {
      console.warn('Failed to get answer and tips from backend, applying fallback', err);
      setAnswerTipsResult({
        success: true,
        question: activeQuestion.question,
        role: activeQuestion.role,
        category: activeQuestion.category,
        difficulty: activeQuestion.difficulty || 'Mid-Level',
        framework: activeQuestion.category === 'technical' ? 'Technical Architecture & Trade-off Analysis' : 'STAR Method Framework',
        hint: activeQuestion.hint,
        recommended_answer: activeQuestion.modelAnswer || 'Structure your answer using Situation, Task, Action, and quantifiable Result.',
        key_tips: [
          'Break down your response systematically: 15% Situation, 15% Task, 50% Action, and 20% Result.',
          'Emphasize your individual technical ownership and engineering choices.',
          'Quantify your business impact with measurable numbers, throughput improvements, or cost savings.',
          'Mention automated tests, regression guards, or post-incident safeguards.'
        ],
        common_pitfalls: [
          'Speaking in broad generalities without anchoring your answer in a specific real-world project.',
          'Focusing only on what the collective team did without detailing your exact personal contribution.',
          'Forgetting to state the quantifiable business result or takeaway.'
        ],
        essential_keywords: activeQuestion.skills || ['STAR Framework', 'Problem Solving', 'Root Cause Analysis', 'Engineering Rigor'],
      });
    } finally {
      setIsGeneratingAnswerTips(false);
    }
  };

  const handleAnalyze = async () => {
    if (!candidateResponse.trim()) return;
    setIsAnalyzing(true);
    setErrorMessage(null);
    setTimerRunning(false);

    try {
      // 1. Perform Neural Sentiment & STAR Tone Analysis
      const res = await api.analyzeSentimentTone(candidateResponse);
      setAnalysisResult(res);

      // 2. Concurrently generate AI Recommended Answer & Comparative Feedback for this question
      try {
        const rec = await api.getInterviewAnswerAndTips({
          question: activeQuestion.question,
          role: activeQuestion.role,
          category: activeQuestion.category,
          difficulty: activeQuestion.difficulty,
          hint: activeQuestion.hint,
          model_answer: activeQuestion.modelAnswer,
          user_response: candidateResponse.trim(),
        });
        setScorecardRecommendedAnswer(rec);
      } catch (recErr) {
        console.warn('Could not fetch recommended answer comparison', recErr);
        if (activeQuestion.modelAnswer) {
          setScorecardRecommendedAnswer({
            success: true,
            question: activeQuestion.question,
            role: activeQuestion.role,
            category: activeQuestion.category,
            difficulty: activeQuestion.difficulty || 'Mid-Level',
            framework: 'STAR Method Framework',
            hint: activeQuestion.hint,
            recommended_answer: activeQuestion.modelAnswer,
            key_tips: ['Include explicit metrics and STAR sequence.'],
            common_pitfalls: ['Vague generalities.'],
            essential_keywords: activeQuestion.skills || ['STAR Framework'],
            strengths: ['Captured the essence of the prompt.'],
            improvement_areas: ['Ensure measurable business impact in the result.'],
          });
        }
      }
    } catch (err: any) {
      console.error('Interview analysis error', err);
      setErrorMessage(err.response?.data?.detail || err.message || 'Failed to complete AI interview evaluation');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const formatTimer = (sec: number) => {
    const mins = Math.floor(sec / 60);
    const s = sec % 60;
    return `${mins}:${s < 10 ? '0' : ''}${s}`;
  };

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Page Header with Action Button */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
            <span className="badge badge-primary">
              <Award size={12} />
              AI Mock Interview & STAR Assessment Studio
            </span>
            <span className="badge badge-success" style={{ background: 'rgba(16, 185, 129, 0.15)', color: '#34D399', border: '1px solid rgba(16, 185, 129, 0.3)' }}>
              <Sparkles size={11} />
              AI Question Synthesizer Active
            </span>
          </div>
          <h1 style={{ fontSize: '28px', fontWeight: 800 }}>
            Interview <span className="gradient-text">Preparation Coach</span>
          </h1>
          <p style={{ fontSize: '14px', color: 'var(--text-muted)' }}>
            Practice real-world technical and STAR behavioral interview questions with real-time AI generation, neural tone diagnostics, and expert scoring.
          </p>
        </div>

        {/* Top Header Buttons */}
        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            className="btn btn-primary"
            onClick={() => setShowGeneratorPanel(!showGeneratorPanel)}
            style={{
              padding: '10px 18px',
              fontSize: '13px',
              fontWeight: 700,
              boxShadow: '0 0 15px rgba(99, 102, 241, 0.4)',
              background: showGeneratorPanel ? 'linear-gradient(135deg, #4F46E5, #06B6D4)' : undefined,
            }}
          >
            <Sparkles size={16} />
            <span>{showGeneratorPanel ? 'Close AI Studio' : '✨ Generate AI Questions'}</span>
          </button>

          <button
            onClick={handleResetBank}
            style={{
              padding: '10px 14px',
              borderRadius: '8px',
              background: 'rgba(30, 41, 59, 0.6)',
              border: '1px solid var(--border-color)',
              color: 'var(--text-muted)',
              fontSize: '13px',
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
            }}
            title="Reset question bank to original templates"
          >
            <RefreshCw size={14} />
            <span>Reset Bank</span>
          </button>
        </div>
      </div>

      {/* Success Notification Alert */}
      {generatorSuccessMsg && (
        <div style={{
          padding: '12px 18px',
          background: 'rgba(16, 185, 129, 0.15)',
          border: '1px solid rgba(16, 185, 129, 0.4)',
          borderRadius: '8px',
          color: '#34D399',
          fontSize: '13px',
          fontWeight: 600,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          animation: 'fadeIn 0.2s ease',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <CheckCircle2 size={16} />
            <span>{generatorSuccessMsg}</span>
          </div>
          <button
            onClick={() => setGeneratorSuccessMsg(null)}
            style={{ background: 'none', border: 'none', color: '#34D399', cursor: 'pointer', fontSize: '14px', fontWeight: 800 }}
          >
            ✕
          </button>
        </div>
      )}

      {/* ── EXPANDABLE AI QUESTION GENERATOR PANEL ── */}
      {showGeneratorPanel && (
        <div className="glass-card" style={{
          padding: '20px 24px',
          background: 'linear-gradient(135deg, rgba(30, 27, 75, 0.8), rgba(15, 23, 42, 0.9))',
          border: '1px solid rgba(99, 102, 241, 0.4)',
          borderRadius: '12px',
          display: 'flex',
          flexDirection: 'column',
          gap: '16px',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
          animation: 'fadeIn 0.25s ease',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ padding: '6px', borderRadius: '8px', background: 'rgba(99, 102, 241, 0.2)', color: '#818CF8' }}>
                <Sparkles size={18} />
              </div>
              <div>
                <h3 style={{ fontSize: '16px', fontWeight: 800, margin: 0 }}>AI Question Synthesizer</h3>
                <p style={{ fontSize: '12px', color: 'var(--text-muted)', margin: 0 }}>
                  Generate realistic role-specific interview questions, STAR strategy tips, and expert model answers.
                </p>
              </div>
            </div>
            <span className="badge badge-primary" style={{ fontSize: '11px' }}>
              Dynamic Role Engine
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '14px' }}>
            {/* Target Role Selector */}
            <div>
              <label style={{ fontSize: '12px', fontWeight: 700, color: '#CBD5E1', marginBottom: '6px', display: 'block' }}>
                🎯 Target Career Role:
              </label>
              <select
                className="form-select"
                value={isCustomRole ? 'CUSTOM' : selectedRole}
                onChange={(e) => {
                  if (e.target.value === 'CUSTOM') {
                    setIsCustomRole(true);
                  } else {
                    setIsCustomRole(false);
                    setSelectedRole(e.target.value);
                  }
                }}
                style={{ width: '100%', padding: '8px 12px', fontSize: '13px' }}
              >
                <option value="All Roles">-- Select or Choose Role --</option>
                {PRESET_ROLES.map((r) => (
                  <option key={r} value={r}>{r}</option>
                ))}
                <option value="CUSTOM">➕ Write Custom Role...</option>
              </select>
            </div>

            {/* Custom Role Text Input (if selected) */}
            {isCustomRole && (
              <div>
                <label style={{ fontSize: '12px', fontWeight: 700, color: '#38BDF8', marginBottom: '6px', display: 'block' }}>
                  ✍️ Enter Custom Role Title:
                </label>
                <input
                  type="text"
                  className="form-input"
                  placeholder="e.g. Flutter Mobile Dev, Rust Core Eng..."
                  value={customRoleInput}
                  onChange={(e) => setCustomRoleInput(e.target.value)}
                  style={{ width: '100%', padding: '8px 12px', fontSize: '13px' }}
                />
              </div>
            )}

            {/* Seniority / Difficulty */}
            <div>
              <label style={{ fontSize: '12px', fontWeight: 700, color: '#CBD5E1', marginBottom: '6px', display: 'block' }}>
                📊 Seniority Level:
              </label>
              <select
                className="form-select"
                value={selectedDifficulty}
                onChange={(e) => setSelectedDifficulty(e.target.value)}
                style={{ width: '100%', padding: '8px 12px', fontSize: '13px' }}
              >
                <option value="all">All Difficulty Levels</option>
                <option value="Entry-Level">Entry-Level / Junior</option>
                <option value="Mid-Level">Mid-Level Engineer</option>
                <option value="Senior / Lead">Senior / Staff / Lead</option>
              </select>
            </div>

            {/* Count of Questions */}
            <div>
              <label style={{ fontSize: '12px', fontWeight: 700, color: '#CBD5E1', marginBottom: '6px', display: 'block' }}>
                🔢 Number of Questions:
              </label>
              <select
                className="form-select"
                value={generateCount}
                onChange={(e) => setGenerateCount(Number(e.target.value))}
                style={{ width: '100%', padding: '8px 12px', fontSize: '13px' }}
              >
                <option value={3}>3 Questions (Quick Practice)</option>
                <option value={5}>5 Questions (Recommended)</option>
                <option value={8}>8 Questions (Full Mock)</option>
                <option value={10}>10 Questions (Deep Dive)</option>
              </select>
            </div>

            {/* Optional Specific Focus Topic */}
            <div style={{ gridColumn: 'span 2' }}>
              <label style={{ fontSize: '12px', fontWeight: 700, color: '#CBD5E1', marginBottom: '6px', display: 'block' }}>
                🔍 Specific Tech Stack or Subtopic Focus (Optional):
              </label>
              <input
                type="text"
                className="form-input"
                placeholder="e.g. System Design, Transformers & Attention, Kubernetes & Terraform, SQL Optimization, Conflict Resolution..."
                value={focusTopicsInput}
                onChange={(e) => setFocusTopicsInput(e.target.value)}
                style={{ width: '100%', padding: '8px 12px', fontSize: '13px' }}
              />
            </div>
          </div>

          {/* Action Generate Button */}
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '6px' }}>
            <button
              className="btn btn-secondary"
              onClick={() => setShowGeneratorPanel(false)}
              style={{ padding: '8px 16px', fontSize: '13px' }}
            >
              Cancel
            </button>
            <button
              className="btn btn-primary"
              onClick={handleGenerateQuestions}
              disabled={isGenerating || (isCustomRole && !customRoleInput.trim())}
              style={{
                padding: '10px 24px',
                fontSize: '13px',
                fontWeight: 700,
                background: 'linear-gradient(135deg, #6366F1, #38BDF8)',
              }}
            >
              {isGenerating ? (
                <>
                  <RotateCcw size={15} className="animate-spin" />
                  <span>Synthesizing Questions with AI...</span>
                </>
              ) : (
                <>
                  <Sparkles size={15} />
                  <span>Generate {generateCount} Tailored Questions</span>
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* Role & Category Filter Controls */}
      <div style={{
        display: 'flex',
        flexWrap: 'wrap',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '12px',
        padding: '14px 18px',
        background: 'rgba(15, 23, 42, 0.6)',
        borderRadius: 'var(--radius-sm)',
        border: '1px solid var(--border-color)',
      }}>
        {/* Role Picker */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-muted)' }}>Target Role:</span>
          <select
            className="form-select"
            value={isCustomRole ? 'CUSTOM' : selectedRole}
            onChange={(e) => {
              if (e.target.value === 'CUSTOM') {
                setIsCustomRole(true);
              } else {
                setIsCustomRole(false);
                setSelectedRole(e.target.value);
                setCurrentQuestionIndex(0);
              }
            }}
            style={{ padding: '6px 12px', fontSize: '13px', minWidth: '180px' }}
          >
            <option value="All Roles">All Roles ({questionBank.length} questions)</option>
            {PRESET_ROLES.map((r) => (
              <option key={r} value={r}>{r}</option>
            ))}
            <option value="CUSTOM">➕ Custom Role...</option>
          </select>

          {isCustomRole && (
            <input
              type="text"
              className="form-input"
              placeholder="Type role..."
              value={customRoleInput}
              onChange={(e) => {
                setCustomRoleInput(e.target.value);
                setCurrentQuestionIndex(0);
              }}
              style={{ padding: '6px 12px', fontSize: '13px', maxWidth: '160px' }}
            />
          )}

          {/* Seniority Filter */}
          <select
            className="form-select"
            value={selectedDifficulty}
            onChange={(e) => {
              setSelectedDifficulty(e.target.value);
              setCurrentQuestionIndex(0);
            }}
            style={{ padding: '6px 10px', fontSize: '12px' }}
          >
            <option value="all">All Levels</option>
            <option value="Entry-Level">Entry-Level</option>
            <option value="Mid-Level">Mid-Level</option>
            <option value="Senior / Lead">Senior / Lead</option>
          </select>
        </div>

        {/* Category Filter Buttons */}
        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
          {[
            { id: 'all', label: 'All Categories' },
            { id: 'behavioral', label: '⭐ Behavioral' },
            { id: 'technical', label: '⚡ Technical' },
            { id: 'leadership', label: '🤝 Leadership' },
            { id: 'situational', label: '🛡️ Situational' },
          ].map((cat) => (
            <button
              key={cat.id}
              onClick={() => {
                setActiveCategory(cat.id);
                setCurrentQuestionIndex(0);
              }}
              style={{
                padding: '6px 12px',
                borderRadius: '6px',
                border: 'none',
                background: activeCategory === cat.id ? 'var(--primary)' : 'rgba(30, 41, 59, 0.6)',
                color: activeCategory === cat.id ? '#FFF' : 'var(--text-muted)',
                fontSize: '12px',
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              {cat.label}
            </button>
          ))}
        </div>
      </div>

      {/* Question Switcher Pill Bar */}
      {filteredQuestions.length > 1 && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          overflowX: 'auto',
          padding: '8px 12px',
          background: 'rgba(15, 23, 42, 0.4)',
          borderRadius: '8px',
          border: '1px solid var(--border-color)',
        }}>
          <span style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', marginRight: '4px', whiteSpace: 'nowrap' }}>
            Questions ({filteredQuestions.length}):
          </span>
          {filteredQuestions.map((q, idx) => (
            <button
              key={q.id || idx}
              onClick={() => {
                setCurrentQuestionIndex(idx);
                setShowModelAnswer(false);
                setCandidateResponse('');
                setAnalysisResult(null);
              }}
              style={{
                padding: '4px 10px',
                borderRadius: '6px',
                border: idx === currentQuestionIndex ? '1px solid var(--primary)' : '1px solid transparent',
                background: idx === currentQuestionIndex ? 'rgba(99, 102, 241, 0.25)' : 'rgba(30, 41, 59, 0.5)',
                color: idx === currentQuestionIndex ? '#FFF' : 'var(--text-muted)',
                fontSize: '11px',
                fontWeight: 700,
                cursor: 'pointer',
                whiteSpace: 'nowrap',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '4px',
              }}
            >
              <span>Q{idx + 1}</span>
              {q.isAiGenerated && <span style={{ fontSize: '9px', color: '#38BDF8' }}>✦</span>}
            </button>
          ))}
        </div>
      )}

      {/* Main 2-Column Split: Practice Room (Left) & AI Scorecard (Right) */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 1.2fr', gap: '24px' }}>
        
        {/* ── LEFT COLUMN: Interactive Practice Room ── */}
        <div className="glass-card" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '18px' }}>
          {/* Active Question Banner */}
          <div style={{
            padding: '18px',
            background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.18), rgba(56, 189, 248, 0.1))',
            borderRadius: 'var(--radius-sm)',
            border: '1px solid rgba(99, 102, 241, 0.35)',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px', flexWrap: 'wrap', gap: '6px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span className="badge badge-primary" style={{ fontSize: '11px' }}>
                  {activeQuestion.role}
                </span>
                {activeQuestion.difficulty && (
                  <span className="badge" style={{ fontSize: '10px', background: 'rgba(51, 65, 85, 0.6)', color: '#CBD5E1' }}>
                    {activeQuestion.difficulty}
                  </span>
                )}
                {activeQuestion.isAiGenerated && (
                  <span className="badge" style={{ fontSize: '10px', background: 'rgba(56, 189, 248, 0.2)', color: '#38BDF8', border: '1px solid rgba(56, 189, 248, 0.4)' }}>
                    🤖 AI Generated
                  </span>
                )}
              </div>
              <span style={{ fontSize: '12px', color: '#38BDF8', fontWeight: 600 }}>
                Question {currentQuestionIndex + 1} of {filteredQuestions.length}
              </span>
            </div>

            <h3 style={{ fontSize: '17px', fontWeight: 800, color: '#FFF', lineHeight: 1.4, margin: '6px 0 10px' }}>
              "{activeQuestion.question}"
            </h3>

            <div style={{ fontSize: '12px', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Lightbulb size={14} color="#F59E0B" />
              <span><strong>Strategy Hint:</strong> {activeQuestion.hint}</span>
            </div>
          </div>

          {/* ── DON'T KNOW THE ANSWER? AI HELPER BUTTON & EXPANDABLE GUIDE ── */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '10px',
            padding: '10px 14px',
            background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(236, 72, 153, 0.1))',
            border: '1px solid rgba(236, 72, 153, 0.35)',
            borderRadius: '8px',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <HelpCircle size={16} color="#EC4899" />
              <div>
                <span style={{ fontSize: '12.5px', fontWeight: 700, color: '#F1F5F9' }}>
                  Unsure how to answer this question?
                </span>
                <p style={{ margin: 0, fontSize: '11px', color: 'var(--text-muted)' }}>
                  Generate an AI recommended answer, STAR strategy tips & key phrases.
                </p>
              </div>
            </div>
            <button
              onClick={() => {
                if (showAnswerTipsModal && answerTipsResult) {
                  setShowAnswerTipsModal(false);
                } else {
                  handleGetAnswerAndTips();
                }
              }}
              disabled={isGeneratingAnswerTips}
              style={{
                padding: '7px 14px',
                borderRadius: '6px',
                background: showAnswerTipsModal && answerTipsResult ? 'rgba(30, 41, 59, 0.8)' : 'linear-gradient(135deg, #EC4899, #8B5CF6)',
                border: showAnswerTipsModal && answerTipsResult ? '1px solid var(--border-color)' : 'none',
                color: '#FFF',
                fontSize: '12px',
                fontWeight: 700,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                whiteSpace: 'nowrap',
                boxShadow: showAnswerTipsModal && answerTipsResult ? 'none' : '0 2px 12px rgba(236, 72, 153, 0.35)',
                transition: 'all 0.2s ease',
              }}
            >
              {isGeneratingAnswerTips ? (
                <>
                  <RotateCcw size={13} className="animate-spin" />
                  <span>Synthesizing...</span>
                </>
              ) : showAnswerTipsModal && answerTipsResult ? (
                <>
                  <span>Hide AI Guide</span>
                </>
              ) : (
                <>
                  <Sparkles size={13} />
                  <span>💡 Get AI Answer & Tips</span>
                </>
              )}
            </button>
          </div>

          {/* Expandable AI Answer & Tips Coaching Panel */}
          {showAnswerTipsModal && (
            <div style={{
              padding: '16px 18px',
              background: 'linear-gradient(135deg, rgba(24, 24, 45, 0.95), rgba(15, 23, 42, 0.95))',
              border: '1px solid rgba(236, 72, 153, 0.45)',
              borderRadius: '10px',
              display: 'flex',
              flexDirection: 'column',
              gap: '12px',
              boxShadow: '0 8px 30px rgba(0, 0, 0, 0.4)',
              animation: 'fadeIn 0.2s ease',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div style={{ padding: '5px', borderRadius: '6px', background: 'rgba(236, 72, 153, 0.2)', color: '#F472B6' }}>
                    <Sparkles size={15} />
                  </div>
                  <div>
                    <h4 style={{ margin: 0, fontSize: '13.5px', fontWeight: 800, color: '#FFF' }}>
                      AI Gold-Standard Answer & Strategy Tips
                    </h4>
                    <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                      Framework: {answerTipsResult?.framework || 'STAR Method'}
                    </span>
                  </div>
                </div>
                <button
                  onClick={() => setShowAnswerTipsModal(false)}
                  style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '15px' }}
                >
                  ✕
                </button>
              </div>

              {isGeneratingAnswerTips ? (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '24px', gap: '10px', color: 'var(--text-muted)' }}>
                  <RotateCcw size={18} className="animate-spin" color="#EC4899" />
                  <span style={{ fontSize: '12.5px' }}>Synthesizing expert model answer and coaching tips...</span>
                </div>
              ) : answerTipsResult && (
                <>
                  {/* Recommended Answer Section */}
                  <div style={{
                    padding: '12px 14px',
                    background: 'rgba(16, 185, 129, 0.08)',
                    border: '1px solid rgba(16, 185, 129, 0.3)',
                    borderRadius: '8px',
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                      <span style={{ fontSize: '12px', fontWeight: 700, color: '#34D399', display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <Award size={14} />
                        <span>Recommended Gold-Standard Response:</span>
                      </span>
                      <button
                        onClick={() => {
                          setCandidateResponse(answerTipsResult.recommended_answer);
                          setCopiedToast(true);
                          setTimeout(() => setCopiedToast(false), 2000);
                        }}
                        style={{
                          padding: '3px 8px',
                          borderRadius: '6px',
                          background: 'rgba(16, 185, 129, 0.2)',
                          border: '1px solid #10B981',
                          color: '#34D399',
                          fontSize: '11px',
                          fontWeight: 600,
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '4px',
                        }}
                      >
                        {copiedToast ? <Check size={11} /> : <Copy size={11} />}
                        <span>{copiedToast ? 'Inserted!' : 'Insert to Editor'}</span>
                      </button>
                    </div>
                    <p style={{ fontSize: '12.5px', color: '#E2E8F0', lineHeight: 1.5, margin: 0 }}>
                      {answerTipsResult.recommended_answer}
                    </p>
                  </div>

                  {/* Strategy Tips Section */}
                  {answerTipsResult.key_tips && answerTipsResult.key_tips.length > 0 && (
                    <div style={{
                      padding: '10px 12px',
                      background: 'rgba(99, 102, 241, 0.08)',
                      border: '1px solid rgba(99, 102, 241, 0.25)',
                      borderRadius: '8px',
                    }}>
                      <div style={{ fontSize: '11.5px', fontWeight: 700, color: '#818CF8', marginBottom: '5px', display: 'flex', alignItems: 'center', gap: '5px' }}>
                        <Lightbulb size={13} />
                        <span>Key Strategic Tips for This Question:</span>
                      </div>
                      <ul style={{ margin: 0, paddingLeft: '16px', fontSize: '11.5px', color: '#CBD5E1', display: 'flex', flexDirection: 'column', gap: '3px' }}>
                        {answerTipsResult.key_tips.map((tip, idx) => (
                          <li key={idx}>{tip}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Keywords & Common Pitfalls Grid */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '8px' }}>
                    {/* Keywords */}
                    {answerTipsResult.essential_keywords && answerTipsResult.essential_keywords.length > 0 && (
                      <div style={{
                        padding: '8px 10px',
                        background: 'rgba(30, 41, 59, 0.5)',
                        border: '1px solid var(--border-color)',
                        borderRadius: '6px',
                      }}>
                        <div style={{ fontSize: '11px', fontWeight: 700, color: '#38BDF8', marginBottom: '4px' }}>
                          🔑 Keywords & Skills:
                        </div>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                          {answerTipsResult.essential_keywords.map((kw, idx) => (
                            <span key={idx} className="badge badge-primary" style={{ fontSize: '10px', padding: '1px 5px' }}>
                              {kw}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Pitfalls */}
                    {answerTipsResult.common_pitfalls && answerTipsResult.common_pitfalls.length > 0 && (
                      <div style={{
                        padding: '8px 10px',
                        background: 'rgba(239, 68, 68, 0.08)',
                        border: '1px solid rgba(239, 68, 68, 0.25)',
                        borderRadius: '6px',
                      }}>
                        <div style={{ fontSize: '11px', fontWeight: 700, color: '#FCA5A5', marginBottom: '4px' }}>
                          ⚠️ Pitfalls to Avoid:
                        </div>
                        <ul style={{ margin: 0, paddingLeft: '14px', fontSize: '11px', color: '#CBD5E1', display: 'flex', flexDirection: 'column', gap: '2px' }}>
                          {answerTipsResult.common_pitfalls.map((pit, idx) => (
                            <li key={idx}>{pit}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          )}

          {/* Practice Tools Toolbar: Timer & Dictation & Sample Toggle */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0 4px', flexWrap: 'wrap', gap: '8px' }}>
            {/* Stopwatch */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '4px 10px',
                borderRadius: '6px',
                background: timerRunning ? 'rgba(239, 68, 68, 0.15)' : 'rgba(30, 41, 59, 0.6)',
                border: timerRunning ? '1px solid #EF4444' : '1px solid var(--border-color)',
                color: timerRunning ? '#FCA5A5' : 'var(--text-muted)',
                fontSize: '13px',
                fontWeight: 700,
                fontFamily: 'var(--font-mono)',
              }}>
                <span>⏱️ {formatTimer(timerSeconds)}</span>
              </div>
              <button
                onClick={() => setTimerRunning(!timerRunning)}
                style={{
                  padding: '5px 10px',
                  borderRadius: '6px',
                  background: 'rgba(51, 65, 85, 0.5)',
                  border: '1px solid var(--border-color)',
                  color: '#FFF',
                  fontSize: '11px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                }}
              >
                {timerRunning ? 'Pause Timer' : 'Start Timer'}
              </button>
            </div>

            {/* Actions: Dictate & Next */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <button
                onClick={toggleListening}
                style={{
                  padding: '6px 12px',
                  borderRadius: '6px',
                  background: isListening ? '#EF4444' : 'rgba(99, 102, 241, 0.2)',
                  border: isListening ? '1px solid #DC2626' : '1px solid #6366F1',
                  color: '#FFF',
                  fontSize: '12px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  animation: isListening ? 'pulseGlow 1.5s infinite' : 'none',
                }}
              >
                {isListening ? <MicOff size={14} /> : <Mic size={14} />}
                <span>{isListening ? 'Stop Recording' : 'Voice Dictate'}</span>
              </button>

              <button
                onClick={handleRandomQuestion}
                style={{
                  padding: '6px 12px',
                  borderRadius: '6px',
                  background: 'rgba(30, 41, 59, 0.6)',
                  border: '1px solid var(--border-color)',
                  color: '#CBD5E1',
                  fontSize: '12px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                }}
                title="Random Question"
              >
                <span>🎲 Random</span>
              </button>

              <button
                onClick={handleNextQuestion}
                style={{
                  padding: '6px 12px',
                  borderRadius: '6px',
                  background: 'rgba(30, 41, 59, 0.8)',
                  border: '1px solid var(--border-color)',
                  color: '#E2E8F0',
                  fontSize: '12px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                }}
              >
                <span>Next Question</span>
                <ArrowRight size={12} />
              </button>
            </div>
          </div>

          {/* Response Textarea */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px', fontSize: '12px', color: 'var(--text-muted)' }}>
              <span>Your Response (Type or Dictate using the mic above):</span>
              <span>{candidateResponse.trim().split(/\s+/).filter(Boolean).length} words</span>
            </div>
            <textarea
              className="form-textarea"
              rows={7}
              value={candidateResponse}
              onChange={(e) => setCandidateResponse(e.target.value)}
              placeholder="Structure your answer using STAR: Situation (context), Task (objective), Action (what you specifically did), Result (quantified business impact)..."
              style={{ width: '100%', fontSize: '13.5px', lineHeight: 1.5 }}
            />
          </div>

          {/* Quick Preload Sample Response or Model Answer */}
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px' }}>
            <button
              onClick={() => {
                setCandidateResponse(activeQuestion.modelAnswer);
              }}
              style={{
                padding: '6px 12px',
                borderRadius: '6px',
                background: 'rgba(30, 41, 59, 0.6)',
                border: '1px solid var(--border-color)',
                color: '#38BDF8',
                fontSize: '11px',
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              📥 Paste Candidate STAR Sample
            </button>

            <button
              onClick={() => setShowModelAnswer(!showModelAnswer)}
              style={{
                padding: '6px 12px',
                borderRadius: '6px',
                background: showModelAnswer ? 'rgba(16, 185, 129, 0.2)' : 'rgba(30, 41, 59, 0.6)',
                border: showModelAnswer ? '1px solid #10B981' : '1px solid var(--border-color)',
                color: showModelAnswer ? '#34D399' : 'var(--text-muted)',
                fontSize: '11px',
                fontWeight: 600,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
              }}
            >
              <BookOpen size={13} />
              <span>{showModelAnswer ? 'Hide Gold-Standard Answer' : '💡 View Gold-Standard Answer'}</span>
            </button>
          </div>

          {/* Gold-Standard Answer Reveal */}
          {showModelAnswer && (
            <div style={{
              padding: '16px',
              background: 'rgba(16, 185, 129, 0.1)',
              border: '1px solid rgba(16, 185, 129, 0.3)',
              borderRadius: 'var(--radius-sm)',
              animation: 'fadeIn 0.2s ease',
            }}>
              <div style={{ fontSize: '12px', fontWeight: 700, color: '#34D399', marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Award size={14} />
                <span>EXPERT GOLD-STANDARD STAR RESPONSE:</span>
              </div>
              <p style={{ fontSize: '13px', color: '#E2E8F0', lineHeight: 1.5, margin: 0 }}>
                {activeQuestion.modelAnswer}
              </p>
            </div>
          )}

          {/* Analyze Button */}
          <button
            className="btn btn-primary"
            style={{ width: '100%', padding: '14px', fontSize: '15px', fontWeight: 700 }}
            onClick={handleAnalyze}
            disabled={isAnalyzing || !candidateResponse.trim()}
          >
            <Sparkles size={18} />
            <span>{isAnalyzing ? 'Evaluating STAR & Generating Recommended Answer...' : '🎙️ Evaluate Response & Generate Scorecard'}</span>
          </button>
        </div>

        {/* ── RIGHT COLUMN: AI Coaching Scorecard ── */}
        <div className="glass-card" style={{ padding: '24px', display: 'flex', flexDirection: 'column' }}>
          <h3 style={{ fontSize: '18px', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <TrendingUp size={20} color="#34D399" />
            <span>AI Interview Scorecard & Feedback</span>
          </h3>

          {isAnalyzing ? (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '50px', gap: '14px', color: 'var(--text-muted)' }}>
              <RotateCcw size={32} className="animate-spin" color="#818CF8" />
              <span>Analyzing STAR completeness, action verbs, and recurrent sentiment...</span>
            </div>
          ) : errorMessage ? (
            <div style={{ padding: '20px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: '8px', textAlign: 'center' }}>
              <AlertTriangle size={24} color="#EF4444" style={{ margin: '0 auto 8px' }} />
              <div style={{ color: '#FCA5A5', fontWeight: 600, fontSize: '14px' }}>{errorMessage}</div>
              <button
                className="btn btn-secondary"
                style={{ marginTop: '12px', fontSize: '12px' }}
                onClick={handleAnalyze}
              >
                Retry Evaluation
              </button>
            </div>
          ) : analysisResult ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              
              {/* 1. Overall Score & Tone Badge Banner */}
              {analysisResult.coach && (
                <div style={{
                  padding: '18px 20px',
                  background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(16, 185, 129, 0.15))',
                  border: '1px solid rgba(99, 102, 241, 0.4)',
                  borderRadius: 'var(--radius-sm)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                }}>
                  <div>
                    <div style={{ fontSize: '11px', color: '#38BDF8', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                      INTERVIEW READINESS SCORE
                    </div>
                    <div style={{ fontSize: '22px', fontWeight: 800, color: analysisResult.coach.tone_color || '#10B981', marginTop: '2px' }}>
                      {analysisResult.coach.tone_label || 'Executive & High Impact'}
                    </div>
                    <div style={{ fontSize: '13px', color: 'var(--text-muted)', marginTop: '2px' }}>
                      {analysisResult.coach.tone_badge}
                    </div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: '32px', fontWeight: 900, color: analysisResult.coach.tone_color || '#10B981' }}>
                      {analysisResult.coach.impact_score ?? 80}
                      <span style={{ fontSize: '16px', color: 'var(--text-muted)' }}>/100</span>
                    </div>
                  </div>
                </div>
              )}

              {/* 2. AI RECOMMENDED ANSWER & COMPARATIVE ANALYSIS */}
              {scorecardRecommendedAnswer && (
                <div style={{
                  padding: '16px 18px',
                  background: 'linear-gradient(135deg, rgba(30, 27, 75, 0.7), rgba(15, 23, 42, 0.85))',
                  border: '1px solid rgba(139, 92, 246, 0.4)',
                  borderRadius: 'var(--radius-sm)',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '12px',
                  animation: 'fadeIn 0.25s ease',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ fontSize: '12px', fontWeight: 800, color: '#C084FC', display: 'flex', alignItems: 'center', gap: '6px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                      <Sparkles size={15} color="#C084FC" />
                      <span>AI Recommended Ideal Answer</span>
                    </div>
                    <span className="badge badge-primary" style={{ fontSize: '10px' }}>
                      Gold Standard Model
                    </span>
                  </div>

                  {/* The Recommended Answer Text */}
                  <div style={{
                    padding: '12px 14px',
                    background: 'rgba(16, 185, 129, 0.08)',
                    border: '1px solid rgba(16, 185, 129, 0.25)',
                    borderRadius: '6px',
                    fontSize: '13px',
                    color: '#E2E8F0',
                    lineHeight: 1.5,
                  }}>
                    {scorecardRecommendedAnswer.recommended_answer}
                  </div>

                  {/* Comparative Insights: Strengths & Improvement Tips */}
                  {scorecardRecommendedAnswer.strengths && scorecardRecommendedAnswer.strengths.length > 0 && (
                    <div style={{ fontSize: '12px', display: 'flex', flexDirection: 'column', gap: '3px' }}>
                      <span style={{ fontWeight: 700, color: '#34D399' }}>✓ What You Handled Well:</span>
                      <ul style={{ margin: 0, paddingLeft: '18px', color: '#CBD5E1', display: 'flex', flexDirection: 'column', gap: '2px' }}>
                        {scorecardRecommendedAnswer.strengths.map((s, idx) => (
                          <li key={idx}>{s}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {scorecardRecommendedAnswer.improvement_areas && scorecardRecommendedAnswer.improvement_areas.length > 0 && (
                    <div style={{ fontSize: '12px', display: 'flex', flexDirection: 'column', gap: '3px' }}>
                      <span style={{ fontWeight: 700, color: '#F59E0B' }}>💡 Key Tips to Bridge the Gap:</span>
                      <ul style={{ margin: 0, paddingLeft: '18px', color: '#CBD5E1', display: 'flex', flexDirection: 'column', gap: '2px' }}>
                        {scorecardRecommendedAnswer.improvement_areas.map((tip, idx) => (
                          <li key={idx}>{tip}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              {/* 3. STAR Framework Breakdown Matrix */}
              {analysisResult.coach?.star_breakdown && (
                <div style={{ padding: '14px 18px', background: 'rgba(15, 23, 42, 0.5)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)' }}>
                  <div style={{ fontSize: '12px', fontWeight: 700, color: '#E2E8F0', marginBottom: '10px' }}>
                    ⭐ STAR Method Framework Evaluation:
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px' }}>
                    {Object.entries(analysisResult.coach.star_breakdown).map(([key, val]) => (
                      <div
                        key={key}
                        style={{
                          padding: '10px 8px',
                          background: val ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.1)',
                          border: `1px solid ${val ? 'rgba(16, 185, 129, 0.4)' : 'rgba(239, 68, 68, 0.3)'}`,
                          borderRadius: '6px',
                          textAlign: 'center',
                        }}
                      >
                        <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-sub)' }}>{key}</div>
                        <div style={{ fontSize: '14px', fontWeight: 800, color: val ? '#34D399' : '#FCA5A5', marginTop: '2px' }}>
                          {val ? '✓ Detected' : '✗ Missing'}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* 4. Bidirectional LSTM vs RNN Side-by-Side */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                <div style={{ padding: '14px', background: 'rgba(99, 102, 241, 0.15)', border: '1px solid rgba(99, 102, 241, 0.35)', borderRadius: 'var(--radius-sm)' }}>
                  <div style={{ fontSize: '11px', color: '#38BDF8', fontWeight: 700 }}>BIDIRECTIONAL LSTM TONE</div>
                  <div style={{ fontSize: '16px', fontWeight: 800, marginTop: '2px', color: '#FFF' }}>
                    {analysisResult.comparison.lstm.sentiment}
                  </div>
                  <div style={{ fontSize: '12px', color: '#34D399', marginTop: '2px' }}>
                    Confidence: {(analysisResult.comparison.lstm.confidence * 100).toFixed(1)}%
                  </div>
                </div>

                <div style={{ padding: '14px', background: 'rgba(30, 41, 59, 0.5)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)' }}>
                  <div style={{ fontSize: '11px', color: 'var(--text-sub)', fontWeight: 700 }}>SIMPLE RNN BASELINE</div>
                  <div style={{ fontSize: '16px', fontWeight: 800, marginTop: '2px', color: '#CBD5E1' }}>
                    {analysisResult.comparison.rnn.sentiment}
                  </div>
                  <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '2px' }}>
                    Confidence: {(analysisResult.comparison.rnn.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* 5. Action Verbs & Weak Phrase Fixes */}
              {analysisResult.coach && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {analysisResult.coach.found_action_verbs && analysisResult.coach.found_action_verbs.length > 0 && (
                    <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                      <strong style={{ color: '#34D399' }}>Strong Action Verbs: </strong>
                      {analysisResult.coach.found_action_verbs.map((v) => (
                        <span key={v} className="badge badge-primary" style={{ marginRight: '4px', fontSize: '11px' }}>
                          {v}
                        </span>
                      ))}
                    </div>
                  )}

                  {analysisResult.coach.found_weak_phrases && analysisResult.coach.found_weak_phrases.length > 0 && (
                    <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                      <strong style={{ color: '#F59E0B' }}>Weak Filler Phrases: </strong>
                      {analysisResult.coach.found_weak_phrases.map((item, idx) => (
                        <span key={idx} style={{ color: '#FCD34D', marginRight: '8px' }}>
                          "{item.phrase}" → <em>{item.fix}</em>
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* 6. AI Enhanced Rephrased Version */}
              {analysisResult.coach?.rephrased_preview && analysisResult.coach.found_weak_phrases && analysisResult.coach.found_weak_phrases.length > 0 && (
                <div style={{ padding: '14px', background: 'rgba(99, 102, 241, 0.1)', border: '1px solid rgba(99, 102, 241, 0.3)', borderRadius: 'var(--radius-sm)' }}>
                  <div style={{ fontSize: '12px', fontWeight: 700, color: '#818CF8', marginBottom: '4px' }}>
                    ✨ AI-Enhanced Polish (Upgraded Assertive Phrasing):
                  </div>
                  <div style={{ fontSize: '13px', color: '#E2E8F0', lineHeight: 1.4 }}>
                    {analysisResult.coach.rephrased_preview}
                  </div>
                </div>
              )}

              {/* 7. Coaching Recommendations */}
              {analysisResult.coach?.tone_feedback && analysisResult.coach.tone_feedback.length > 0 && (
                <div style={{ padding: '14px', background: 'rgba(15, 23, 42, 0.5)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)' }}>
                  <div style={{ fontSize: '13px', fontWeight: 700, color: '#38BDF8', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <CheckCircle2 size={16} color="#38BDF8" />
                    <span>Actionable Recommendations:</span>
                  </div>
                  <ul style={{ paddingLeft: '18px', fontSize: '12px', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '6px', margin: 0 }}>
                    {analysisResult.coach.tone_feedback.map((tip, idx) => (
                      <li key={idx}>{tip}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ) : (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '40px', gap: '12px', color: 'var(--text-muted)' }}>
              <Volume2 size={36} color="#6366F1" style={{ opacity: 0.7 }} />
              <div style={{ textAlign: 'center', maxWidth: '320px', fontSize: '13px' }}>
                Select a question on the left, dictate or type your answer, and click <strong>"Evaluate Response & Generate Scorecard"</strong>.
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
