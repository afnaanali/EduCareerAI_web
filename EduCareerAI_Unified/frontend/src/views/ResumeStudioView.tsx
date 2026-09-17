import React, { useState, useEffect } from 'react';
import confetti from 'canvas-confetti';
import {
  FileText,
  Upload,
  AlertTriangle,
  Target,
  Sparkles,
  Layers,
} from 'lucide-react';
import { api } from '../api';
import type { ResumeAnalysisResult, User } from '../types';

interface ResumeStudioViewProps {
  user: User | null;
  onSetContext: (context: { resumeName?: string }) => void;
  onNavigateToRoadmap?: (career: string) => void;
}

const DEFAULT_TARGET_ROLES = [
  'Data Analyst',
  'Data Scientist',
  'Software Engineer',
  'DevOps Engineer',
  'Machine Learning Engineer',
  'Full Stack Developer',
  'Cloud Engineer',
  'Product Manager',
  'Cybersecurity Analyst',
  'AI Engineer',
];

export const ResumeStudioView: React.FC<ResumeStudioViewProps> = ({
  user,
  onSetContext,
}) => {
  const [tab, setTab] = useState<'upload' | 'paste'>('upload');
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [resumeText, setResumeText] = useState('');
  const [targetJob, setTargetJob] = useState('Data Analyst');
  const [targetRoles, setTargetRoles] = useState<string[]>(DEFAULT_TARGET_ROLES);
  const [jobDescription] = useState('');

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<ResumeAnalysisResult | null>(null);

  useEffect(() => {
    const fetchRoles = async () => {
      try {
        const res = await api.getTargetRoles();
        if (res && res.roles && res.roles.length > 0) {
          setTargetRoles(res.roles);
        }
      } catch (err) {
        console.warn('Using default target roles', err);
      }
    };
    fetchRoles();
  }, []);

  // Hydrate previous scan or reset when user changes
  useEffect(() => {
    if (!user?.id) {
      setResults(null);
      setFile(null);
      setResumeText('');
      return;
    }

    const loadUserResume = async () => {
      try {
        const history = await api.getUserHistory(user.id);
        if (history && history.resume_scans && history.resume_scans.length > 0) {
          const latest = history.resume_scans[0];
          setResults({
            success: true,
            resume_name: latest.resume_name,
            ats_score: latest.ats_score,
            target_job: latest.target_job,
            match_score: latest.match_score,
            breakdown: {
              'Format & Structure': 12,
              'Contact Info': 10,
              'Core Keywords': 18,
              'Action Verbs': 12,
              'Measurable Impact': 10,
              'Section Headings': 12,
              'Grammar & Style': 10,
              'Technical Density': 14,
            },
            extracted_skills: latest.skills || [],
            ann_vector: { skills: [], vector: [] },
            matched_job_skills: [],
            missing_job_skills: [],
            skill_gap: [],
            matched_careers: [],
            issues: latest.issues || [],
            sections_detected: {},
            word_count: 350,
          });
          onSetContext({ resumeName: latest.resume_name });
        } else {
          setResults(null);
          setFile(null);
          setResumeText('');
        }
      } catch (e) {
        console.warn('Failed to load user resume history', e);
      }
    };

    loadUserResume();
  }, [user?.id]);

  const handleFileSelect = (selectedFile: File | null) => {
    setError(null);
    if (!selectedFile) {
      setFile(null);
      return;
    }
    const ext = selectedFile.name.toLowerCase().split('.').pop();
    if (!['pdf', 'docx', 'doc', 'txt'].includes(ext || '')) {
      setError(`Unsupported file extension .${ext}. Please select a PDF, DOCX, or TXT file.`);
      return;
    }
    if (selectedFile.size > 15 * 1024 * 1024) {
      setError('File size exceeds the 15MB limit. Please upload a smaller document.');
      return;
    }
    setFile(selectedFile);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);
    try {
      let res: ResumeAnalysisResult;
      if (tab === 'upload') {
        if (!file) {
          setError('Please select or drop a resume file first.');
          setLoading(false);
          return;
        }
        const formData = new FormData();
        formData.append('file', file, file.name);
        if (user?.id) formData.append('user_id', user.id.toString());
        if (targetJob) formData.append('target_job', targetJob);
        if (jobDescription) formData.append('job_description', jobDescription);
        res = await api.analyzeResumeFile(formData);
      } else {
        if (!resumeText.trim()) {
          setError('Please paste your resume text before running diagnostics.');
          setLoading(false);
          return;
        }
        res = await api.analyzeResumeText({
          user_id: user?.id,
          resume_text: resumeText,
          resume_name: 'Pasted_Resume.txt',
          target_job: targetJob,
          job_description: jobDescription || undefined,
        });
      }
      setResults(res);
      onSetContext({ resumeName: res.resume_name });
      if (res.ats_score >= 70) {
        confetti({ particleCount: 70, spread: 60, origin: { y: 0.6 } });
      }
    } catch (err: any) {
      console.error('Resume analysis failed', err);
      const detail =
        err?.response?.data?.detail ||
        err?.response?.data?.message ||
        err?.message ||
        'Failed to parse or analyze resume document.';
      setError(detail);
    } finally {
      setLoading(false);
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 75) return '#10B981';
    if (score >= 50) return '#F59E0B';
    return '#F43F5E';
  };

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Header */}
      <div>
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
          <span className="badge badge-primary">
            <FileText size={12} />
            9-Vector ATS Diagnostics & Skill Gap
          </span>
        </div>
        <h1 style={{ fontSize: '28px', fontWeight: 800 }}>
          Resume & ATS <span className="gradient-text">Studio</span>
        </h1>
        <p style={{ fontSize: '14px', color: 'var(--text-muted)' }}>
          Evaluate applicant tracking system compatibility, parse technical competencies, and benchmark against targeted roles.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(320px, 1fr) minmax(360px, 1.4fr)', gap: '24px' }}>
        {/* Left Column: Input Form */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Target Role Selector */}
          <div className="glass-card" style={{ padding: '20px' }}>
            <label className="input-label" style={{ fontSize: '14px', color: '#FFF' }}>
              🎯 Target Career Benchmark
            </label>
            <select
              className="form-select"
              value={targetJob}
              onChange={(e) => setTargetJob(e.target.value)}
            >
              {targetRoles.map((r) => (
                <option key={r} value={r}>{r}</option>
              ))}
            </select>
          </div>

          {/* Upload or Paste Box */}
          <div className="glass-card" style={{ padding: '20px' }}>
            {/* Tab switch */}
            <div style={{
              display: 'flex',
              background: 'rgba(15, 23, 42, 0.6)',
              borderRadius: 'var(--radius-sm)',
              padding: '4px',
              marginBottom: '16px',
            }}>
              <button
                onClick={() => { setTab('upload'); setError(null); }}
                style={{
                  flex: 1,
                  padding: '8px',
                  border: 'none',
                  borderRadius: '6px',
                  background: tab === 'upload' ? 'var(--primary)' : 'transparent',
                  color: tab === 'upload' ? '#FFF' : 'var(--text-muted)',
                  fontWeight: 600,
                  fontSize: '13px',
                  cursor: 'pointer',
                }}
              >
                Upload File (PDF/DOCX)
              </button>
              <button
                onClick={() => { setTab('paste'); setError(null); }}
                style={{
                  flex: 1,
                  padding: '8px',
                  border: 'none',
                  borderRadius: '6px',
                  background: tab === 'paste' ? 'var(--primary)' : 'transparent',
                  color: tab === 'paste' ? '#FFF' : 'var(--text-muted)',
                  fontWeight: 600,
                  fontSize: '13px',
                  cursor: 'pointer',
                }}
              >
                Paste Plain Text
              </button>
            </div>

            {tab === 'upload' ? (
              <div
                style={{
                  border: isDragging ? '2px dashed #818CF8' : '2px dashed var(--border-color)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '30px 20px',
                  textAlign: 'center',
                  background: isDragging ? 'rgba(99, 102, 241, 0.15)' : 'rgba(15, 23, 42, 0.4)',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                }}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => document.getElementById('resume-file-input')?.click()}
              >
                <input
                  id="resume-file-input"
                  type="file"
                  accept=".pdf,.docx,.doc,.txt"
                  style={{ display: 'none' }}
                  onChange={(e) => handleFileSelect(e.target.files?.[0] || null)}
                />
                <Upload size={32} color={isDragging ? '#A5B4FC' : '#818CF8'} style={{ marginBottom: '10px' }} />
                <div style={{ fontSize: '14px', fontWeight: 600, color: '#FFF' }}>
                  {file ? file.name : (isDragging ? 'Drop resume file here...' : 'Click to Browse or Drag Resume')}
                </div>
                {file ? (
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px', marginTop: '8px' }}>
                    <span style={{ fontSize: '12px', color: '#94A3B8' }}>
                      {(file.size / 1024).toFixed(1)} KB
                    </span>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setFile(null);
                        const input = document.getElementById('resume-file-input') as HTMLInputElement;
                        if (input) input.value = '';
                      }}
                      style={{
                        background: 'rgba(239, 68, 68, 0.15)',
                        color: '#F87171',
                        border: '1px solid rgba(239, 68, 68, 0.3)',
                        borderRadius: '4px',
                        padding: '2px 8px',
                        fontSize: '11px',
                        cursor: 'pointer',
                        fontWeight: 600,
                      }}
                    >
                      Remove File
                    </button>
                  </div>
                ) : (
                  <div style={{ fontSize: '12px', color: 'var(--text-sub)', marginTop: '4px' }}>
                    Supports PDF, DOCX, TXT files (up to 15MB)
                  </div>
                )}
              </div>
            ) : (
              <div>
                <label className="input-label">Resume Raw Text</label>
                <textarea
                  className="form-textarea"
                  rows={10}
                  placeholder="Paste complete resume content here including contact, education, skills, and work history..."
                  value={resumeText}
                  onChange={(e) => { setResumeText(e.target.value); setError(null); }}
                />
              </div>
            )}
          </div>

          {/* Error Message if any */}
          {error && (
            <div style={{
              display: 'flex',
              alignItems: 'flex-start',
              gap: '10px',
              padding: '12px 14px',
              background: 'rgba(239, 68, 68, 0.12)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              borderRadius: 'var(--radius-sm)',
              color: '#FCA5A5',
              fontSize: '13px',
              lineHeight: 1.4,
            }}>
              <AlertTriangle size={18} style={{ flexShrink: 0, marginTop: '2px', color: '#EF4444' }} />
              <div>
                <b>Upload / Scan Error:</b> {error}
              </div>
            </div>
          )}

          <button
            className="btn btn-primary"
            style={{ padding: '14px', fontSize: '15px' }}
            onClick={handleAnalyze}
            disabled={loading || (tab === 'upload' && !file) || (tab === 'paste' && !resumeText.trim())}
          >
            <Sparkles size={18} />
            <span>{loading ? 'Evaluating ATS Diagnostics...' : 'Run ATS & Skill Gap Scan'}</span>
          </button>
        </div>

        {/* Right Column: Diagnostic Report */}
        <div>
          {results ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              {/* Score Header Card */}
              <div className="glass-card" style={{
                padding: '24px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                flexWrap: 'wrap',
                gap: '20px',
                borderLeft: `6px solid ${getScoreColor(results.ats_score)}`,
              }}>
                <div>
                  <div style={{ fontSize: '12px', color: 'var(--text-sub)', fontWeight: 700, textTransform: 'uppercase' }}>
                    Diagnostic Score For: {results.resume_name}
                  </div>
                  <div style={{ fontSize: '22px', fontWeight: 800, marginTop: '4px' }}>
                    ATS Compatibility Rating
                  </div>
                  <div style={{ fontSize: '13px', color: 'var(--text-muted)', marginTop: '2px' }}>
                    Target Job: <b style={{ color: '#38BDF8' }}>{results.target_job}</b>
                    {results.match_score !== null && (
                      <span> • Match: <b style={{ color: '#34D399' }}>{results.match_score}%</b></span>
                    )}
                  </div>
                </div>

                {/* Score Number Dial */}
                <div style={{
                  width: '90px',
                  height: '90px',
                  borderRadius: '50%',
                  background: 'rgba(15, 23, 42, 0.8)',
                  border: `4px solid ${getScoreColor(results.ats_score)}`,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  boxShadow: `0 0 20px ${getScoreColor(results.ats_score)}44`,
                }}>
                  <span style={{ fontSize: '28px', fontWeight: 900, color: getScoreColor(results.ats_score), lineHeight: 1 }}>
                    {results.ats_score}
                  </span>
                  <span style={{ fontSize: '10px', color: 'var(--text-sub)', fontWeight: 700 }}>/ 100</span>
                </div>
              </div>

              {/* 9-Vector Breakdown */}
              <div className="glass-card" style={{ padding: '20px' }}>
                <h3 style={{ fontSize: '16px', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Layers size={18} color="#818CF8" />
                  <span>9-Vector ATS Scoring Breakdown</span>
                </h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                  {Object.entries(results.breakdown).map(([category, pts]) => (
                    <div key={category} style={{ background: 'rgba(15, 23, 42, 0.4)', padding: '10px 12px', borderRadius: 'var(--radius-sm)' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                        <span style={{ color: 'var(--text-muted)' }}>{category}</span>
                        <span style={{ fontWeight: 700, color: '#FFF' }}>{pts} pts</span>
                      </div>
                      <div style={{ width: '100%', height: '6px', background: '#334155', borderRadius: '3px', overflow: 'hidden' }}>
                        <div style={{
                          height: '100%',
                          width: `${Math.min(pts * 6.6, 100)}%`,
                          background: 'linear-gradient(90deg, #6366F1, #38BDF8)',
                        }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Extracted Skills & Missing Skills */}
              <div className="glass-card" style={{ padding: '20px' }}>
                <h3 style={{ fontSize: '16px', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Target size={18} color="#34D399" />
                  <span>Extracted Skills ({results.extracted_skills.length})</span>
                </h3>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginBottom: '16px' }}>
                  {results.extracted_skills.map((s) => (
                    <span key={s} className="badge badge-success" style={{ fontSize: '12px' }}>
                      {s}
                    </span>
                  ))}
                </div>

                {results.missing_job_skills.length > 0 && (
                  <div>
                    <div style={{ fontSize: '12px', fontWeight: 700, color: '#FB7185', marginBottom: '8px', textTransform: 'uppercase' }}>
                      ⚠️ Missing Skills for {results.target_job}
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                      {results.missing_job_skills.map((s) => (
                        <span key={s} className="badge" style={{ background: 'rgba(244, 63, 94, 0.15)', color: '#FB7185', border: '1px solid rgba(244, 63, 94, 0.3)', fontSize: '12px' }}>
                          + {s}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Actionable Fixes */}
              {results.issues.length > 0 && (
                <div className="glass-card" style={{ padding: '20px' }}>
                  <h3 style={{ fontSize: '16px', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <AlertTriangle size={18} color="#FBBF24" />
                    <span>Actionable Resume Optimizations</span>
                  </h3>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    {results.issues.map((iss, i) => (
                      <div key={i} style={{ padding: '12px', background: 'rgba(15, 23, 42, 0.5)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)' }}>
                        <div style={{ fontSize: '14px', fontWeight: 700, color: '#FBBF24' }}>{iss.title}</div>
                        <div style={{ fontSize: '12px', color: 'var(--text-muted)', margin: '4px 0' }}>{iss.why}</div>
                        <div style={{ fontSize: '12px', color: '#34D399', fontWeight: 600 }}>💡 Fix: {iss.fix}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="glass-card" style={{
              padding: '60px 24px',
              textAlign: 'center',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              minHeight: '400px',
            }}>
              <div style={{
                width: '64px',
                height: '64px',
                borderRadius: '50%',
                background: 'rgba(16, 185, 129, 0.1)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: '16px',
              }}>
                <FileText size={32} color="#10B981" />
              </div>
              <h3 style={{ fontSize: '18px', marginBottom: '8px' }}>Ready for Diagnostics</h3>
              <p style={{ fontSize: '13px', color: 'var(--text-muted)', maxWidth: '320px', lineHeight: 1.5 }}>
                Upload a resume or paste text on the left to benchmark against ATS algorithms and view your skill gap analysis.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
