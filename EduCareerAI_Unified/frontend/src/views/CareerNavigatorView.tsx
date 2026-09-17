import React, { useState, useEffect } from 'react';
import confetti from 'canvas-confetti';
import {
  Compass,
  Sparkles,
  BookOpen,
  Check,
  ChevronRight,
  BarChart3,
  Award,
  Zap,
} from 'lucide-react';
import { api } from '../api';
import type { CareerOption, CareerPredictionResult, User } from '../types';

interface CareerNavigatorViewProps {
  user: User | null;
  onNavigateToRoadmap: (careerName: string) => void;
  onSetContext: (context: { topCareer?: string }) => void;
}

const DEFAULT_CAREER_OPTIONS: CareerOption = {
  fields: [
    'Computer Science & IT',
    'Engineering',
    'Medical & Healthcare',
    'Business & Finance',
    'Science & Research',
    'Arts & Design',
    'Education',
    'Law & Government',
    'Media & Communication',
    'Sports & Fitness',
  ],
  grades: [
    { id: 'grade_math', label: 'Math' },
    { id: 'grade_science', label: 'Science' },
    { id: 'grade_lang', label: 'Language' },
    { id: 'grade_social', label: 'Social Science' },
    { id: 'grade_cs', label: 'Computer Science' },
  ],
  aptitudes: [
    { id: 'score_analytical', label: 'Analytical Aptitude' },
    { id: 'score_numeric', label: 'Numerical Aptitude' },
    { id: 'score_verbal', label: 'Verbal Aptitude' },
    { id: 'score_creative', label: 'Creative Aptitude' },
    { id: 'score_social', label: 'Social Aptitude' },
  ],
  hobbies: [
    { id: 'hobby_coding', label: 'Coding' },
    { id: 'hobby_gaming', label: 'Gaming' },
    { id: 'hobby_reading', label: 'Reading' },
    { id: 'hobby_writing', label: 'Writing' },
    { id: 'hobby_music', label: 'Music' },
    { id: 'hobby_drawing_art', label: 'Drawing Art' },
    { id: 'hobby_sports', label: 'Sports' },
    { id: 'hobby_cooking', label: 'Cooking' },
    { id: 'hobby_photography', label: 'Photography' },
    { id: 'hobby_travel', label: 'Travel' },
    { id: 'hobby_science_experiments', label: 'Science Experiments' },
    { id: 'hobby_volunteering', label: 'Volunteering' },
    { id: 'hobby_debating', label: 'Debating' },
    { id: 'hobby_robotics', label: 'Robotics' },
    { id: 'hobby_fashion', label: 'Fashion' },
    { id: 'hobby_business_trading', label: 'Business Trading' },
  ],
};

const INITIAL_HOBBIES: Record<string, number> = {
  hobby_coding: 1,
  hobby_robotics: 1,
  hobby_gaming: 1,
  hobby_reading: 0,
  hobby_writing: 0,
  hobby_music: 0,
  hobby_drawing_art: 0,
  hobby_sports: 0,
  hobby_cooking: 0,
  hobby_photography: 0,
  hobby_travel: 0,
  hobby_science_experiments: 0,
  hobby_volunteering: 0,
  hobby_debating: 0,
  hobby_fashion: 0,
  hobby_business_trading: 0,
};

export const CareerNavigatorView: React.FC<CareerNavigatorViewProps> = ({
  user,
  onNavigateToRoadmap,
  onSetContext,
}) => {
  const [options, setOptions] = useState<CareerOption>(DEFAULT_CAREER_OPTIONS);
  const [field, setField] = useState('Computer Science & IT');
  const [hobbies, setHobbies] = useState<Record<string, number>>(INITIAL_HOBBIES);
  const [grades, setGrades] = useState<Record<string, number>>({
    grade_math: 85,
    grade_science: 80,
    grade_lang: 75,
    grade_social: 70,
    grade_cs: 90,
  });
  const [aptitudes, setAptitudes] = useState<Record<string, number>>({
    score_analytical: 88,
    score_numeric: 82,
    score_verbal: 78,
    score_creative: 75,
    score_social: 70,
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<CareerPredictionResult | null>(null);

  useEffect(() => {
    const fetchOptions = async () => {
      try {
        const opts = await api.getCareerOptions();
        if (opts && opts.fields && opts.fields.length > 0) {
          setOptions(opts);
        }
      } catch (err) {
        console.warn('Using default career options', err);
      }
    };
    fetchOptions();
  }, []);

  // Hydrate user profile and latest assessment when user changes
  useEffect(() => {
    if (!user?.id) {
      setResults(null);
      setField('Computer Science & IT');
      setHobbies(INITIAL_HOBBIES);
      setGrades({
        grade_math: 85,
        grade_science: 80,
        grade_lang: 75,
        grade_social: 70,
        grade_cs: 90,
      });
      setAptitudes({
        score_analytical: 88,
        score_numeric: 82,
        score_verbal: 78,
        score_creative: 75,
        score_social: 70,
      });
      return;
    }

    const loadUserData = async () => {
      try {
        const profileRes = await api.getProfile(user.id);
        if (profileRes?.profile) {
          const prof = profileRes.profile;
          if (prof.field) setField(prof.field);
          if (prof.hobbies && Object.keys(prof.hobbies).length > 0) {
            setHobbies((prev) => ({ ...prev, ...prof.hobbies }));
          }
          if (prof.grades && Object.keys(prof.grades).length > 0) {
            setGrades((prev) => ({ ...prev, ...prof.grades }));
          }
          if (prof.aptitudes && Object.keys(prof.aptitudes).length > 0) {
            setAptitudes((prev) => ({ ...prev, ...prof.aptitudes }));
          }
        }

        const latestAssessment = profileRes?.latest_assessment;
        if (latestAssessment && latestAssessment.careers && latestAssessment.careers.length > 0) {
          const currentGrades = profileRes.profile?.grades || grades;
          const currentAptitudes = profileRes.profile?.aptitudes || aptitudes;
          const currentHobbies = profileRes.profile?.hobbies || hobbies;
          const gVals = Object.values(currentGrades) as number[];
          const aVals = Object.values(currentAptitudes) as number[];
          const hVals = Object.values(currentHobbies) as number[];

          setResults({
            success: true,
            careers: latestAssessment.careers,
            courses: latestAssessment.courses || [],
            input_summary: {
              field: profileRes.profile?.field || field,
              academic_avg: gVals.length ? Math.round(gVals.reduce((a, b) => a + b, 0) / gVals.length) : 80,
              aptitude_avg: aVals.length ? Math.round(aVals.reduce((a, b) => a + b, 0) / aVals.length) : 80,
              active_hobbies_count: hVals.filter(Boolean).length,
            },
          });
          onSetContext({ topCareer: latestAssessment.top_career });
        } else {
          setResults(null);
        }
      } catch (e) {
        console.warn('Could not load user profile or assessment history', e);
      }
    };

    loadUserData();
  }, [user?.id]);

  const toggleHobby = (hobbyId: string) => {
    setHobbies((prev) => ({
      ...prev,
      [hobbyId]: prev[hobbyId] ? 0 : 1,
    }));
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.predictCareers({
        user_id: user?.id,
        field,
        hobbies,
        grades,
        aptitudes,
        top_n: 5,
      });
      setResults(res);
      if (res && res.careers && res.careers.length > 0) {
        onSetContext({ topCareer: res.careers[0].Career });
        try {
          confetti({
            particleCount: 80,
            spread: 70,
            origin: { y: 0.6 },
          });
        } catch (_c) {}
      }
    } catch (err: any) {
      console.error('Prediction failed', err);
      setError(err.response?.data?.detail || err.message || 'Prediction failed. Please ensure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Error banner */}
      {error && (
        <div style={{
          padding: '12px 16px',
          background: 'rgba(244, 63, 94, 0.15)',
          border: '1px solid rgba(244, 63, 94, 0.4)',
          borderRadius: 'var(--radius-sm)',
          color: '#FB7185',
          fontSize: '14px',
          fontWeight: 600,
        }}>
          ⚠️ {error}
        </div>
      )}
      {/* Header */}
      <div>
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
          <span className="badge badge-primary">
            <Sparkles size={12} />
            HistGradientBoosting ML Classifier
          </span>
        </div>
        <h1 style={{ fontSize: '28px', fontWeight: 800 }}>
          Career & Course <span className="gradient-text">Navigator</span>
        </h1>
        <p style={{ fontSize: '14px', color: 'var(--text-muted)' }}>
          Provide your academic marks, cognitive aptitudes, and personal hobbies to receive calibrated ML recommendations.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(340px, 1fr) minmax(360px, 1.2fr)', gap: '24px' }}>
        {/* Left Column: Input Form */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Target Field */}
          <div className="glass-card" style={{ padding: '20px' }}>
            <label className="input-label" style={{ fontSize: '14px', color: '#FFF' }}>
              🎯 Primary Domain / Field of Interest
            </label>
            <select
              className="form-select"
              value={field}
              onChange={(e) => setField(e.target.value)}
            >
              {options?.fields.map((f) => (
                <option key={f} value={f}>{f}</option>
              ))}
            </select>
          </div>

          {/* Academic Grades */}
          <div className="glass-card" style={{ padding: '20px' }}>
            <h3 style={{ fontSize: '16px', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <BarChart3 size={18} color="#818CF8" />
              <span>Academic Performance (0–100%)</span>
            </h3>
            {options?.grades.map((g) => (
              <div key={g.id} className="slider-container">
                <div className="slider-header">
                  <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>{g.label}</span>
                  <span className="slider-value">{grades[g.id] ?? 65}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={grades[g.id] ?? 65}
                  onChange={(e) => setGrades({ ...grades, [g.id]: parseFloat(e.target.value) })}
                />
              </div>
            ))}
          </div>

          {/* Cognitive Aptitudes */}
          <div className="glass-card" style={{ padding: '20px' }}>
            <h3 style={{ fontSize: '16px', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Zap size={18} color="#06B6D4" />
              <span>Cognitive & Aptitude Scores (0–100%)</span>
            </h3>
            {options?.aptitudes.map((a) => (
              <div key={a.id} className="slider-container">
                <div className="slider-header">
                  <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>{a.label}</span>
                  <span className="slider-value" style={{ color: '#06B6D4' }}>{aptitudes[a.id] ?? 65}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={aptitudes[a.id] ?? 65}
                  onChange={(e) => setAptitudes({ ...aptitudes, [a.id]: parseFloat(e.target.value) })}
                />
              </div>
            ))}
          </div>

          {/* Hobby Chips */}
          <div className="glass-card" style={{ padding: '20px' }}>
            <h3 style={{ fontSize: '16px', marginBottom: '14px' }}>
              🎨 Select Active Hobbies & Interests
            </h3>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
              {options?.hobbies.map((h) => {
                const active = Boolean(hobbies[h.id]);
                return (
                  <button
                    key={h.id}
                    onClick={() => toggleHobby(h.id)}
                    style={{
                      padding: '6px 12px',
                      borderRadius: 'var(--radius-full)',
                      border: active ? '1px solid rgba(99, 102, 241, 0.6)' : '1px solid var(--border-color)',
                      background: active ? 'rgba(99, 102, 241, 0.25)' : 'rgba(15, 23, 42, 0.5)',
                      color: active ? '#818CF8' : 'var(--text-muted)',
                      fontSize: '12px',
                      fontWeight: 600,
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      transition: 'all 0.15s ease',
                    }}
                  >
                    {active && <Check size={12} />}
                    <span>{h.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Predict Button */}
          <button
            className="btn btn-primary"
            style={{ padding: '14px 20px', fontSize: '16px' }}
            onClick={handlePredict}
            disabled={loading}
          >
            <Sparkles size={18} />
            <span>{loading ? 'Evaluating ML Model...' : 'Calculate AI Career Ranking'}</span>
          </button>
        </div>

        {/* Right Column: Prediction Results */}
        <div>
          {results ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              {/* Summary Banner */}
              <div className="glass-card" style={{
                padding: '20px',
                background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(6, 182, 212, 0.1) 100%)',
                border: '1px solid rgba(99, 102, 241, 0.3)',
              }}>
                <div style={{ fontSize: '12px', color: 'var(--accent-cyan)', fontWeight: 700, textTransform: 'uppercase' }}>
                  Top ML Match Identified
                </div>
                <div style={{ fontSize: '24px', fontWeight: 800, color: '#FFF', margin: '4px 0 10px' }}>
                  {results.careers[0]?.Career}
                </div>
                <div style={{ display: 'flex', gap: '16px', fontSize: '13px', color: 'var(--text-muted)' }}>
                  <span>Suitability: <b style={{ color: '#34D399' }}>{results.careers[0]?.Suitability_Percent}%</b></span>
                  <span>Academic Avg: <b style={{ color: '#FFF' }}>{results.input_summary.academic_avg}%</b></span>
                  <span>Aptitude Avg: <b style={{ color: '#FFF' }}>{results.input_summary.aptitude_avg}%</b></span>
                </div>
              </div>

              {/* Ranked Careers List */}
              <div className="glass-card" style={{ padding: '20px' }}>
                <h3 style={{ fontSize: '16px', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Award size={18} color="#FBBF24" />
                  <span>Top 5 Ranked Careers</span>
                </h3>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {results.careers.map((c, idx) => (
                    <div
                      key={c.Career}
                      style={{
                        padding: '14px',
                        borderRadius: 'var(--radius-sm)',
                        background: idx === 0 ? 'rgba(99, 102, 241, 0.1)' : 'rgba(15, 23, 42, 0.4)',
                        border: `1px solid ${idx === 0 ? 'rgba(99, 102, 241, 0.4)' : 'var(--border-color)'}`,
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{
                            width: '24px',
                            height: '24px',
                            borderRadius: '50%',
                            background: idx === 0 ? '#6366F1' : '#334155',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: '12px',
                            fontWeight: 700,
                          }}>
                            #{c.Rank}
                          </span>
                          <span style={{ fontSize: '15px', fontWeight: 700 }}>{c.Career}</span>
                        </div>

                        <button
                          className="btn btn-ghost"
                          style={{ padding: '4px 8px', fontSize: '12px', color: '#818CF8' }}
                          onClick={() => onNavigateToRoadmap(c.Career)}
                        >
                          <span>Roadmap</span>
                          <ChevronRight size={14} />
                        </button>
                      </div>

                      {/* Probability Bar */}
                      <div style={{ width: '100%', height: '8px', background: '#334155', borderRadius: '4px', overflow: 'hidden' }}>
                        <div style={{
                          height: '100%',
                          width: `${c.Suitability_Percent}%`,
                          background: idx === 0
                            ? 'linear-gradient(90deg, #6366F1 0%, #38BDF8 100%)'
                            : 'linear-gradient(90deg, #475569 0%, #64748B 100%)',
                          borderRadius: '4px',
                          transition: 'width 0.6s ease',
                        }} />
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'flex-end', fontSize: '11px', color: 'var(--text-sub)', marginTop: '4px' }}>
                        <span>Suitability: {c.Suitability_Percent}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Recommended Courses Pathway */}
              <div className="glass-card" style={{ padding: '20px' }}>
                <h3 style={{ fontSize: '16px', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <BookOpen size={18} color="#34D399" />
                  <span>Recommended Degree & Course Pathways</span>
                </h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                  {results.courses.map((crs, i) => (
                    <div
                      key={i}
                      style={{
                        padding: '12px',
                        background: 'rgba(15, 23, 42, 0.4)',
                        border: '1px solid var(--border-color)',
                        borderRadius: 'var(--radius-sm)',
                      }}
                    >
                      <div style={{ fontSize: '13px', fontWeight: 600, color: '#FFF' }}>{crs.course}</div>
                      <span className="badge badge-info" style={{ fontSize: '10px', marginTop: '6px' }}>
                        {crs.field || 'Relevant Major'}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
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
                background: 'rgba(99, 102, 241, 0.1)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: '16px',
              }}>
                <Compass size={32} color="#818CF8" />
              </div>
              <h3 style={{ fontSize: '18px', marginBottom: '8px' }}>Awaiting Assessment Inputs</h3>
              <p style={{ fontSize: '13px', color: 'var(--text-muted)', maxWidth: '320px', lineHeight: 1.5 }}>
                Adjust your academic grades, aptitudes, and hobbies on the left, then click <b>Calculate AI Career Ranking</b> to execute the ML pipeline.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
