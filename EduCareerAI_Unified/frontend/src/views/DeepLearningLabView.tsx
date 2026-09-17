import React, { useState, useEffect, useRef } from 'react';
import {
  Brain,
  Cpu,
  Eye,
  MessageSquare,
  Sparkles,
  TrendingUp,
  Upload,
  RefreshCw,
  Edit3,
  Eraser,
  CheckCircle2,
  AlertTriangle,
  Zap,
  Layers,
  ArrowRight,
} from 'lucide-react';
import { api } from '../api';
import type { DeepLearningANNResult, DigitCNNResult, SentimentResult } from '../types';

interface DeepLearningLabViewProps {
  onNavigateToInterviewCoach?: () => void;
}

export const DeepLearningLabView: React.FC<DeepLearningLabViewProps> = ({ onNavigateToInterviewCoach }) => {
  const [activeSubTab, setActiveSubTab] = useState<'ann' | 'cnn' | 'sentiment'>('ann');

  // ──────────────────────────────────────────────
  // 1. ANN State
  // ──────────────────────────────────────────────
  const [annSkills] = useState<string[]>([
    'Python', 'SQL', 'AWS', 'Docker', 'Linux', 'Git', 'PowerBI', 'Excel'
  ]);
  const [annVector, setAnnVector] = useState<number[]>([1, 1, 0, 0, 0, 1, 1, 1]);
  const [annLoading, setAnnLoading] = useState(false);
  const [annResult, setAnnResult] = useState<DeepLearningANNResult | null>(null);
  const [annError, setAnnError] = useState<string | null>(null);

  // ANN Presets
  const annPresets = [
    { label: '🎯 Data Scientist', vec: [1, 1, 0, 0, 0, 1, 1, 0] },
    { label: '☁️ Cloud & DevOps', vec: [1, 0, 1, 1, 1, 1, 0, 0] },
    { label: '💻 Full Stack Dev', vec: [1, 1, 0, 1, 1, 1, 0, 0] },
    { label: '📊 Data Analyst', vec: [1, 1, 0, 0, 0, 0, 1, 1] },
    { label: '🔄 Clear All', vec: [0, 0, 0, 0, 0, 0, 0, 0] },
  ];

  // ──────────────────────────────────────────────
  // 2. CNN State
  // ──────────────────────────────────────────────
  const [cnnMode, setCnnMode] = useState<'draw' | 'upload'>('draw');
  const [digitFile, setDigitFile] = useState<File | null>(null);
  const [digitPreview, setDigitPreview] = useState<string | null>(null);
  const [digitLoading, setDigitLoading] = useState(false);
  const [digitResult, setDigitResult] = useState<DigitCNNResult | null>(null);
  const [digitError, setDigitError] = useState<string | null>(null);

  // Canvas drawing state
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);

  // ──────────────────────────────────────────────
  // 3. Sentiment State (RNN vs LSTM)
  // ──────────────────────────────────────────────
  const [interviewText, setInterviewText] = useState(
    "I have led cross-functional data teams and developed production machine learning pipelines that reduced processing time by 40%."
  );
  const [sentimentLoading, setSentimentLoading] = useState(false);
  const [sentimentResult, setSentimentResult] = useState<SentimentResult | null>(null);
  const [sentimentError, setSentimentError] = useState<string | null>(null);

  const samplePrompts = [
    {
      title: '🌟 High-Impact STAR Response',
      text: 'I spearheaded data engineering pipelines using PyTorch and optimized inference throughput by 45%, delivering $200k in annual compute savings.',
    },
    {
      title: '⚠️ Hesitant / Passive Response',
      text: 'I think I sort of helped the team with some data tasks and basically tried to fix a few bugs whenever I could.',
    },
    {
      title: '💼 Cloud Architect Response',
      text: 'I architected and automated our multi-region Kubernetes deployment on AWS, reducing cluster failover latency to under 30 seconds.',
    },
  ];

  // ──────────────────────────────────────────────
  // Lifecycle & Initial Predict
  // ──────────────────────────────────────────────
  useEffect(() => {
    handleAnnPredict(annVector);
  }, []);

  // Initialize canvas
  useEffect(() => {
    if (activeSubTab === 'cnn' && cnnMode === 'draw') {
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.fillStyle = '#000000';
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          drawSampleDigit(7); // default sample
        }
      }
    }
  }, [activeSubTab, cnnMode]);

  // ──────────────────────────────────────────────
  // ANN Handlers
  // ──────────────────────────────────────────────
  const toggleAnnSkill = (index: number) => {
    const updated = [...annVector];
    updated[index] = updated[index] === 1 ? 0 : 1;
    setAnnVector(updated);
    handleAnnPredict(updated);
  };

  const applyAnnPreset = (vec: number[]) => {
    setAnnVector(vec);
    handleAnnPredict(vec);
  };

  const handleAnnPredict = async (vec: number[] = annVector) => {
    setAnnLoading(true);
    setAnnError(null);
    try {
      const res = await api.predictAnn(vec);
      setAnnResult(res);
    } catch (err: any) {
      console.error('ANN prediction error', err);
      setAnnError(err.response?.data?.detail || err.message || 'Failed to compute ANN forward pass');
    } finally {
      setAnnLoading(false);
    }
  };

  // ──────────────────────────────────────────────
  // CNN Handlers & Canvas Logic
  // ──────────────────────────────────────────────
  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    setIsDrawing(true);
    draw(e);
  };

  const stopDrawing = () => {
    setIsDrawing(false);
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx?.beginPath();
    }
  };

  const getCanvasCoords = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    let clientX = 0;
    let clientY = 0;
    if ('touches' in e) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    } else {
      clientX = e.clientX;
      clientY = e.clientY;
    }
    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY,
    };
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    if (!isDrawing && e.type !== 'mousedown' && e.type !== 'touchstart') return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { x, y } = getCanvasCoords(e);
    ctx.lineWidth = 14;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#FFFFFF';

    if (e.type === 'mousedown' || e.type === 'touchstart') {
      ctx.beginPath();
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
      ctx.stroke();
    }
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setDigitResult(null);
    setDigitError(null);
  };

  const drawSampleDigit = (digit: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 14;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    const w = canvas.width;
    const h = canvas.height;

    ctx.beginPath();
    if (digit === 7) {
      ctx.moveTo(w * 0.25, h * 0.25);
      ctx.lineTo(w * 0.75, h * 0.25);
      ctx.lineTo(w * 0.45, h * 0.85);
      ctx.stroke();
    } else if (digit === 8) {
      ctx.arc(w * 0.5, h * 0.38, w * 0.18, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(w * 0.5, h * 0.66, w * 0.22, 0, Math.PI * 2);
      ctx.stroke();
    } else if (digit === 2) {
      ctx.arc(w * 0.5, h * 0.35, w * 0.18, Math.PI, 0);
      ctx.lineTo(w * 0.3, h * 0.8);
      ctx.lineTo(w * 0.75, h * 0.8);
      ctx.stroke();
    } else if (digit === 0) {
      ctx.ellipse(w * 0.5, h * 0.5, w * 0.22, h * 0.3, 0, 0, Math.PI * 2);
      ctx.stroke();
    } else if (digit === 4) {
      ctx.moveTo(w * 0.65, h * 0.2);
      ctx.lineTo(w * 0.3, h * 0.6);
      ctx.lineTo(w * 0.75, h * 0.6);
      ctx.moveTo(w * 0.65, h * 0.2);
      ctx.lineTo(w * 0.65, h * 0.85);
      ctx.stroke();
    }
  };

  const handleDigitUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setDigitFile(file);
    setDigitPreview(URL.createObjectURL(file));
    setDigitResult(null);
    setDigitError(null);
  };

  const handleDigitPredict = async () => {
    setDigitLoading(true);
    setDigitError(null);
    try {
      const formData = new FormData();

      if (cnnMode === 'draw') {
        const canvas = canvasRef.current;
        if (!canvas) throw new Error('Canvas not initialized');
        const dataUrl = canvas.toDataURL('image/png');
        formData.append('image_base64', dataUrl);
      } else {
        if (!digitFile) {
          throw new Error('Please upload an image file first');
        }
        formData.append('file', digitFile);
      }

      const res = await api.predictDigit(formData);
      setDigitResult(res);
    } catch (err: any) {
      console.error('CNN prediction error', err);
      setDigitError(err.response?.data?.detail || err.message || 'Error executing CNN inference');
    } finally {
      setDigitLoading(false);
    }
  };

  // ──────────────────────────────────────────────
  // Sentiment Handlers
  // ──────────────────────────────────────────────
  const handleSentimentAnalyze = async (textToAnalyze: string = interviewText) => {
    if (!textToAnalyze.trim()) return;
    setSentimentLoading(true);
    setSentimentError(null);
    try {
      const res = await api.analyzeSentimentTone(textToAnalyze);
      setSentimentResult(res);
    } catch (err: any) {
      console.error('Sentiment analysis error', err);
      setSentimentError(err.response?.data?.detail || err.message || 'Failed to analyze recurrent tone');
    } finally {
      setSentimentLoading(false);
    }
  };

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Header */}
      <div>
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
          <span className="badge badge-primary">
            <Brain size={12} />
            PyTorch & NumPy Neural Architectures
          </span>
        </div>
        <h1 style={{ fontSize: '28px', fontWeight: 800 }}>
          Deep Learning <span className="gradient-text">Neural Lab</span>
        </h1>
        <p style={{ fontSize: '14px', color: 'var(--text-muted)' }}>
          Interactive neural inference testbed: Artificial Neural Networks, Vision CNNs, and Recurrent LSTM sentiment classifiers.
        </p>
      </div>

      {/* Sub-Tab Navigation */}
      <div style={{
        display: 'flex',
        background: 'rgba(15, 23, 42, 0.6)',
        borderRadius: 'var(--radius-sm)',
        padding: '4px',
        border: '1px solid var(--border-color)',
        maxWidth: '520px',
      }}>
        <button
          onClick={() => setActiveSubTab('ann')}
          style={{
            flex: 1,
            padding: '10px 14px',
            border: 'none',
            borderRadius: '6px',
            background: activeSubTab === 'ann' ? 'var(--primary)' : 'transparent',
            color: activeSubTab === 'ann' ? '#FFF' : 'var(--text-muted)',
            fontWeight: 600,
            fontSize: '13px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px',
            transition: 'all 0.2s ease',
          }}
        >
          <Cpu size={16} />
          <span>8-Skill ANN</span>
        </button>

        <button
          onClick={() => setActiveSubTab('cnn')}
          style={{
            flex: 1,
            padding: '10px 14px',
            border: 'none',
            borderRadius: '6px',
            background: activeSubTab === 'cnn' ? 'var(--primary)' : 'transparent',
            color: activeSubTab === 'cnn' ? '#FFF' : 'var(--text-muted)',
            fontWeight: 600,
            fontSize: '13px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px',
            transition: 'all 0.2s ease',
          }}
        >
          <Eye size={16} />
          <span>Vision CNN</span>
        </button>

        <button
          onClick={() => setActiveSubTab('sentiment')}
          style={{
            flex: 1,
            padding: '10px 14px',
            border: 'none',
            borderRadius: '6px',
            background: activeSubTab === 'sentiment' ? 'var(--primary)' : 'transparent',
            color: activeSubTab === 'sentiment' ? '#FFF' : 'var(--text-muted)',
            fontWeight: 600,
            fontSize: '13px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px',
            transition: 'all 0.2s ease',
          }}
        >
          <MessageSquare size={16} />
          <span>RNN vs LSTM</span>
        </button>
      </div>

      {/* ────────────────────────────────────────────────────────── */}
      {/* ── TAB 1: ANN ── */}
      {/* ────────────────────────────────────────────────────────── */}
      {activeSubTab === 'ann' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.2fr', gap: '24px' }}>
          {/* Left: Skill Input Vector & Presets */}
          <div className="glass-card" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div>
              <h3 style={{ fontSize: '18px', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Zap size={20} color="#818CF8" />
                <span>8-Skill Input Vector</span>
              </h3>
              <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
                Toggle binary skill flags or select a career archetype to observe dense forward-pass weight propagation.
              </p>
            </div>

            {/* Presets Toolbar */}
            <div>
              <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-sub)', marginBottom: '8px' }}>
                Quick Skill Archetypes:
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                {annPresets.map((preset) => (
                  <button
                    key={preset.label}
                    onClick={() => applyAnnPreset(preset.vec)}
                    style={{
                      padding: '5px 10px',
                      borderRadius: '6px',
                      background: 'rgba(30, 41, 59, 0.7)',
                      border: '1px solid var(--border-color)',
                      color: '#E2E8F0',
                      fontSize: '12px',
                      cursor: 'pointer',
                      transition: 'all 0.15s ease',
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.borderColor = '#6366F1')}
                    onMouseLeave={(e) => (e.currentTarget.style.borderColor = 'var(--border-color)')}
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
            </div>

            {/* 8 Skill Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
              {annSkills.map((skill, i) => {
                const active = annVector[i] === 1;
                return (
                  <button
                    key={skill}
                    onClick={() => toggleAnnSkill(i)}
                    style={{
                      padding: '12px 14px',
                      borderRadius: 'var(--radius-sm)',
                      background: active ? 'rgba(99, 102, 241, 0.22)' : 'rgba(15, 23, 42, 0.5)',
                      border: active ? '1px solid #6366F1' : '1px solid var(--border-color)',
                      color: active ? '#FFF' : 'var(--text-muted)',
                      fontWeight: 600,
                      fontSize: '13px',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      transition: 'all 0.15s ease',
                    }}
                  >
                    <span>{skill}</span>
                    <span style={{
                      padding: '2px 8px',
                      borderRadius: '4px',
                      background: active ? '#6366F1' : '#334155',
                      fontSize: '11px',
                      fontWeight: 700,
                      color: '#FFF',
                    }}>
                      {active ? '1' : '0'}
                    </span>
                  </button>
                );
              })}
            </div>

            {/* Vector representation */}
            <div style={{
              fontSize: '12px',
              color: 'var(--text-sub)',
              fontFamily: 'var(--font-mono)',
              background: 'rgba(15, 23, 42, 0.4)',
              padding: '8px 12px',
              borderRadius: '6px',
              border: '1px solid var(--border-color)',
            }}>
              Current Vector: [{annVector.join(', ')}]
            </div>

            {/* ── Prominent Analyze / Forward-Pass Button ── */}
            <button
              className="btn btn-primary"
              style={{
                width: '100%',
                padding: '14px',
                fontSize: '14px',
                fontWeight: 700,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                marginTop: '4px',
              }}
              onClick={() => handleAnnPredict(annVector)}
              disabled={annLoading}
            >
              <Sparkles size={18} />
              <span>{annLoading ? 'Forward-Propagating Layers...' : '⚡ Compute Neural Forward Pass'}</span>
            </button>
          </div>

          {/* Right: ANN Results */}
          <div className="glass-card" style={{ padding: '24px', display: 'flex', flexDirection: 'column' }}>
            <h3 style={{ fontSize: '18px', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <TrendingUp size={20} color="#34D399" />
              <span>ANN Career Probability Output</span>
            </h3>

            {annLoading ? (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '40px', gap: '14px', color: 'var(--text-muted)' }}>
                <RefreshCw size={28} className="animate-spin" color="#818CF8" />
                <span>Computing Dense(8→16→8→5) Softmax activations...</span>
              </div>
            ) : annError ? (
              <div style={{ padding: '20px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: '8px', textAlign: 'center' }}>
                <AlertTriangle size={24} color="#EF4444" style={{ margin: '0 auto 8px' }} />
                <div style={{ color: '#FCA5A5', fontWeight: 600, fontSize: '14px' }}>{annError}</div>
                <button
                  className="btn btn-secondary"
                  style={{ marginTop: '12px', fontSize: '12px' }}
                  onClick={() => handleAnnPredict(annVector)}
                >
                  Retry Prediction
                </button>
              </div>
            ) : annResult?.predictions ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                {/* Winner Callout */}
                {annResult.top_career && (
                  <div style={{
                    padding: '16px 20px',
                    borderRadius: 'var(--radius-sm)',
                    background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(56, 189, 248, 0.15))',
                    border: '1px solid rgba(99, 102, 241, 0.4)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                  }}>
                    <div>
                      <div style={{ fontSize: '11px', color: '#38BDF8', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                        Top Neural Classification Match
                      </div>
                      <div style={{ fontSize: '20px', fontWeight: 800, color: '#FFF', marginTop: '2px' }}>
                        {annResult.top_career}
                      </div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: '24px', fontWeight: 900, color: '#34D399' }}>
                        {Math.round((annResult.confidence || annResult.predictions[0]?.Confidence || 0) * 100)}%
                      </div>
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Confidence</div>
                    </div>
                  </div>
                )}

                {/* Ranked Probability Bars */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-muted)' }}>
                    All Softmax Output Probabilities:
                  </div>
                  {annResult.predictions.map((p, idx) => {
                    const conf = p.Confidence ?? p.Probability ?? 0;
                    const pct = Math.round(conf * 100);
                    return (
                      <div
                        key={p.Career}
                        style={{
                          padding: '12px 16px',
                          borderRadius: 'var(--radius-sm)',
                          background: idx === 0 ? 'rgba(99, 102, 241, 0.12)' : 'rgba(15, 23, 42, 0.4)',
                          border: `1px solid ${idx === 0 ? 'rgba(99, 102, 241, 0.4)' : 'var(--border-color)'}`,
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', alignItems: 'center' }}>
                          <span style={{ fontWeight: 700, fontSize: '14px', color: idx === 0 ? '#FFF' : 'var(--text-main)' }}>
                            {idx + 1}. {p.Career}
                          </span>
                          <span style={{ fontWeight: 700, fontSize: '13px', color: idx === 0 ? '#38BDF8' : 'var(--text-muted)' }}>
                            {pct}%
                          </span>
                        </div>
                        <div style={{ width: '100%', height: '8px', background: '#334155', borderRadius: '4px', overflow: 'hidden' }}>
                          <div style={{
                            height: '100%',
                            width: `${pct}%`,
                            background: idx === 0 ? 'linear-gradient(90deg, #6366F1, #38BDF8)' : '#64748B',
                            transition: 'width 0.4s ease',
                          }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '40px', gap: '12px', color: 'var(--text-muted)' }}>
                <Layers size={36} color="#6366F1" style={{ opacity: 0.7 }} />
                <span>Toggle skills and click <strong>"Compute Neural Forward Pass"</strong> to view real-time career classification.</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ────────────────────────────────────────────────────────── */}
      {/* ── TAB 2: CNN Digit Classifier ── */}
      {/* ────────────────────────────────────────────────────────── */}
      {activeSubTab === 'cnn' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 1.1fr', gap: '24px' }}>
          {/* Left: Drawing Canvas / Image Upload */}
          <div className="glass-card" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div>
              <h3 style={{ fontSize: '18px', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Eye size={20} color="#818CF8" />
                <span>MNIST Handwritten Digit CNN</span>
              </h3>
              <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
                Draw a digit (0-9) on the interactive canvas or upload an image to run 2D convolution feature extraction.
              </p>
            </div>

            {/* Mode Switcher */}
            <div style={{ display: 'flex', gap: '8px', background: 'rgba(15, 23, 42, 0.5)', padding: '3px', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
              <button
                onClick={() => setCnnMode('draw')}
                style={{
                  flex: 1,
                  padding: '7px',
                  borderRadius: '4px',
                  border: 'none',
                  background: cnnMode === 'draw' ? 'var(--primary)' : 'transparent',
                  color: cnnMode === 'draw' ? '#FFF' : 'var(--text-muted)',
                  fontSize: '12px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '6px',
                }}
              >
                <Edit3 size={14} />
                <span>Draw on Canvas</span>
              </button>
              <button
                onClick={() => setCnnMode('upload')}
                style={{
                  flex: 1,
                  padding: '7px',
                  borderRadius: '4px',
                  border: 'none',
                  background: cnnMode === 'upload' ? 'var(--primary)' : 'transparent',
                  color: cnnMode === 'upload' ? '#FFF' : 'var(--text-muted)',
                  fontSize: '12px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '6px',
                }}
              >
                <Upload size={14} />
                <span>Upload Image File</span>
              </button>
            </div>

            {cnnMode === 'draw' ? (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px' }}>
                {/* Drawing Canvas */}
                <div style={{
                  position: 'relative',
                  border: '2px solid #6366F1',
                  borderRadius: '12px',
                  overflow: 'hidden',
                  background: '#000',
                  boxShadow: '0 0 20px rgba(99, 102, 241, 0.25)',
                }}>
                  <canvas
                    ref={canvasRef}
                    width={200}
                    height={200}
                    style={{ display: 'block', cursor: 'crosshair', touchAction: 'none' }}
                    onMouseDown={startDrawing}
                    onMouseMove={draw}
                    onMouseUp={stopDrawing}
                    onMouseLeave={stopDrawing}
                    onTouchStart={startDrawing}
                    onTouchMove={draw}
                    onTouchEnd={stopDrawing}
                  />
                </div>

                {/* Canvas Controls & Benchmark Samples */}
                <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center' }}>
                  <button
                    onClick={clearCanvas}
                    style={{
                      padding: '6px 12px',
                      borderRadius: '6px',
                      background: 'rgba(239, 68, 68, 0.15)',
                      border: '1px solid rgba(239, 68, 68, 0.3)',
                      color: '#FCA5A5',
                      fontSize: '12px',
                      fontWeight: 600,
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px',
                    }}
                  >
                    <Eraser size={14} />
                    <span>Clear Canvas</span>
                  </button>

                  {/* Benchmark Presets */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginRight: '4px' }}>Samples:</span>
                    {[7, 8, 2, 0, 4].map((d) => (
                      <button
                        key={d}
                        onClick={() => drawSampleDigit(d)}
                        style={{
                          padding: '4px 8px',
                          borderRadius: '4px',
                          background: 'rgba(30, 41, 59, 0.8)',
                          border: '1px solid var(--border-color)',
                          color: '#E2E8F0',
                          fontSize: '11px',
                          fontWeight: 700,
                          cursor: 'pointer',
                        }}
                      >
                        {d}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div>
                <div
                  style={{
                    border: '2px dashed var(--border-color)',
                    borderRadius: 'var(--radius-sm)',
                    padding: '24px',
                    textAlign: 'center',
                    background: 'rgba(15, 23, 42, 0.4)',
                    cursor: 'pointer',
                    marginBottom: '12px',
                  }}
                  onClick={() => document.getElementById('digit-upload-input')?.click()}
                >
                  <input
                    id="digit-upload-input"
                    type="file"
                    accept="image/*"
                    style={{ display: 'none' }}
                    onChange={handleDigitUpload}
                  />
                  {digitPreview ? (
                    <img
                      src={digitPreview}
                      alt="Digit preview"
                      style={{ width: '90px', height: '90px', objectFit: 'contain', margin: '0 auto 10px', borderRadius: '8px', border: '1px solid #6366F1' }}
                    />
                  ) : (
                    <Upload size={32} color="#818CF8" style={{ margin: '0 auto 10px' }} />
                  )}
                  <div style={{ fontSize: '13px', fontWeight: 600, color: '#FFF' }}>
                    {digitFile ? digitFile.name : 'Select or Drop Digit Image (PNG/JPG)'}
                  </div>
                </div>
              </div>
            )}

            {/* Classify Button */}
            <button
              className="btn btn-primary"
              style={{ width: '100%', padding: '12px', fontSize: '14px', fontWeight: 700 }}
              onClick={handleDigitPredict}
              disabled={digitLoading || (cnnMode === 'upload' && !digitFile)}
            >
              <Sparkles size={16} />
              <span>{digitLoading ? 'Extracting Convolutions...' : 'Classify Handwritten Digit'}</span>
            </button>
          </div>

          {/* Right: CNN Result Output */}
          <div className="glass-card" style={{ padding: '24px', display: 'flex', flexDirection: 'column' }}>
            <h3 style={{ fontSize: '18px', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Layers size={20} color="#34D399" />
              <span>Vision CNN Classification Output</span>
            </h3>

            {digitLoading ? (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '40px', gap: '14px', color: 'var(--text-muted)' }}>
                <RefreshCw size={28} className="animate-spin" color="#818CF8" />
                <span>Running Conv2D(32) → MaxPool → Conv2D(64) → Dense...</span>
              </div>
            ) : digitError ? (
              <div style={{ padding: '20px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: '8px', textAlign: 'center' }}>
                <AlertTriangle size={24} color="#EF4444" style={{ margin: '0 auto 8px' }} />
                <div style={{ color: '#FCA5A5', fontWeight: 600, fontSize: '14px' }}>{digitError}</div>
                <button
                  className="btn btn-secondary"
                  style={{ marginTop: '12px', fontSize: '12px' }}
                  onClick={handleDigitPredict}
                >
                  Retry Classification
                </button>
              </div>
            ) : digitResult ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                {/* Predicted Class Card */}
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '24px',
                  padding: '20px',
                  background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(168, 85, 247, 0.15))',
                  border: '1px solid rgba(99, 102, 241, 0.4)',
                  borderRadius: 'var(--radius-sm)',
                }}>
                  <div style={{
                    width: '80px',
                    height: '80px',
                    borderRadius: '12px',
                    background: 'linear-gradient(135deg, #6366F1, #A855F7)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '48px',
                    fontWeight: 900,
                    color: '#FFF',
                    boxShadow: '0 4px 14px rgba(99, 102, 241, 0.4)',
                  }}>
                    {digitResult.predicted_digit}
                  </div>
                  <div>
                    <div style={{ fontSize: '12px', color: '#38BDF8', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                      PREDICTED DIGIT CLASS
                    </div>
                    <div style={{ fontSize: '24px', fontWeight: 800, color: '#FFF' }}>
                      Digit "{digitResult.predicted_digit}"
                    </div>
                    <div style={{ fontSize: '14px', color: '#34D399', fontWeight: 600, marginTop: '2px' }}>
                      Confidence: {(digitResult.confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                {/* 10-Class Softmax Probabilities Grid */}
                <div>
                  <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '10px' }}>
                    10-Class Softmax Probability Distribution:
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '8px' }}>
                    {digitResult.probabilities.map((prob, d) => {
                      const isWinner = d === digitResult.predicted_digit;
                      const pct = (prob * 100).toFixed(1);
                      return (
                        <div
                          key={d}
                          style={{
                            padding: '10px 8px',
                            background: isWinner ? 'rgba(99, 102, 241, 0.25)' : 'rgba(15, 23, 42, 0.5)',
                            borderRadius: '6px',
                            textAlign: 'center',
                            border: isWinner ? '1px solid #6366F1' : '1px solid var(--border-color)',
                            transition: 'all 0.2s ease',
                          }}
                        >
                          <div style={{ fontSize: '14px', fontWeight: 800, color: isWinner ? '#FFF' : 'var(--text-muted)' }}>
                            {d}
                          </div>
                          <div style={{ fontSize: '11px', fontWeight: 600, color: isWinner ? '#38BDF8' : 'var(--text-sub)', marginTop: '2px' }}>
                            {pct}%
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Architecture Info */}
                <div style={{ fontSize: '12px', color: 'var(--text-sub)', borderTop: '1px solid var(--border-color)', paddingTop: '12px' }}>
                  🧠 <strong>Pipeline:</strong> 28×28 Grayscale Tensor → Conv2D(3×3×32, ReLU) → MaxPool(2×2) → Conv2D(3×3×64, ReLU) → MaxPool(2×2) → Flatten(1600) → Dense(64) → Dense(10, Softmax).
                </div>
              </div>
            ) : (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '40px', gap: '12px', color: 'var(--text-muted)' }}>
                <Eye size={36} color="#6366F1" style={{ opacity: 0.7 }} />
                <span>Draw or upload a handwritten digit on the left and click <strong>"Classify Handwritten Digit"</strong>.</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ────────────────────────────────────────────────────────── */}
      {/* ── TAB 3: Sentiment RNN vs LSTM ── */}
      {/* ────────────────────────────────────────────────────────── */}
      {activeSubTab === 'sentiment' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.2fr', gap: '24px' }}>
          {/* Left: Input Response & Prompts */}
          <div className="glass-card" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div>
              <div style={{
                padding: '12px 16px',
                background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.25), rgba(236, 72, 153, 0.2))',
                border: '1px solid rgba(99, 102, 241, 0.4)',
                borderRadius: '8px',
                marginBottom: '14px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: '12px',
              }}>
                <div>
                  <div style={{ fontSize: '13px', fontWeight: 800, color: '#FFF' }}>
                    🎙️ Need Full Interactive Mock Interviews?
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                    Practice role-specific questions with timer, voice dictation & gold-standard answers.
                  </div>
                </div>
                {onNavigateToInterviewCoach && (
                  <button
                    className="btn btn-primary"
                    style={{ fontSize: '12px', padding: '6px 12px', whiteSpace: 'nowrap' }}
                    onClick={onNavigateToInterviewCoach}
                  >
                    <span>Launch Studio →</span>
                  </button>
                )}
              </div>

              <h3 style={{ fontSize: '18px', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <MessageSquare size={20} color="#818CF8" />
                <span>Interview Tone & Sentiment Coach</span>
              </h3>
              <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
                Evaluate interview responses, compare Simple RNN vs Bidirectional LSTM sentiment activations, and view actionable coaching feedback.
              </p>
            </div>

            {/* Sample Prompts */}
            <div>
              <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-sub)', marginBottom: '8px' }}>
                Try Sample Interview Prompts:
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {samplePrompts.map((p) => (
                  <button
                    key={p.title}
                    onClick={() => {
                      setInterviewText(p.text);
                      handleSentimentAnalyze(p.text);
                    }}
                    style={{
                      padding: '8px 12px',
                      borderRadius: '6px',
                      background: 'rgba(30, 41, 59, 0.6)',
                      border: '1px solid var(--border-color)',
                      color: '#E2E8F0',
                      fontSize: '12px',
                      cursor: 'pointer',
                      textAlign: 'left',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      transition: 'all 0.15s ease',
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.borderColor = '#6366F1')}
                    onMouseLeave={(e) => (e.currentTarget.style.borderColor = 'var(--border-color)')}
                  >
                    <span>{p.title}</span>
                    <ArrowRight size={12} color="#818CF8" />
                  </button>
                ))}
              </div>
            </div>

            {/* Textarea */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px', fontSize: '12px', color: 'var(--text-muted)' }}>
                <span>Response Text:</span>
                <span>{interviewText.trim().split(/\s+/).filter(Boolean).length} words</span>
              </div>
              <textarea
                className="form-textarea"
                rows={5}
                value={interviewText}
                onChange={(e) => setInterviewText(e.target.value)}
                placeholder="Paste interview response or cover letter paragraph here..."
                style={{ width: '100%' }}
              />
            </div>

            <button
              className="btn btn-primary"
              style={{ width: '100%', padding: '12px', fontSize: '14px', fontWeight: 700 }}
              onClick={() => handleSentimentAnalyze(interviewText)}
              disabled={sentimentLoading || !interviewText.trim()}
            >
              <Sparkles size={16} />
              <span>{sentimentLoading ? 'Evaluating Recurrent Gates...' : '🎙️ Analyze Tone & Compare Models'}</span>
            </button>
          </div>

          {/* Right: RNN vs LSTM Comparison & Coaching Output */}
          <div className="glass-card" style={{ padding: '24px', display: 'flex', flexDirection: 'column' }}>
            <h3 style={{ fontSize: '18px', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <TrendingUp size={20} color="#34D399" />
              <span>Model Comparison & Tone Diagnostics</span>
            </h3>

            {sentimentLoading ? (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '40px', gap: '14px', color: 'var(--text-muted)' }}>
                <RefreshCw size={28} className="animate-spin" color="#818CF8" />
                <span>Computing SimpleRNN and LSTM hidden recurrent states...</span>
              </div>
            ) : sentimentError ? (
              <div style={{ padding: '20px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: '8px', textAlign: 'center' }}>
                <AlertTriangle size={24} color="#EF4444" style={{ margin: '0 auto 8px' }} />
                <div style={{ color: '#FCA5A5', fontWeight: 600, fontSize: '14px' }}>{sentimentError}</div>
                <button
                  className="btn btn-secondary"
                  style={{ marginTop: '12px', fontSize: '12px' }}
                  onClick={() => handleSentimentAnalyze(interviewText)}
                >
                  Retry Analysis
                </button>
              </div>
            ) : sentimentResult ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                {/* 1. Side-by-side LSTM vs RNN Cards */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                  {/* LSTM Card */}
                  <div style={{
                    padding: '16px',
                    background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(56, 189, 248, 0.1))',
                    border: '1px solid rgba(99, 102, 241, 0.4)',
                    borderRadius: 'var(--radius-sm)',
                  }}>
                    <div style={{ fontSize: '11px', color: '#38BDF8', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                      BIDIRECTIONAL LSTM
                    </div>
                    <div style={{ fontSize: '17px', fontWeight: 800, marginTop: '4px', color: '#FFF' }}>
                      {sentimentResult.comparison.lstm.sentiment}
                    </div>
                    <div style={{ fontSize: '13px', color: '#34D399', fontWeight: 600, marginTop: '2px' }}>
                      Confidence: {(sentimentResult.comparison.lstm.confidence * 100).toFixed(1)}%
                    </div>
                    <div style={{ fontSize: '11px', color: 'var(--text-sub)', marginTop: '8px', lineHeight: 1.4 }}>
                      ⚡ Uses Input, Forget & Output memory gates to preserve long-term context without vanishing gradients.
                    </div>
                  </div>

                  {/* RNN Card */}
                  <div style={{
                    padding: '16px',
                    background: 'rgba(30, 41, 59, 0.5)',
                    border: '1px solid var(--border-color)',
                    borderRadius: 'var(--radius-sm)',
                  }}>
                    <div style={{ fontSize: '11px', color: 'var(--text-sub)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                      SIMPLE RNN
                    </div>
                    <div style={{ fontSize: '17px', fontWeight: 800, marginTop: '4px', color: '#CBD5E1' }}>
                      {sentimentResult.comparison.rnn.sentiment}
                    </div>
                    <div style={{ fontSize: '13px', color: 'var(--text-muted)', fontWeight: 600, marginTop: '2px' }}>
                      Confidence: {(sentimentResult.comparison.rnn.confidence * 100).toFixed(1)}%
                    </div>
                    <div style={{ fontSize: '11px', color: 'var(--text-sub)', marginTop: '8px', lineHeight: 1.4 }}>
                      ⚠️ Basic recurrent hidden loop; susceptible to vanishing gradients across longer sentences.
                    </div>
                  </div>
                </div>

                {/* 2. Impact Score & Tone Badge */}
                {sentimentResult.coach && (
                  <div style={{
                    padding: '14px 18px',
                    background: 'rgba(15, 23, 42, 0.6)',
                    border: '1px solid var(--border-color)',
                    borderRadius: 'var(--radius-sm)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                  }}>
                    <div>
                      <div style={{ fontSize: '11px', color: 'var(--text-sub)', fontWeight: 700 }}>OVERALL IMPACT SCORE</div>
                      <div style={{ fontSize: '20px', fontWeight: 800, color: sentimentResult.coach.tone_color || '#10B981', marginTop: '2px' }}>
                        {sentimentResult.coach.tone_label || 'Professional & Clear'}
                      </div>
                      <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '2px' }}>
                        {sentimentResult.coach.tone_badge}
                      </div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: '28px', fontWeight: 900, color: sentimentResult.coach.tone_color || '#10B981' }}>
                        {sentimentResult.coach.impact_score ?? 80}
                        <span style={{ fontSize: '14px', color: 'var(--text-muted)' }}>/100</span>
                      </div>
                    </div>
                  </div>
                )}

                {/* 3. STAR Framework Breakdown */}
                {sentimentResult.coach?.star_breakdown && (
                  <div style={{ padding: '12px 16px', background: 'rgba(15, 23, 42, 0.4)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)' }}>
                    <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-muted)', marginBottom: '8px' }}>
                      ⭐ STAR Framework Alignment:
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px' }}>
                      {Object.entries(sentimentResult.coach.star_breakdown).map(([key, val]) => (
                        <div
                          key={key}
                          style={{
                            padding: '6px 8px',
                            background: val ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.1)',
                            border: `1px solid ${val ? 'rgba(16, 185, 129, 0.4)' : 'rgba(239, 68, 68, 0.3)'}`,
                            borderRadius: '4px',
                            textAlign: 'center',
                            fontSize: '11px',
                            fontWeight: 700,
                            color: val ? '#34D399' : '#FCA5A5',
                          }}
                        >
                          {val ? '✓' : '✗'} {key}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* 4. Action Verbs & Weak Phrasing */}
                {sentimentResult.coach && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {sentimentResult.coach.found_action_verbs && sentimentResult.coach.found_action_verbs.length > 0 && (
                      <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                        <strong style={{ color: '#34D399' }}>Strong Action Verbs: </strong>
                        {sentimentResult.coach.found_action_verbs.map((v) => (
                          <span key={v} className="badge badge-primary" style={{ marginRight: '4px', fontSize: '11px' }}>
                            {v}
                          </span>
                        ))}
                      </div>
                    )}

                    {sentimentResult.coach.found_weak_phrases && sentimentResult.coach.found_weak_phrases.length > 0 && (
                      <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                        <strong style={{ color: '#F59E0B' }}>Weak Phrases to Upgrade: </strong>
                        {sentimentResult.coach.found_weak_phrases.map((item, idx) => (
                          <span key={idx} style={{ color: '#FCD34D', marginRight: '8px' }}>
                            "{item.phrase}" → <em>{item.fix}</em>
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* 5. Coaching Feedback List */}
                {sentimentResult.coach?.tone_feedback && sentimentResult.coach.tone_feedback.length > 0 && (
                  <div style={{ padding: '14px', background: 'rgba(15, 23, 42, 0.5)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)' }}>
                    <div style={{ fontSize: '13px', fontWeight: 700, color: '#38BDF8', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <CheckCircle2 size={16} color="#38BDF8" />
                      <span>Coaching Feedback & Actionable Recommendations:</span>
                    </div>
                    <ul style={{ paddingLeft: '18px', fontSize: '12px', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '6px', margin: 0 }}>
                      {sentimentResult.coach.tone_feedback.map((tip, idx) => (
                        <li key={idx}>{tip}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ) : (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '40px', gap: '12px', color: 'var(--text-muted)' }}>
                <MessageSquare size={36} color="#6366F1" style={{ opacity: 0.7 }} />
                <span>Click <strong>"Analyze Tone & Compare Models"</strong> to observe recurrent neural network evaluation.</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
