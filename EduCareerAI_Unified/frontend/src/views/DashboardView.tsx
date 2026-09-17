import React from 'react';
import {
  Compass,
  Map,
  FileText,
  Bot,
  Brain,
  Cpu,
  ShieldCheck,
  TrendingUp,
  Zap,
  CheckCircle2,
  ArrowRight,
  Mic,
} from 'lucide-react';

import type { User } from '../types';

interface DashboardViewProps {
  user?: User | null;
  onSelectTab: (tab: string) => void;
  activeContext?: {
    topCareer?: string;
    resumeName?: string;
  };
}

export const DashboardView: React.FC<DashboardViewProps> = ({ user, onSelectTab, activeContext }) => {
  const features = [
    {
      id: 'career_nav',
      title: 'Career & Course AI',
      desc: 'HistGB Machine Learning model predicts your top careers & degree pathways from academic grades, aptitudes, and 16 hobby indicators.',
      icon: Compass,
      color: '#6366F1',
      badge: 'HistGB ML',
    },
    {
      id: 'roadmap',
      title: 'Interactive Roadmaps',
      desc: 'Structured 4-phase milestone learning roadmaps for 30+ tech disciplines with capstone projects, certifications, and resources.',
      icon: Map,
      color: '#06B6D4',
      badge: '30+ Pathways',
    },
    {
      id: 'resume',
      title: 'Resume & ATS Studio',
      desc: '9-vector ATS diagnostics, automated skill gap detector, role-specific job matching, and actionable keyword optimization tips.',
      icon: FileText,
      color: '#10B981',
      badge: '9-Vector ATS',
    },
    {
      id: 'interview_coach',
      title: 'Interview Preparation Coach',
      desc: 'Practice role-specific mock interviews with live STAR rubric scoring, recurrent neural sentiment analysis, voice dictation, and expert gold-standard answers.',
      icon: Mic,
      color: '#EC4899',
      badge: 'STAR Method',
    },
    {
      id: 'chat',
      title: 'AI Counselor & RAG',
      desc: 'Educational counselor and technical mentor with document-aware RAG search across uploaded syllabuses and textbooks.',
      icon: Bot,
      color: '#A855F7',
      badge: 'RAG Studio',
    },
    {
      id: 'lab',
      title: 'Deep Learning Lab',
      desc: 'Explore Artificial Neural Networks (8-skill ANN), MNIST Vision CNN, and Recurrent Neural Networks (RNN vs LSTM sentiment engine).',
      icon: Brain,
      color: '#F59E0B',
      badge: 'PyTorch / NumPy',
    },
    {
      id: 'history',
      title: 'Saved Records & History',
      desc: 'Persistent SQLite records of your past career assessments, resume ATS scans, and conversation transcripts.',
      icon: ShieldCheck,
      color: '#EC4899',
      badge: 'SQLite',
    },
  ];

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
      {/* Hero Banner */}
      <div className="glass-card" style={{
        padding: '36px',
        background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%)',
        border: '1px solid rgba(99, 102, 241, 0.25)',
        position: 'relative',
        overflow: 'hidden',
      }}>
        <div style={{
          position: 'absolute',
          top: '-40px',
          right: '-40px',
          width: '240px',
          height: '240px',
          background: 'radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, transparent 70%)',
          borderRadius: '50%',
          filter: 'blur(30px)',
        }} />
        
        <div style={{ maxWidth: '800px', position: 'relative', zIndex: 1 }}>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', marginBottom: '12px', flexWrap: 'wrap' }}>
            <span className="badge badge-primary">
              <Zap size={13} />
              Production Decoupled Platform
            </span>
            <span className="badge badge-success">
              <CheckCircle2 size={13} />
              24/7 Always Active
            </span>
            {user && (
              <span className="badge" style={{ background: 'rgba(56, 189, 248, 0.15)', border: '1px solid rgba(56, 189, 248, 0.3)', color: '#38BDF8' }}>
                👤 Logged In: @{user.username}
              </span>
            )}
          </div>

          <h1 style={{ fontSize: '36px', fontWeight: 800, marginBottom: '14px', letterSpacing: '-0.5px' }}>
            {user ? (
              <>Welcome back, <span className="gradient-text">{user.full_name || user.username}</span>!</>
            ) : (
              <>Unified Education & <span className="gradient-text">Career Intelligence</span></>
            )}
          </h1>

          <p style={{ fontSize: '16px', color: 'var(--text-muted)', lineHeight: 1.6, marginBottom: '24px' }}>
            {user ? (
              activeContext?.topCareer ? (
                <>Your personalized active career target is <b style={{ color: '#38BDF8' }}>{activeContext.topCareer}</b>. Track your roadmaps, evaluate resume ATS scores, and practice mock interviews.</>
              ) : (
                <>Ready to assess your personalized career pathway? Complete an AI assessment to unlock your custom roadmap, degree recommendations, and tailored interview coach.</>
              )
            ) : (
              <>Seamlessly bridging machine learning career ranking, synchronized degree pathways, ATS resume diagnostics, conversational AI mentorship, and neural deep learning models.</>
            )}
          </p>

          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
            <button className="btn btn-primary" onClick={() => onSelectTab('career_nav')}>
              <span>{activeContext?.topCareer ? 'Update Assessment' : 'Start Career Assessment'}</span>
              <ArrowRight size={16} />
            </button>
            <button className="btn btn-secondary" onClick={() => onSelectTab('roadmap')}>
              <Map size={16} />
              <span>{activeContext?.topCareer ? `Explore ${activeContext.topCareer} Roadmap` : 'Explore Roadmaps'}</span>
            </button>
            <button className="btn btn-secondary" onClick={() => onSelectTab('resume')}>
              <FileText size={16} />
              <span>Analyze Resume</span>
            </button>
            <button className="btn btn-secondary" onClick={() => onSelectTab('chat')}>
              <Bot size={16} />
              <span>Ask AI Mentor</span>
            </button>
          </div>
        </div>
      </div>

      {/* Metrics Row */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
        gap: '16px',
      }}>
        <div className="glass-card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
            <span style={{ fontSize: '13px', color: 'var(--text-muted)', fontWeight: 600 }}>Trained AI Engines</span>
            <Cpu size={20} color="#818CF8" />
          </div>
          <div style={{ fontSize: '26px', fontWeight: 800, color: '#FFF' }}>6 Models</div>
          <div style={{ fontSize: '12px', color: 'var(--accent-cyan)', marginTop: '4px' }}>HistGB • MultiOutput • DL</div>
        </div>

        <div className="glass-card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
            <span style={{ fontSize: '13px', color: 'var(--text-muted)', fontWeight: 600 }}>Career Disciplines</span>
            <Map size={20} color="#34D399" />
          </div>
          <div style={{ fontSize: '26px', fontWeight: 800, color: '#FFF' }}>30+ Pathways</div>
          <div style={{ fontSize: '12px', color: '#34D399', marginTop: '4px' }}>Multi-phase roadmaps</div>
        </div>

        <div className="glass-card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
            <span style={{ fontSize: '13px', color: 'var(--text-muted)', fontWeight: 600 }}>ATS Diagnostic Vectors</span>
            <FileText size={20} color="#FBBF24" />
          </div>
          <div style={{ fontSize: '26px', fontWeight: 800, color: '#FFF' }}>9 Categories</div>
          <div style={{ fontSize: '12px', color: '#FBBF24', marginTop: '4px' }}>Weighted scoring engine</div>
        </div>

        <div className="glass-card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
            <span style={{ fontSize: '13px', color: 'var(--text-muted)', fontWeight: 600 }}>System Availability</span>
            <TrendingUp size={20} color="#38BDF8" />
          </div>
          <div style={{ fontSize: '26px', fontWeight: 800, color: '#FFF' }}>24/7 Active</div>
          <div style={{ fontSize: '12px', color: '#38BDF8', marginTop: '4px' }}>Vercel CDN + FastAPI Render</div>
        </div>
      </div>

      {/* Feature Modules Grid */}
      <div>
        <h2 style={{ fontSize: '20px', marginBottom: '16px', fontWeight: 700 }}>
          Explore Platform Modules
        </h2>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
          gap: '20px',
        }}>
          {features.map((f) => {
            const Icon = f.icon;
            return (
              <div
                key={f.id}
                className="glass-card"
                onClick={() => onSelectTab(f.id)}
                style={{
                  padding: '24px',
                  cursor: 'pointer',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'space-between',
                  minHeight: '190px',
                }}
              >
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
                    <div style={{
                      width: '42px',
                      height: '42px',
                      borderRadius: '10px',
                      background: `rgba(${parseInt(f.color.slice(1, 3), 16)}, ${parseInt(f.color.slice(3, 5), 16)}, ${parseInt(f.color.slice(5, 7), 16)}, 0.15)`,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}>
                      <Icon size={22} color={f.color} />
                    </div>
                    <span className="badge badge-primary" style={{ fontSize: '11px' }}>
                      {f.badge}
                    </span>
                  </div>
                  <h3 style={{ fontSize: '18px', marginBottom: '8px' }}>{f.title}</h3>
                  <p style={{ fontSize: '13px', color: 'var(--text-muted)', lineHeight: 1.5 }}>{f.desc}</p>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: f.color, fontSize: '13px', fontWeight: 600, marginTop: '16px' }}>
                  <span>Launch Module</span>
                  <ArrowRight size={14} />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
