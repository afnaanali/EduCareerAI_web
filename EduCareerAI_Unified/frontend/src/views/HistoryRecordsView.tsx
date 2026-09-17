import React, { useState, useEffect } from 'react';
import {
  History,
  Trash2,
  Calendar,
  Award,
  FileText,
  User as UserIcon,
  Lock,
} from 'lucide-react';
import { api } from '../api';
import type { User } from '../types';

interface HistoryRecordsViewProps {
  user: User | null;
  onOpenAuth: () => void;
}

export const HistoryRecordsView: React.FC<HistoryRecordsViewProps> = ({ user, onOpenAuth }) => {
  const [history, setHistory] = useState<{ assessments: any[]; resume_scans: any[] }>({
    assessments: [],
    resume_scans: [],
  });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (user?.id) {
      loadHistory();
    } else {
      setHistory({ assessments: [], resume_scans: [] });
    }
  }, [user?.id]);

  const loadHistory = async () => {
    if (!user?.id) return;
    setLoading(true);
    try {
      const res = await api.getUserHistory(user.id);
      setHistory(res);
    } catch (err) {
      console.error('Failed to load user history', err);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteAssessment = async (id: number) => {
    if (!user?.id) return;
    try {
      await api.deleteAssessment(user.id, id);
      setHistory((prev) => ({
        ...prev,
        assessments: prev.assessments.filter((a) => a.id !== id),
      }));
    } catch (err) {
      console.error('Delete assessment error', err);
    }
  };

  const handleDeleteResumeScan = async (id: number) => {
    if (!user?.id) return;
    try {
      await api.deleteResumeScan(user.id, id);
      setHistory((prev) => ({
        ...prev,
        resume_scans: prev.resume_scans.filter((r) => r.id !== id),
      }));
    } catch (err) {
      console.error('Delete scan error', err);
    }
  };

  if (!user) {
    return (
      <div className="glass-card animate-fade-in" style={{
        padding: '60px 24px',
        textAlign: 'center',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        maxWidth: '540px',
        margin: '40px auto',
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
          <Lock size={32} color="#818CF8" />
        </div>
        <h2 style={{ fontSize: '22px', marginBottom: '8px' }}>User Account Required</h2>
        <p style={{ fontSize: '14px', color: 'var(--text-muted)', lineHeight: 1.6, marginBottom: '24px' }}>
          Create a free account or sign in to persist your career assessments, ATS resume scans, and customized roadmaps to the database.
        </p>
        <button className="btn btn-primary" onClick={onOpenAuth}>
          <UserIcon size={16} />
          <span>Sign In / Register</span>
        </button>
      </div>
    );
  }

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Header */}
      <div>
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
          <span className="badge badge-primary">
            <History size={12} />
            Persistent SQLite Records
          </span>
        </div>
        <h1 style={{ fontSize: '28px', fontWeight: 800 }}>
          User Records & <span className="gradient-text">History</span>
        </h1>
        <p style={{ fontSize: '14px', color: 'var(--text-muted)' }}>
          Review and manage all past career assessments and ATS diagnostic reports for <b>@{user.username}</b>.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
        {/* Assessment History */}
        <div className="glass-card" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '18px', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Award size={20} color="#818CF8" />
            <span>Saved Career Assessments ({history.assessments.length})</span>
          </h3>

          {loading ? (
            <div style={{ color: 'var(--text-muted)', fontSize: '13px' }}>Loading records...</div>
          ) : history.assessments.length === 0 ? (
            <div style={{ color: 'var(--text-muted)', fontSize: '13px', padding: '20px 0' }}>
              No career assessments saved yet. Complete an assessment in the Career Navigator to record results.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {history.assessments.map((a) => (
                <div
                  key={a.id}
                  style={{
                    padding: '14px',
                    background: 'rgba(15, 23, 42, 0.4)',
                    border: '1px solid var(--border-color)',
                    borderRadius: 'var(--radius-sm)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                  }}
                >
                  <div>
                    <div style={{ fontSize: '15px', fontWeight: 700, color: '#FFF' }}>
                      {a.top_career}
                    </div>
                    <div style={{ fontSize: '12px', color: 'var(--accent-cyan)', marginTop: '2px' }}>
                      Top Degree: {a.top_course || 'N/A'}
                    </div>
                    <div style={{ fontSize: '11px', color: 'var(--text-sub)', marginTop: '4px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <Calendar size={12} />
                      <span>{a.created_at}</span>
                    </div>
                  </div>

                  <button
                    className="btn btn-ghost"
                    style={{ color: '#FB7185', padding: '6px' }}
                    onClick={() => handleDeleteAssessment(a.id)}
                    title="Delete record"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Resume Scans History */}
        <div className="glass-card" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '18px', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <FileText size={20} color="#10B981" />
            <span>Saved ATS Resume Scans ({history.resume_scans.length})</span>
          </h3>

          {loading ? (
            <div style={{ color: 'var(--text-muted)', fontSize: '13px' }}>Loading scans...</div>
          ) : history.resume_scans.length === 0 ? (
            <div style={{ color: 'var(--text-muted)', fontSize: '13px', padding: '20px 0' }}>
              No resume scans recorded yet. Scan a resume in the Resume Studio.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {history.resume_scans.map((r) => (
                <div
                  key={r.id}
                  style={{
                    padding: '14px',
                    background: 'rgba(15, 23, 42, 0.4)',
                    border: '1px solid var(--border-color)',
                    borderRadius: 'var(--radius-sm)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                  }}
                >
                  <div>
                    <div style={{ fontSize: '15px', fontWeight: 700, color: '#FFF' }}>
                      {r.resume_name}
                    </div>
                    <div style={{ fontSize: '12px', color: '#34D399', marginTop: '2px' }}>
                      ATS Score: <b>{r.ats_score} / 100</b> • Target: {r.target_job}
                    </div>
                    <div style={{ fontSize: '11px', color: 'var(--text-sub)', marginTop: '4px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <Calendar size={12} />
                      <span>{r.created_at}</span>
                    </div>
                  </div>

                  <button
                    className="btn btn-ghost"
                    style={{ color: '#FB7185', padding: '6px' }}
                    onClick={() => handleDeleteResumeScan(r.id)}
                    title="Delete scan"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
