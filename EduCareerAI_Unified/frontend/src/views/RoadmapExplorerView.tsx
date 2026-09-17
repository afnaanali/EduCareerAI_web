import React, { useState, useEffect } from 'react';
import {
  Map,
  Clock,
  CheckCircle,
  FolderGit2,
  Award,
  Layers,
} from 'lucide-react';
import { api } from '../api';
import type { RoadmapData, User } from '../types';

interface RoadmapExplorerViewProps {
  user?: User | null;
  initialCareer?: string;
}

const DEFAULT_ROADMAP_CAREERS = [
  'Data Analyst',
  'Data Scientist',
  'Machine Learning Engineer',
  'Software Engineer',
  'DevOps Engineer',
  'Cloud Architect',
  'Full Stack Developer',
  'Cybersecurity Analyst',
  'UI/UX Designer',
  'Product Manager',
  'AI Research Scientist',
  'Blockchain Developer',
];

export const RoadmapExplorerView: React.FC<RoadmapExplorerViewProps> = ({
  user,
  initialCareer = 'Data Analyst',
}) => {
  const [careerList, setCareerList] = useState<string[]>(DEFAULT_ROADMAP_CAREERS);
  const [selectedCareer, setSelectedCareer] = useState<string>(initialCareer);
  const [roadmap, setRoadmap] = useState<RoadmapData | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const loadCareers = async () => {
      try {
        const res = await api.listRoadmapCareers();
        if (res && res.careers && res.careers.length > 0) {
          setCareerList(res.careers);
        }
      } catch (err) {
        console.warn('Using default roadmap careers', err);
      }
    };
    loadCareers();
  }, []);

  useEffect(() => {
    if (initialCareer) {
      setSelectedCareer(initialCareer);
    }
  }, [initialCareer]);

  useEffect(() => {
    const loadRoadmap = async () => {
      if (!selectedCareer) return;
      setLoading(true);
      try {
        const res = await api.getRoadmap(selectedCareer);
        setRoadmap(res.roadmap);
      } catch (err) {
        console.error('Failed to load roadmap details', err);
      } finally {
        setLoading(false);
      }
    };
    loadRoadmap();
  }, [selectedCareer]);

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', marginBottom: '6px', flexWrap: 'wrap' }}>
            <span className="badge badge-primary">
              <Map size={12} />
              30+ Disciplines
            </span>
            {user && (
              <span className="badge" style={{ background: 'rgba(56, 189, 248, 0.15)', border: '1px solid rgba(56, 189, 248, 0.3)', color: '#38BDF8' }}>
                👤 {user.full_name || `@${user.username}`}'s Pathway
              </span>
            )}
          </div>
          <h1 style={{ fontSize: '28px', fontWeight: 800 }}>
            Interactive Learning <span className="gradient-text">Roadmaps</span>
          </h1>
          <p style={{ fontSize: '14px', color: 'var(--text-muted)' }}>
            Step-by-step milestone learning journeys, capstone projects, and industry certifications.
          </p>
        </div>

        {/* Career Selector */}
        <div style={{ minWidth: '260px' }}>
          <label className="input-label">Select Career Pathway</label>
          <select
            className="form-select"
            value={selectedCareer}
            onChange={(e) => setSelectedCareer(e.target.value)}
          >
            {careerList.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>
      </div>

      {loading ? (
        <div className="glass-card" style={{ padding: '60px', textAlign: 'center', color: 'var(--text-muted)' }}>
          Loading roadmap pathway...
        </div>
      ) : roadmap ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {/* Roadmap Overview Card */}
          <div className="glass-card" style={{
            padding: '24px',
            background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%)',
            border: '1px solid rgba(99, 102, 241, 0.3)',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px', marginBottom: '12px' }}>
              <h2 style={{ fontSize: '24px', fontWeight: 800 }}>{roadmap.title}</h2>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', background: 'rgba(6, 182, 212, 0.15)', border: '1px solid rgba(6, 182, 212, 0.3)', padding: '6px 14px', borderRadius: 'var(--radius-full)', color: '#38BDF8', fontSize: '13px', fontWeight: 600 }}>
                <Clock size={14} />
                <span>{roadmap.duration}</span>
              </div>
            </div>

            <p style={{ fontSize: '15px', color: 'var(--text-muted)', lineHeight: 1.6, marginBottom: '20px' }}>
              {roadmap.description}
            </p>

            {/* Key Skills */}
            <div>
              <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-sub)', marginBottom: '8px', textTransform: 'uppercase' }}>
                Core Competencies & Stack
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                {roadmap.key_skills.map((s) => (
                  <span key={s} className="badge badge-primary" style={{ padding: '6px 12px', fontSize: '13px' }}>
                    {s}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Phases Grid */}
          <div>
            <h3 style={{ fontSize: '18px', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Layers size={20} color="#818CF8" />
              <span>Milestone Curriculum Phases</span>
            </h3>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px' }}>
              {roadmap.phases.map((p, index) => (
                <div
                  key={index}
                  className="glass-card"
                  style={{
                    padding: '24px',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'space-between',
                    borderTop: `4px solid ${['#6366F1', '#06B6D4', '#10B981', '#A855F7'][index % 4]}`,
                  }}
                >
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '14px' }}>
                      <span style={{
                        width: '24px',
                        height: '24px',
                        borderRadius: '50%',
                        background: ['#6366F1', '#06B6D4', '#10B981', '#A855F7'][index % 4],
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '12px',
                        fontWeight: 700,
                        color: '#FFF',
                      }}>
                        {index + 1}
                      </span>
                      <h4 style={{ fontSize: '16px', fontWeight: 700 }}>{p.phase}</h4>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '20px' }}>
                      {p.topics.map((t, tidx) => (
                        <div key={tidx} style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', fontSize: '13px', color: 'var(--text-muted)' }}>
                          <CheckCircle size={14} color="#34D399" style={{ marginTop: '2px', flexShrink: 0 }} />
                          <span>{t}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Project Callout */}
                  <div style={{
                    padding: '12px 14px',
                    borderRadius: 'var(--radius-sm)',
                    background: 'rgba(15, 23, 42, 0.6)',
                    border: '1px solid var(--border-color)',
                  }}>
                    <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--accent-cyan)', display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
                      <FolderGit2 size={13} />
                      <span>CAPSTONE PROJECT</span>
                    </div>
                    <div style={{ fontSize: '13px', fontWeight: 600, color: '#FFF' }}>
                      {p.project}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Industry Certifications */}
          <div className="glass-card" style={{ padding: '24px' }}>
            <h3 style={{ fontSize: '18px', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Award size={20} color="#FBBF24" />
              <span>Recommended Industry Certifications</span>
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '12px' }}>
              {roadmap.certifications.map((cert, cidx) => (
                <div
                  key={cidx}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                    padding: '14px',
                    background: 'rgba(15, 23, 42, 0.5)',
                    border: '1px solid var(--border-color)',
                    borderRadius: 'var(--radius-sm)',
                  }}
                >
                  <Award size={22} color="#FBBF24" />
                  <span style={{ fontSize: '14px', fontWeight: 600, color: '#FFF' }}>{cert}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
};
