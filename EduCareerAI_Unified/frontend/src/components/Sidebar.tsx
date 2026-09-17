import React from 'react';
import {
  LayoutDashboard,
  Compass,
  Map,
  FileText,
  Bot,
  Brain,
  History,
  Mic,
} from 'lucide-react';

interface SidebarProps {
  activeTab: string;
  onSelectTab: (tab: string) => void;
  activeContext?: {
    topCareer?: string;
    resumeName?: string;
  };
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onSelectTab,
  activeContext,
}) => {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'career_nav', label: 'Career & Course AI', icon: Compass, badge: 'HistGB' },
    { id: 'roadmap', label: 'Learning Roadmaps', icon: Map },
    { id: 'resume', label: 'Resume & ATS Studio', icon: FileText, badge: '9-Vec' },
    { id: 'interview_coach', label: 'Interview Prep Coach', icon: Mic, badge: 'STAR' },
    { id: 'chat', label: 'AI Counselor & RAG', icon: Bot, badge: 'RAG' },
    { id: 'lab', label: 'Deep Learning Lab', icon: Brain, badge: 'Neural' },
    { id: 'history', label: 'Records & Profile', icon: History },
  ];

  return (
    <aside style={{
      width: '260px',
      minWidth: '260px',
      backgroundColor: 'rgba(15, 23, 42, 0.65)',
      borderRight: '1px solid var(--border-color)',
      padding: '20px 14px',
      display: 'flex',
      flexDirection: 'column',
      gap: '24px',
      height: 'calc(100vh - 65px)',
      position: 'sticky',
      top: '65px',
    }}>
      <div>
        <div style={{
          fontSize: '11px',
          fontWeight: 700,
          textTransform: 'uppercase',
          letterSpacing: '1px',
          color: 'var(--text-sub)',
          padding: '0 12px 10px',
        }}>
          Navigation Hub
        </div>
        <nav style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onSelectTab(item.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '10px 14px',
                  borderRadius: 'var(--radius-sm)',
                  backgroundColor: isActive ? 'rgba(99, 102, 241, 0.15)' : 'transparent',
                  color: isActive ? '#FFFFFF' : 'var(--text-muted)',
                  border: isActive ? '1px solid rgba(99, 102, 241, 0.4)' : '1px solid transparent',
                  cursor: 'pointer',
                  textAlign: 'left',
                  transition: 'all 0.15s ease',
                  fontSize: '13.5px',
                  fontWeight: isActive ? 600 : 500,
                  width: '100%',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <Icon size={18} color={isActive ? '#818CF8' : '#94A3B8'} />
                  <span>{item.label}</span>
                </div>
                {item.badge && (
                  <span className={`badge ${isActive ? 'badge-primary' : 'badge-info'}`} style={{ fontSize: '10px', padding: '2px 6px' }}>
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Active Session Context Box */}
      {activeContext && (activeContext.topCareer || activeContext.resumeName) && (
        <div style={{
          marginTop: 'auto',
          background: 'rgba(30, 41, 59, 0.6)',
          border: '1px solid var(--border-color)',
          borderRadius: 'var(--radius-sm)',
          padding: '12px',
        }}>
          <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-sub)', marginBottom: '8px' }}>
            ACTIVE CONTEXT
          </div>
          {activeContext.topCareer && (
            <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '4px' }}>
              🎯 Career: <b style={{ color: '#38BDF8' }}>{activeContext.topCareer}</b>
            </div>
          )}
          {activeContext.resumeName && (
            <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              📄 Resume: <b style={{ color: '#34D399' }}>{activeContext.resumeName}</b>
            </div>
          )}
        </div>
      )}
    </aside>
  );
};
