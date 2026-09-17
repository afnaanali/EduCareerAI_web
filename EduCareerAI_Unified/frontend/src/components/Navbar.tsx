import { GraduationCap, User as UserIcon, LogOut } from 'lucide-react';
import type { User } from '../types';

interface NavbarProps {
  user: User | null;
  backendOnline: boolean;
  onOpenAuth: () => void;
  onLogout: () => void;
  activeTab?: string;
  onSelectTab: (tab: string) => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  user,
  backendOnline,
  onOpenAuth,
  onLogout,
  onSelectTab,
}) => {
  return (
    <header style={{
      borderBottom: '1px solid var(--border-color)',
      backgroundColor: 'rgba(15, 23, 42, 0.85)',
      backdropFilter: 'blur(16px)',
      position: 'sticky',
      top: 0,
      zIndex: 50,
      padding: '12px 24px',
    }}>
      <div style={{
        maxWidth: '1440px',
        margin: '0 auto',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        {/* Brand Logo */}
        <div 
          onClick={() => onSelectTab('dashboard')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            cursor: 'pointer',
          }}
        >
          <div style={{
            width: '40px',
            height: '40px',
            borderRadius: '10px',
            background: 'linear-gradient(135deg, #6366F1 0%, #A855F7 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 15px rgba(99, 102, 241, 0.5)',
          }}>
            <GraduationCap size={24} color="#FFFFFF" />
          </div>
          <div>
            <div style={{
              fontSize: '20px',
              fontWeight: 800,
              letterSpacing: '-0.5px',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
            }}>
              <span>EduCareer</span>
              <span className="gradient-text">AI</span>
              <span className="badge badge-primary" style={{ fontSize: '10px', padding: '2px 6px' }}>v2.0</span>
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-sub)', fontWeight: 500 }}>
              Unified Intelligence Platform
            </div>
          </div>
        </div>

        {/* Center - System Keep-Alive Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '6px 14px',
            borderRadius: 'var(--radius-full)',
            background: backendOnline ? 'rgba(16, 185, 129, 0.1)' : 'rgba(244, 63, 94, 0.1)',
            border: `1px solid ${backendOnline ? 'rgba(16, 185, 129, 0.3)' : 'rgba(244, 63, 94, 0.3)'}`,
          }}>
            <span style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: backendOnline ? '#10B981' : '#F43F5E',
              boxShadow: backendOnline ? '0 0 8px #10B981' : 'none',
              animation: backendOnline ? 'pulseGlow 2s infinite' : 'none',
            }} />
            <span style={{
              fontSize: '12px',
              fontWeight: 600,
              color: backendOnline ? '#34D399' : '#FB7185',
            }}>
              {backendOnline ? 'Backend 24/7 Active' : 'Connecting to API...'}
            </span>
          </div>
        </div>

        {/* Right - Account & Actions */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          {user ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div 
                onClick={() => onSelectTab('history')}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                  background: 'rgba(51, 65, 85, 0.5)',
                  padding: '6px 14px',
                  borderRadius: 'var(--radius-full)',
                  border: '1px solid var(--border-color)',
                  cursor: 'pointer',
                }}
              >
                <div style={{
                  width: '28px',
                  height: '28px',
                  borderRadius: '50%',
                  background: 'var(--primary)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontWeight: 700,
                  fontSize: '13px',
                }}>
                  {(user.full_name || user.username)[0].toUpperCase()}
                </div>
                <div style={{ textAlign: 'left' }}>
                  <div style={{ fontSize: '13px', fontWeight: 600 }}>{user.full_name || user.username}</div>
                  <div style={{ fontSize: '10px', color: 'var(--accent-cyan)' }}>@{user.username}</div>
                </div>
              </div>
              <button 
                className="btn btn-ghost" 
                onClick={onLogout}
                title="Log Out"
                style={{ padding: '8px' }}
              >
                <LogOut size={18} />
              </button>
            </div>
          ) : (
            <button className="btn btn-primary" onClick={onOpenAuth}>
              <UserIcon size={16} />
              <span>Sign In / Register</span>
            </button>
          )}
        </div>
      </div>
    </header>
  );
};
