import { useState, useEffect, useCallback } from 'react';
import { Navbar } from './components/Navbar';
import { Sidebar } from './components/Sidebar';
import { AuthModal } from './components/AuthModal';

import { DashboardView } from './views/DashboardView';
import { CareerNavigatorView } from './views/CareerNavigatorView';
import { RoadmapExplorerView } from './views/RoadmapExplorerView';
import { ResumeStudioView } from './views/ResumeStudioView';
import { AiAssistantView } from './views/AiAssistantView';
import { InterviewCoachView } from './views/InterviewCoachView';
import { DeepLearningLabView } from './views/DeepLearningLabView';
import { HistoryRecordsView } from './views/HistoryRecordsView';

import { api } from './api';
import type { User } from './types';

export function App() {
  const [activeTab, setActiveTab] = useState<string>('dashboard');
  const [user, setUser] = useState<User | null>(null);
  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [backendOnline, setBackendOnline] = useState<boolean>(true);

  // Shared application context
  const [selectedRoadmapCareer, setSelectedRoadmapCareer] = useState<string>('Data Analyst');
  const [activeContext, setActiveContext] = useState<{
    topCareer?: string;
    resumeName?: string;
  }>({});

  // Health check polling
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const res = await api.checkHealth();
        setBackendOnline(res.status === 'healthy');
      } catch (err) {
        setBackendOnline(false);
      }
    };

    checkBackend();
    const interval = setInterval(checkBackend, 30000); // 30s keep-alive poll
    return () => clearInterval(interval);
  }, []);

  // Hydrate user-specific context from profile and history
  const hydrateUserContext = useCallback(async (userId: number) => {
    try {
      const profileRes = await api.getProfile(userId);
      let topCareer: string | undefined = undefined;
      let resumeName: string | undefined = undefined;

      if (profileRes?.latest_assessment?.top_career) {
        topCareer = profileRes.latest_assessment.top_career;
      }
      if (profileRes?.latest_resume?.resume_name) {
        resumeName = profileRes.latest_resume.resume_name;
      }

      if (!topCareer || !resumeName) {
        const history = await api.getUserHistory(userId);
        if (!topCareer && history.assessments && history.assessments.length > 0) {
          topCareer = history.assessments[0].top_career;
        }
        if (!resumeName && history.resume_scans && history.resume_scans.length > 0) {
          resumeName = history.resume_scans[0].resume_name;
        }
      }

      setActiveContext({ topCareer, resumeName });
      if (topCareer) {
        setSelectedRoadmapCareer(topCareer);
      }
    } catch (e) {
      console.warn('Failed to hydrate user context from API', e);
    }
  }, []);

  // Restore stored user from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('educareer_user');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setUser(parsed);
        if (parsed?.id) {
          hydrateUserContext(parsed.id);
        }
      } catch (e) {
        localStorage.removeItem('educareer_user');
      }
    }
  }, [hydrateUserContext]);

  const handleLoginSuccess = (
    newUser: User,
    _profile?: any,
    latestAssessment?: any,
    latestResume?: any
  ) => {
    setUser(newUser);
    localStorage.setItem('educareer_user', JSON.stringify(newUser));

    let topCareer = latestAssessment?.top_career;
    let resumeName = latestResume?.resume_name;

    if (topCareer || resumeName) {
      setActiveContext({ topCareer, resumeName });
      if (topCareer) {
        setSelectedRoadmapCareer(topCareer);
      }
    } else {
      hydrateUserContext(newUser.id);
    }
  };

  const handleLogout = () => {
    setUser(null);
    localStorage.removeItem('educareer_user');
    setActiveContext({});
    setSelectedRoadmapCareer('Data Analyst');
  };

  const navigateToRoadmap = (careerName: string) => {
    setSelectedRoadmapCareer(careerName);
    setActiveTab('roadmap');
  };

  const userKey = user ? `user-${user.id}` : 'guest-session';

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Top Navigation */}
      <Navbar
        user={user}
        backendOnline={backendOnline}
        onOpenAuth={() => setAuthModalOpen(true)}
        onLogout={handleLogout}
        activeTab={activeTab}
        onSelectTab={setActiveTab}
      />

      {/* Main Layout */}
      <div style={{ display: 'flex', flex: 1, maxWidth: '1440px', margin: '0 auto', width: '100%' }}>
        <Sidebar
          activeTab={activeTab}
          onSelectTab={setActiveTab}
          activeContext={activeContext}
        />

        {/* View Content Area with dynamic user key */}
        <main key={userKey} style={{ flex: 1, padding: '24px 32px', minWidth: 0 }}>
          {activeTab === 'dashboard' && (
            <DashboardView
              user={user}
              onSelectTab={setActiveTab}
              activeContext={activeContext}
            />
          )}

          {activeTab === 'career_nav' && (
            <CareerNavigatorView
              user={user}
              onNavigateToRoadmap={navigateToRoadmap}
              onSetContext={(ctx) => setActiveContext((prev) => ({ ...prev, ...ctx }))}
            />
          )}

          {activeTab === 'roadmap' && (
            <RoadmapExplorerView
              user={user}
              initialCareer={selectedRoadmapCareer}
            />
          )}

          {activeTab === 'resume' && (
            <ResumeStudioView
              user={user}
              onSetContext={(ctx) => setActiveContext((prev) => ({ ...prev, ...ctx }))}
              onNavigateToRoadmap={navigateToRoadmap}
            />
          )}

          {activeTab === 'interview_coach' && <InterviewCoachView />}

          {activeTab === 'chat' && (
            <AiAssistantView
              user={user}
              activeContext={activeContext}
            />
          )}

          {activeTab === 'lab' && <DeepLearningLabView onNavigateToInterviewCoach={() => setActiveTab('interview_coach')} />}

          {activeTab === 'history' && (
            <HistoryRecordsView
              user={user}
              onOpenAuth={() => setAuthModalOpen(true)}
            />
          )}
        </main>
      </div>

      {/* Auth Modal */}
      <AuthModal
        isOpen={authModalOpen}
        onClose={() => setAuthModalOpen(false)}
        onLoginSuccess={handleLoginSuccess}
      />
    </div>
  );
}

export default App;
