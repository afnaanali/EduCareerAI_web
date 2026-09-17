import React, { useState, useEffect, useRef } from 'react';
import {
  Bot,
  Send,
  Upload,
  Sparkles,
  FileCheck,
  User as UserIcon,
  BookOpen,
  Briefcase,
} from 'lucide-react';
import { api } from '../api';
import type { ChatMessage, User } from '../types';

interface AiAssistantViewProps {
  user: User | null;
  activeContext?: {
    topCareer?: string;
    resumeName?: string;
  };
}

export const AiAssistantView: React.FC<AiAssistantViewProps> = ({
  user,
  activeContext,
}) => {
  const DEFAULT_GREETING: ChatMessage = {
    role: 'assistant',
    content:
      '👋 Hello! I am **EduCareer AI**, your unified career counselor and educational intelligence mentor. Ask me anything about career roadmaps, degree pathways, interview preparation, or upload a textbook/syllabus for document-aware Q&A!',
  };

  const [messages, setMessages] = useState<ChatMessage[]>([DEFAULT_GREETING]);
  const [input, setInput] = useState('');
  const [mode, setMode] = useState<'💼 Career Assistant' | '📚 Study Assistant'>('💼 Career Assistant');
  const [loading, setLoading] = useState(false);
  const [ragFiles, setRagFiles] = useState<string[]>([]);
  const [ragChunksCount, setRagChunksCount] = useState(0);
  const [uploadingDoc, setUploadingDoc] = useState(false);

  const chatEndRef = useRef<HTMLDivElement>(null);
  const sessionId = user ? `user_${user.id}` : 'guest_session';

  // Hydrate chat history when user changes
  useEffect(() => {
    if (!user?.id) {
      setMessages([DEFAULT_GREETING]);
      return;
    }

    const loadHistory = async () => {
      try {
        const res = await api.getChatHistory(user.id);
        if (res && res.history && res.history.length > 0) {
          const userMsgs: ChatMessage[] = res.history.map((h: any) => ({
            role: h.role as 'user' | 'assistant',
            content: h.content,
          }));
          setMessages([DEFAULT_GREETING, ...userMsgs]);
        } else {
          setMessages([DEFAULT_GREETING]);
        }
      } catch (err) {
        console.warn('Failed to load chat history from server', err);
      }
    };

    loadHistory();
  }, [user?.id]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleSend = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!input.trim() || loading) return;

    const userQuery = input.trim();
    setInput('');
    const newMessages: ChatMessage[] = [...messages, { role: 'user', content: userQuery }];
    setMessages(newMessages);
    setLoading(true);

    try {
      const historyPayload = newMessages.slice(-6).map((m) => ({
        role: m.role,
        content: m.content,
      }));

      const res = await api.sendChatMessage({
        session_id: sessionId,
        user_id: user?.id,
        message: userQuery,
        mode: mode,
        resume_context: activeContext?.resumeName ? `Active resume: ${activeContext.resumeName}` : undefined,
        history: historyPayload,
      });

      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: res.response,
          citations: res.citations,
          has_rag_context: res.has_rag_context,
        },
      ]);
    } catch (err: any) {
      console.error('Chat API error:', err);
      const errorDetail =
        err?.response?.data?.detail ||
        (err?.message === 'Network Error'
          ? 'Could not connect to backend server. Please make sure the backend is running.'
          : 'An error occurred while communicating with the AI service. Please try again.');
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `⚠️ ${errorDetail}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleDocUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploadingDoc(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('session_id', sessionId);
      const res = await api.uploadRagDocument(formData);
      if (res.success) {
        setRagFiles(res.files);
        setRagChunksCount(res.total_chunks);
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: `📄 **Document Indexed!** Added **${file.name}** (${res.chunks_added} chunks). I can now answer questions grounded in this document.`,
          },
        ]);
      }
    } catch (err: any) {
      console.error('Document upload failed', err);
      const detail =
        err?.response?.data?.detail ||
        err?.response?.data?.message ||
        err?.message ||
        'Failed to process document.';
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `⚠️ **Upload Failed**: ${detail}`,
        },
      ]);
    } finally {
      setUploadingDoc(false);
      e.target.value = '';
    }
  };

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 110px)', gap: '16px' }}>
      {/* Top Header & Mode Toolbar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '12px',
        paddingBottom: '8px',
        borderBottom: '1px solid var(--border-color)',
      }}>
        <div>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
            <span className="badge badge-primary">
              <Sparkles size={12} />
              RAG & LLM Mentor
            </span>
          </div>
          <h1 style={{ fontSize: '24px', fontWeight: 800 }}>AI Career & Educational <span className="gradient-text">Counselor</span></h1>
        </div>

        {/* Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {/* Mode Selector */}
          <div style={{
            display: 'flex',
            background: 'rgba(15, 23, 42, 0.6)',
            borderRadius: 'var(--radius-sm)',
            padding: '3px',
            border: '1px solid var(--border-color)',
          }}>
            <button
              onClick={() => setMode('💼 Career Assistant')}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '6px 12px',
                border: 'none',
                borderRadius: '6px',
                background: mode === '💼 Career Assistant' ? 'var(--primary)' : 'transparent',
                color: mode === '💼 Career Assistant' ? '#FFF' : 'var(--text-muted)',
                fontWeight: 600,
                fontSize: '12px',
                cursor: 'pointer',
              }}
            >
              <Briefcase size={14} />
              <span>Career Mode</span>
            </button>
            <button
              onClick={() => setMode('📚 Study Assistant')}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '6px 12px',
                border: 'none',
                borderRadius: '6px',
                background: mode === '📚 Study Assistant' ? 'var(--primary)' : 'transparent',
                color: mode === '📚 Study Assistant' ? '#FFF' : 'var(--text-muted)',
                fontWeight: 600,
                fontSize: '12px',
                cursor: 'pointer',
              }}
            >
              <BookOpen size={14} />
              <span>Study Mode</span>
            </button>
          </div>

          {/* Upload RAG File */}
          <label className="btn btn-secondary" style={{ padding: '7px 12px', fontSize: '12px', cursor: 'pointer' }}>
            <Upload size={14} />
            <span>{uploadingDoc ? 'Indexing...' : 'Upload Knowledge (RAG)'}</span>
            <input
              type="file"
              accept=".pdf,.txt,.docx"
              style={{ display: 'none' }}
              onChange={handleDocUpload}
              disabled={uploadingDoc}
            />
          </label>
        </div>
      </div>

      {/* RAG status banner if files exist */}
      {ragFiles.length > 0 && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '8px 14px',
          background: 'rgba(6, 182, 212, 0.1)',
          border: '1px solid rgba(6, 182, 212, 0.3)',
          borderRadius: 'var(--radius-sm)',
          fontSize: '12px',
          color: '#38BDF8',
        }}>
          <FileCheck size={16} />
          <span><b>Active RAG Knowledge:</b> {ragFiles.join(', ')} ({ragChunksCount} semantic chunks indexed)</span>
        </div>
      )}

      {/* Messages Container */}
      <div className="glass-card" style={{
        flex: 1,
        padding: '20px',
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: '16px',
      }}>
        {messages.map((m, idx) => (
          <div
            key={idx}
            style={{
              display: 'flex',
              gap: '12px',
              alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
              maxWidth: '85%',
            }}
          >
            {m.role === 'assistant' && (
              <div style={{
                width: '34px',
                height: '34px',
                borderRadius: '8px',
                background: 'linear-gradient(135deg, #6366F1, #A855F7)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0,
              }}>
                <Bot size={20} color="#FFF" />
              </div>
            )}

            <div style={{
              padding: '14px 18px',
              borderRadius: 'var(--radius-md)',
              background: m.role === 'user'
                ? 'linear-gradient(135deg, #4F46E5, #6366F1)'
                : 'rgba(15, 23, 42, 0.75)',
              border: m.role === 'user' ? 'none' : '1px solid var(--border-color)',
              color: '#FFF',
              fontSize: '14px',
              lineHeight: 1.6,
              whiteSpace: 'pre-wrap',
            }}>
              {m.content}

              {m.citations && (
                <div style={{
                  marginTop: '12px',
                  paddingTop: '10px',
                  borderTop: '1px solid rgba(255, 255, 255, 0.1)',
                  fontSize: '12px',
                  color: 'var(--accent-cyan)',
                }}>
                  {m.citations}
                </div>
              )}
            </div>

            {m.role === 'user' && (
              <div style={{
                width: '34px',
                height: '34px',
                borderRadius: '8px',
                background: 'rgba(51, 65, 85, 0.8)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0,
              }}>
                <UserIcon size={18} color="#FFF" />
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>
            <div style={{
              width: '34px',
              height: '34px',
              borderRadius: '8px',
              background: 'linear-gradient(135deg, #6366F1, #A855F7)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <Bot size={20} color="#FFF" />
            </div>
            <span>EduCareer AI is generating advice...</span>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      {/* Input Bar */}
      <form onSubmit={handleSend} style={{ display: 'flex', gap: '10px' }}>
        <input
          type="text"
          className="form-input"
          placeholder="Ask a question (e.g., 'How to become a Data Scientist?', 'Explain Docker', 'What are its types?')..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={loading}
          style={{ flex: 1, padding: '14px 18px', fontSize: '14px' }}
        />
        <button
          type="submit"
          className="btn btn-primary"
          style={{ padding: '0 24px' }}
          disabled={loading || !input.trim()}
        >
          <Send size={18} />
          <span>Send</span>
        </button>
      </form>
    </div>
  );
};
