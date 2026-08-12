import React, { useState, useRef, useEffect } from 'react';
import api from '../services/api';

export default function ChatInterface({ conversationId, onConversationCreate }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const send = (e) => {
    e.preventDefault();
    const question = input.trim();
    if (!question || loading) return;

    // Push the user message + an empty assistant placeholder we stream into.
    setMessages((prev) => [
      ...prev,
      { role: 'user', content: question },
      { role: 'assistant', content: '', sources: null, meta: null, streaming: true },
    ]);
    setInput('');
    setLoading(true);
    setError(null);

    const updateLast = (patch) =>
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        next[next.length - 1] =
          typeof patch === 'function' ? patch(last) : { ...last, ...patch };
        return next;
      });

    api.chatStream(
      { question, conversationId },
      {
        onSources: (chunks) => updateLast({ sources: chunks }),
        onToken: (text) =>
          updateLast((last) => ({ ...last, content: last.content + text })),
        onDone: (meta) => {
          if (!conversationId && meta.conversation_id) {
            onConversationCreate(meta.conversation_id);
          }
          updateLast({
            streaming: false,
            meta: { tokens: meta.tokens_used, ms: meta.processing_time_ms },
          });
          setLoading(false);
        },
        onError: (detail) => {
          setError(detail || 'Erreur réseau');
          // Drop the empty assistant placeholder on failure.
          setMessages((prev) => {
            const next = [...prev];
            if (next.length && next[next.length - 1].role === 'assistant' && !next[next.length - 1].content) {
              next.pop();
            }
            return next;
          });
          setLoading(false);
        },
      }
    );
  };

  return (
    <div className="flex h-full flex-col rounded-xl border border-slate-200 bg-white">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4">
        {messages.length === 0 && !loading && <EmptyState />}
        <div className="mx-auto max-w-3xl space-y-4">
          {messages.map((msg, i) => (
            <MessageBubble key={i} msg={msg} />
          ))}
          {error && (
            <div className="rounded-lg bg-red-50 px-4 py-2 text-sm text-red-600">
              ⚠️ {error}
            </div>
          )}
          <div ref={endRef} />
        </div>
      </div>

      {/* Input */}
      <form onSubmit={send} className="border-t border-slate-100 p-3">
        <div className="mx-auto flex max-w-3xl items-end gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) send(e);
            }}
            placeholder="Posez une question sur vos documents…"
            rows={1}
            disabled={loading}
            className="max-h-32 flex-1 resize-none rounded-lg border border-slate-200 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500 disabled:opacity-60"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Envoyer
          </button>
        </div>
      </form>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex h-full flex-col items-center justify-center text-center text-slate-400">
      <div className="text-4xl">💬</div>
      <p className="mt-2 max-w-sm text-sm">
        Uploadez des documents puis posez une question. Les réponses citent les
        passages sources utilisés.
      </p>
    </div>
  );
}

function MessageBubble({ msg }) {
  const isUser = msg.role === 'user';

  // Assistant placeholder before the first token: show the typing animation.
  if (!isUser && msg.streaming && !msg.content) {
    return <TypingBubble />;
  }

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm ${
          isUser
            ? 'rounded-br-sm bg-brand-600 text-white'
            : 'rounded-bl-sm border border-slate-200 bg-slate-50 text-slate-800'
        }`}
      >
        <div
          className="prose-msg whitespace-pre-wrap"
          dangerouslySetInnerHTML={{
            __html:
              renderMarkdown(msg.content) +
              (msg.streaming
                ? '<span class="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-slate-400 align-middle"></span>'
                : ''),
          }}
        />
        {msg.sources && msg.sources.length > 0 && <Sources sources={msg.sources} />}
        {msg.meta && (
          <div className="mt-1 text-[11px] text-slate-400">
            {msg.meta.tokens > 0 && `${msg.meta.tokens} tokens · `}
            {Math.round(msg.meta.ms)} ms
          </div>
        )}
      </div>
    </div>
  );
}

function Sources({ sources }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-2 border-t border-slate-200 pt-2">
      <button
        onClick={() => setOpen((o) => !o)}
        className="text-xs font-medium text-brand-600 hover:underline"
      >
        {open ? '▼' : '▶'} {sources.length} source{sources.length > 1 ? 's' : ''}
      </button>
      {open && (
        <ul className="mt-2 space-y-2">
          {sources.map((s) => (
            <li key={s.id} className="rounded-lg bg-white p-2 text-xs">
              <div className="mb-1 flex items-center justify-between">
                <span className="font-medium text-slate-700">
                  📎 {s.filename || `doc #${s.document_id}`}
                </span>
                <span className="rounded bg-brand-50 px-1.5 py-0.5 text-brand-700">
                  {(s.similarity_score * 100).toFixed(0)}%
                </span>
              </div>
              <p className="text-slate-500">
                {s.text.length > 220 ? s.text.slice(0, 220) + '…' : s.text}
              </p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function TypingBubble() {
  return (
    <div className="flex justify-start">
      <div className="rounded-2xl rounded-bl-sm border border-slate-200 bg-slate-50 px-4 py-3">
        <div className="flex gap-1">
          {[0, 0.2, 0.4].map((d) => (
            <span
              key={d}
              className="h-2 w-2 animate-bounce-dot rounded-full bg-slate-400"
              style={{ animationDelay: `${d}s` }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

// Minimal, safe-ish markdown: escape HTML first, then apply a few patterns.
function renderMarkdown(text) {
  const escaped = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  return escaped
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');
}
