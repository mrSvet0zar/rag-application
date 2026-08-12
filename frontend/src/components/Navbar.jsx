import React from 'react';

export default function Navbar({ stats }) {
  return (
    <header className="border-b border-slate-200 bg-white">
      <div className="mx-auto flex w-full max-w-7xl items-center justify-between px-4 py-3">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand-600 text-lg font-bold text-white">
            R
          </div>
          <div>
            <h1 className="text-lg font-semibold leading-tight text-slate-900">
              RAG Chatbot
            </h1>
            <p className="text-xs text-slate-500">
              Questions-réponses sur vos documents
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4 text-sm">
          {stats && (
            <>
              <Stat label="Documents" value={stats.total_documents} />
              <Stat label="Chunks" value={stats.total_chunks} />
              <Stat label="Questions" value={stats.total_messages} />
              <span
                className={`rounded-full px-2.5 py-1 text-xs font-medium ${
                  stats.demo_mode
                    ? 'bg-amber-100 text-amber-700'
                    : 'bg-emerald-100 text-emerald-700'
                }`}
                title={stats.demo_mode ? 'Aucune clé Anthropic configurée' : stats.chat_model}
              >
                {stats.demo_mode ? '⚙️ Mode démo' : `✅ ${stats.chat_model}`}
              </span>
            </>
          )}
        </div>
      </div>
    </header>
  );
}

function Stat({ label, value }) {
  return (
    <div className="hidden text-center sm:block">
      <div className="font-semibold text-slate-900">{value ?? '—'}</div>
      <div className="text-[11px] uppercase tracking-wide text-slate-400">{label}</div>
    </div>
  );
}
