import useTheme from '../hooks/useTheme';

export default function Navbar({ stats }) {
  const { theme, toggle } = useTheme();

  return (
    <header className="border-b border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">
      <div className="mx-auto flex w-full max-w-7xl items-center justify-between px-4 py-3">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand-600 text-lg font-bold text-white">
            R
          </div>
          <div>
            <h1 className="text-lg font-semibold leading-tight text-slate-900 dark:text-slate-100">
              RAG Chatbot
            </h1>
            <p className="text-xs text-slate-500 dark:text-slate-400">
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
                    ? 'bg-amber-100 text-amber-700 dark:bg-amber-500/15 dark:text-amber-300'
                    : 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-300'
                }`}
                title={
                  stats.demo_mode ? 'Aucune clé Anthropic configurée' : stats.chat_model
                }
              >
                {stats.demo_mode ? '⚙️ Mode démo' : `✅ ${stats.chat_model}`}
              </span>
            </>
          )}

          <button
            onClick={toggle}
            aria-label={
              theme === 'dark' ? 'Passer en mode clair' : 'Passer en mode sombre'
            }
            title={theme === 'dark' ? 'Mode clair' : 'Mode sombre'}
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-slate-200 text-slate-600 transition hover:bg-slate-100 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
          >
            {theme === 'dark' ? '☀️' : '🌙'}
          </button>
        </div>
      </div>
    </header>
  );
}

function Stat({ label, value }) {
  return (
    <div className="hidden text-center sm:block">
      <div className="font-semibold text-slate-900 dark:text-slate-100">
        {value ?? '—'}
      </div>
      <div className="text-[11px] uppercase tracking-wide text-slate-400 dark:text-slate-500">
        {label}
      </div>
    </div>
  );
}
