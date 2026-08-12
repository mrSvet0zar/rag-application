import React, { useRef, useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import api from '../services/api';

export default function DocumentPanel() {
  const queryClient = useQueryClient();
  const fileInputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState(null);
  const [url, setUrl] = useState('');

  const { data: documents = [], isLoading } = useQuery({
    queryKey: ['documents'],
    queryFn: api.listDocuments,
  });

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ['documents'] });
    queryClient.invalidateQueries({ queryKey: ['stats'] });
  };

  const uploadMutation = useMutation({
    mutationFn: api.uploadDocument,
    onMutate: () => setError(null),
    onSuccess: invalidate,
    onError: (err) =>
      setError(err.response?.data?.detail || err.message || 'Échec de l’upload'),
  });

  const importUrlMutation = useMutation({
    mutationFn: api.importUrl,
    onMutate: () => setError(null),
    onSuccess: () => {
      setUrl('');
      invalidate();
    },
    onError: (err) =>
      setError(err.response?.data?.detail || err.message || 'Échec de l’import'),
  });

  const deleteMutation = useMutation({
    mutationFn: api.deleteDocument,
    onSuccess: invalidate,
  });

  const handleFiles = (files) => {
    for (const file of files) uploadMutation.mutate(file);
  };

  return (
    <div className="flex h-full flex-col rounded-xl border border-slate-200 bg-white">
      <div className="border-b border-slate-100 px-4 py-3">
        <h2 className="text-sm font-semibold text-slate-900">📄 Documents</h2>
        <p className="text-xs text-slate-500">Base de connaissances</p>
      </div>

      {/* Dropzone */}
      <div className="p-4">
        <div
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            handleFiles(e.dataTransfer.files);
          }}
          className={`cursor-pointer rounded-lg border-2 border-dashed p-6 text-center transition ${
            dragOver
              ? 'border-brand-500 bg-brand-50'
              : 'border-slate-200 hover:border-brand-400 hover:bg-slate-50'
          }`}
        >
          <div className="text-2xl">⬆️</div>
          <p className="mt-1 text-sm font-medium text-slate-700">
            Glissez un fichier ici
          </p>
          <p className="text-xs text-slate-400">
            ou cliquez — .txt, .md, .pdf, .docx, .html
          </p>
          <input
            ref={fileInputRef}
            type="file"
            accept=".txt,.md,.markdown,.pdf,.docx,.html,.htm,.csv,.json"
            multiple
            className="hidden"
            onChange={(e) => handleFiles(e.target.files)}
          />
        </div>

        {/* URL import */}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (url.trim()) importUrlMutation.mutate(url.trim());
          }}
          className="mt-3 flex gap-2"
        >
          <input
            type="url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="🔗 Importer depuis une URL…"
            disabled={importUrlMutation.isPending}
            className="min-w-0 flex-1 rounded-lg border border-slate-200 px-2.5 py-1.5 text-xs focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500 disabled:opacity-60"
          />
          <button
            type="submit"
            disabled={importUrlMutation.isPending || !url.trim()}
            className="shrink-0 rounded-lg bg-brand-600 px-3 py-1.5 text-xs font-medium text-white transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Importer
          </button>
        </form>

        {(uploadMutation.isPending || importUrlMutation.isPending) && (
          <p className="mt-2 text-xs text-brand-600">⏳ Traitement en cours…</p>
        )}
        {error && <p className="mt-2 text-xs text-red-600">{error}</p>}
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto px-4 pb-4">
        {isLoading ? (
          <p className="text-sm text-slate-400">Chargement…</p>
        ) : documents.length === 0 ? (
          <p className="text-sm text-slate-400">Aucun document pour l’instant.</p>
        ) : (
          <ul className="space-y-2">
            {documents.map((doc) => (
              <li
                key={doc.id}
                className="group flex items-center justify-between rounded-lg border border-slate-100 bg-slate-50 px-3 py-2"
              >
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-slate-800">
                    {doc.filename}
                  </p>
                  <p className="text-xs text-slate-400">
                    {doc.total_chunks} chunks · {statusLabel(doc.status)}
                  </p>
                </div>
                <button
                  onClick={() => deleteMutation.mutate(doc.id)}
                  className="ml-2 shrink-0 rounded p-1 text-slate-300 hover:bg-red-50 hover:text-red-500"
                  title="Supprimer"
                >
                  ✕
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function statusLabel(status) {
  return (
    { completed: '✅ prêt', processing: '⏳ traitement', failed: '❌ échec' }[status] ||
    status
  );
}
