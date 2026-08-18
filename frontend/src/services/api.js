import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const TIMEOUT = Number(import.meta.env.VITE_API_TIMEOUT) || 60000;

const client = axios.create({
  baseURL: API_URL,
  timeout: TIMEOUT,
});

export const api = {
  async health() {
    const { data } = await client.get('/api/health');
    return data;
  },

  async listDocuments() {
    const { data } = await client.get('/api/documents');
    return data;
  },

  async uploadDocument(file) {
    const formData = new FormData();
    formData.append('file', file);
    const { data } = await client.post('/api/documents/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return data;
  },

  async importUrl(url) {
    const { data } = await client.post('/api/documents/import-url', { url });
    return data;
  },

  async deleteDocument(id) {
    const { data } = await client.delete(`/api/documents/${id}`);
    return data;
  },

  async chat({ question, conversationId, k = 5, temperature = 0.7 }) {
    const { data } = await client.post('/api/chat', {
      question,
      conversation_id: conversationId || null,
      k,
      temperature,
    });
    return data;
  },

  async stats() {
    const { data } = await client.get('/api/stats');
    return data;
  },

  /**
   * Streaming chat over SSE. `fetch` is used (not EventSource) because we POST
   * a JSON body. Callbacks: onSources(chunks), onToken(text), onDone(meta),
   * onError(detail). Returns a function to abort the stream.
   */
  chatStream(
    { question, conversationId, k = 5 },
    { onSources, onToken, onDone, onError }
  ) {
    const controller = new AbortController();

    (async () => {
      try {
        const resp = await fetch(`${API_URL}/api/chat/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question, conversation_id: conversationId || null, k }),
          signal: controller.signal,
        });

        if (!resp.ok || !resp.body) {
          const text = await resp.text().catch(() => '');
          onError?.(text || `HTTP ${resp.status}`);
          return;
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          let sep;
          while ((sep = buffer.indexOf('\n\n')) >= 0) {
            const frame = buffer.slice(0, sep);
            buffer = buffer.slice(sep + 2);
            handleFrame(frame, { onSources, onToken, onDone, onError });
          }
        }
      } catch (err) {
        if (err.name !== 'AbortError') onError?.(err.message || 'Erreur réseau');
      }
    })();

    return () => controller.abort();
  },
};

function handleFrame(frame, { onSources, onToken, onDone, onError }) {
  let event = 'message';
  let data = '';
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim();
    else if (line.startsWith('data:')) data += line.slice(5).trim();
  }
  if (!data) return;

  let payload;
  try {
    payload = JSON.parse(data);
  } catch {
    return;
  }

  if (event === 'sources') onSources?.(payload.retrieved_chunks || []);
  else if (event === 'token') onToken?.(payload.text || '');
  else if (event === 'done') onDone?.(payload);
  else if (event === 'error') onError?.(payload.detail || 'Erreur serveur');
}

export default api;
