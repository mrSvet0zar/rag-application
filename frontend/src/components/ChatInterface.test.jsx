/**
 * The chat view renders assistant text through `dangerouslySetInnerHTML`, so
 * escaping is a security property here, not a formatting detail: the text comes
 * from an LLM summarising documents a user uploaded.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import ChatInterface from './ChatInterface';
import api from '../services/api';

vi.mock('../services/api', () => ({
  default: { chatStream: vi.fn() },
}));

/** Drive the component's stream callbacks by hand. */
function respondWith({ sources = [], tokens = [], done = {}, error = null } = {}) {
  api.chatStream.mockImplementation((_req, handlers) => {
    if (error) {
      handlers.onError(error);
      return () => {};
    }
    handlers.onSources(sources);
    tokens.forEach((t) => handlers.onToken(t));
    handlers.onDone({ tokens_used: 0, processing_time_ms: 1, ...done });
    return () => {};
  });
}

async function ask(question = 'ma question') {
  const user = userEvent.setup();
  render(<ChatInterface conversationId={null} onConversationCreate={() => {}} />);
  await user.type(screen.getByPlaceholderText(/Posez une question/i), question);
  await user.click(screen.getByRole('button', { name: /Envoyer/i }));
  return user;
}

describe('ChatInterface', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    respondWith({ tokens: ['Une réponse.'] });
  });

  it('shows the question and the streamed answer', async () => {
    await ask('Quel index vectoriel ?');

    expect(await screen.findByText('Quel index vectoriel ?')).toBeInTheDocument();
    expect(await screen.findByText(/Une réponse\./)).toBeInTheDocument();
  });

  it('does not send an empty question', async () => {
    const user = userEvent.setup();
    render(<ChatInterface conversationId={null} onConversationCreate={() => {}} />);

    await user.click(screen.getByRole('button', { name: /Envoyer/i }));

    expect(api.chatStream).not.toHaveBeenCalled();
  });

  it('reports the conversation id back to the parent', async () => {
    respondWith({ tokens: ['ok'], done: { conversation_id: 'conv-42' } });
    const onCreate = vi.fn();
    const user = userEvent.setup();
    render(<ChatInterface conversationId={null} onConversationCreate={onCreate} />);

    await user.type(screen.getByPlaceholderText(/Posez une question/i), 'q');
    await user.click(screen.getByRole('button', { name: /Envoyer/i }));

    await waitFor(() => expect(onCreate).toHaveBeenCalledWith('conv-42'));
  });

  it('shows an error instead of an empty answer bubble', async () => {
    respondWith({ error: 'Trop de requêtes.' });

    await ask();

    expect(await screen.findByText(/Trop de requêtes\./)).toBeInTheDocument();
  });

  describe('escaping', () => {
    it('renders markup in the answer as text, never as elements', async () => {
      respondWith({ tokens: ['<img src=x onerror="alert(1)">'] });

      await ask();

      await screen.findByText(/img src=x/);
      expect(document.querySelector('img')).toBeNull();
    });

    it('escapes a source excerpt too', async () => {
      respondWith({
        tokens: ['réponse'],
        sources: [
          {
            id: 1,
            filename: 'a.md',
            document_id: 1,
            text: '<script>alert(1)</script>',
            similarity_score: 0.5,
          },
        ],
      });

      const user = await ask();
      await user.click(await screen.findByRole('button', { name: /1 source/i }));

      expect(document.querySelector('script')).toBeNull();
      expect(await screen.findByText(/alert\(1\)/)).toBeInTheDocument();
    });

    it('still applies the intended bold formatting', async () => {
      respondWith({ tokens: ['un mot **important** ici'] });

      await ask();

      expect(await screen.findByText('important')).toHaveRole('strong');
    });
  });

  describe('source badges', () => {
    const source = (extra) => ({
      id: 1,
      filename: 'guide.md',
      document_id: 1,
      text: 'un extrait',
      ...extra,
    });

    it('shows the rerank score when the passage was reranked', async () => {
      respondWith({
        tokens: ['r'],
        sources: [source({ rerank_score: 0.94, similarity_score: 0.6 })],
      });

      const user = await ask();
      await user.click(await screen.findByRole('button', { name: /1 source/i }));

      expect(await screen.findByText('⭐ 94%')).toBeInTheDocument();
    });

    it('falls back to the cosine score when there was no reranking', async () => {
      respondWith({ tokens: ['r'], sources: [source({ similarity_score: 0.42 })] });

      const user = await ask();
      await user.click(await screen.findByRole('button', { name: /1 source/i }));

      expect(await screen.findByText('42%')).toBeInTheDocument();
    });

    it('labels a lexical-only hit rather than showing a fabricated 0%', async () => {
      respondWith({ tokens: ['r'], sources: [source({ lexical_score: 0.3 })] });

      const user = await ask();
      await user.click(await screen.findByRole('button', { name: /1 source/i }));

      expect(await screen.findByText(/lexical/)).toBeInTheDocument();
      expect(screen.queryByText('0%')).toBeNull();
    });
  });
});
