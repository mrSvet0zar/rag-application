/**
 * The SSE client is hand-rolled frame parsing over a byte stream, which is the
 * most fragile code in the frontend: a network read boundary can fall anywhere,
 * including in the middle of a frame or of a UTF-8 character.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import api from './api';

/** A fetch stub whose body yields exactly the given chunks, in order. */
function streamOf(chunks, { ok = true, status = 200 } = {}) {
  const encoder = new TextEncoder();
  const queue = chunks.map((c) => (typeof c === 'string' ? encoder.encode(c) : c));
  let index = 0;

  return vi.fn().mockResolvedValue({
    ok,
    status,
    text: async () => 'erreur',
    body: {
      getReader: () => ({
        read: async () =>
          index < queue.length
            ? { done: false, value: queue[index++] }
            : { done: true, value: undefined },
      }),
    },
  });
}

function collect() {
  const calls = { sources: [], tokens: [], done: [], errors: [] };
  return {
    calls,
    handlers: {
      onSources: (s) => calls.sources.push(s),
      onToken: (t) => calls.tokens.push(t),
      onDone: (d) => calls.done.push(d),
      onError: (e) => calls.errors.push(e),
    },
  };
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

describe('chatStream', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('dispatches sources, tokens and done in order', async () => {
    vi.stubGlobal(
      'fetch',
      streamOf([
        'event: sources\ndata: {"retrieved_chunks":[{"id":1}]}\n\n',
        'event: token\ndata: {"text":"Bon"}\n\n',
        'event: token\ndata: {"text":"jour"}\n\n',
        'event: done\ndata: {"tokens_used":42,"conversation_id":"abc"}\n\n',
      ])
    );
    const { calls, handlers } = collect();

    api.chatStream({ question: 'q' }, handlers);
    await flush();

    expect(calls.sources).toEqual([[{ id: 1 }]]);
    expect(calls.tokens).toEqual(['Bon', 'jour']);
    expect(calls.done[0].tokens_used).toBe(42);
    expect(calls.errors).toEqual([]);
  });

  it('reassembles a frame split across two reads', async () => {
    // The real hazard: a read boundary lands in the middle of a frame.
    vi.stubGlobal('fetch', streamOf(['event: token\ndata: {"te', 'xt":"coupé"}\n\n']));
    const { calls, handlers } = collect();

    api.chatStream({ question: 'q' }, handlers);
    await flush();

    expect(calls.tokens).toEqual(['coupé']);
  });

  it('handles several frames arriving in one read', async () => {
    vi.stubGlobal(
      'fetch',
      streamOf([
        'event: token\ndata: {"text":"a"}\n\nevent: token\ndata: {"text":"b"}\n\n',
      ])
    );
    const { calls, handlers } = collect();

    api.chatStream({ question: 'q' }, handlers);
    await flush();

    expect(calls.tokens).toEqual(['a', 'b']);
  });

  it('decodes a multi-byte character split across reads', async () => {
    // "é" is two bytes in UTF-8; a naive decode per chunk would corrupt it.
    const bytes = new TextEncoder().encode('event: token\ndata: {"text":"é"}\n\n');
    const cut = bytes.indexOf(0xc3) + 1; // between the two bytes of "é"
    vi.stubGlobal('fetch', streamOf([bytes.slice(0, cut), bytes.slice(cut)]));
    const { calls, handlers } = collect();

    api.chatStream({ question: 'q' }, handlers);
    await flush();

    expect(calls.tokens).toEqual(['é']);
  });

  it('reports an error event instead of a token', async () => {
    vi.stubGlobal(
      'fetch',
      streamOf(['event: error\ndata: {"detail":"le modèle a échoué"}\n\n'])
    );
    const { calls, handlers } = collect();

    api.chatStream({ question: 'q' }, handlers);
    await flush();

    expect(calls.errors).toEqual(['le modèle a échoué']);
    expect(calls.tokens).toEqual([]);
  });

  it('reports a non-2xx response rather than parsing its body', async () => {
    vi.stubGlobal('fetch', streamOf([], { ok: false, status: 429 }));
    const { calls, handlers } = collect();

    api.chatStream({ question: 'q' }, handlers);
    await flush();

    expect(calls.errors).toHaveLength(1);
    expect(calls.tokens).toEqual([]);
  });

  it('ignores a malformed frame instead of throwing', async () => {
    vi.stubGlobal(
      'fetch',
      streamOf([
        'event: token\ndata: {ceci nest pas du json\n\n',
        'event: token\ndata: {"text":"ok"}\n\n',
      ])
    );
    const { calls, handlers } = collect();

    api.chatStream({ question: 'q' }, handlers);
    await flush();

    expect(calls.tokens).toEqual(['ok']);
  });

  it('returns an abort function that stops the request', async () => {
    vi.stubGlobal('fetch', streamOf([]));

    const abort = api.chatStream({ question: 'q' }, collect().handlers);

    expect(typeof abort).toBe('function');
    expect(() => abort()).not.toThrow();
  });

  it('sends the conversation id when there is one', async () => {
    const fetchStub = streamOf([]);
    vi.stubGlobal('fetch', fetchStub);

    api.chatStream({ question: 'q', conversationId: 'conv-1' }, collect().handlers);
    await flush();

    const body = JSON.parse(fetchStub.mock.calls[0][1].body);
    expect(body).toMatchObject({ question: 'q', conversation_id: 'conv-1' });
  });
});
