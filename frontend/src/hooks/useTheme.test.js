/**
 * The theme is persisted and applied before paint, so its rules are easy to
 * break silently: a wrong default means a flash of the wrong colours on every
 * visit, which no other test would notice.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

import useTheme from './useTheme';

function systemPrefers(dark) {
  vi.stubGlobal(
    'matchMedia',
    vi.fn().mockReturnValue({
      matches: dark,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    })
  );
}

describe('useTheme', () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.classList.remove('dark');
    systemPrefers(false);
  });

  it('follows the system preference on a first visit', () => {
    systemPrefers(true);

    const { result } = renderHook(() => useTheme());

    expect(result.current.theme).toBe('dark');
    expect(document.documentElement).toHaveClass('dark');
  });

  it('prefers a stored choice over the system preference', () => {
    systemPrefers(true);
    localStorage.setItem('theme', 'light');

    const { result } = renderHook(() => useTheme());

    expect(result.current.theme).toBe('light');
    expect(document.documentElement).not.toHaveClass('dark');
  });

  it('toggling flips the class on the document root', () => {
    const { result } = renderHook(() => useTheme());

    act(() => result.current.toggle());

    expect(result.current.theme).toBe('dark');
    expect(document.documentElement).toHaveClass('dark');
  });

  it('persists the choice so the next visit does not flash', () => {
    const { result } = renderHook(() => useTheme());

    act(() => result.current.toggle());

    expect(localStorage.getItem('theme')).toBe('dark');
  });

  it('ignores a nonsense stored value and falls back to the system', () => {
    localStorage.setItem('theme', 'chartreuse');
    systemPrefers(true);

    const { result } = renderHook(() => useTheme());

    expect(result.current.theme).toBe('dark');
  });
});
