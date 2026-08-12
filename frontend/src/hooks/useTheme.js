import { useCallback, useEffect, useState } from 'react';

/**
 * Persistent light/dark theme.
 *
 * Initial value mirrors the inline <head> script in index.html (localStorage
 * override, else system preference), so React state matches the class already
 * on <html> and there's no flash. `toggle` flips it, persists the choice, and
 * updates the `dark` class on the document root.
 */
function getInitialTheme() {
  if (typeof window === 'undefined') return 'light';
  const stored = localStorage.getItem('theme');
  if (stored === 'light' || stored === 'dark') return stored;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export default function useTheme() {
  const [theme, setTheme] = useState(getInitialTheme);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggle = useCallback(
    () => setTheme((t) => (t === 'dark' ? 'light' : 'dark')),
    []
  );

  return { theme, toggle };
}
