import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach, vi } from 'vitest';

// Without this, a component from one test is still mounted during the next and
// queries match the wrong tree.
afterEach(cleanup);

// jsdom implements no layout, so scroll APIs simply do not exist on elements.
// The chat view scrolls to the newest message on every render.
window.HTMLElement.prototype.scrollIntoView = vi.fn();
