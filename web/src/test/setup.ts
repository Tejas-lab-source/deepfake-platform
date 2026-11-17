import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach, vi } from 'vitest';

// jsdom has no object URLs; images and downloads only need a stable fake one.
let n = 0;
URL.createObjectURL = vi.fn(() => `blob:test-${++n}`);
URL.revokeObjectURL = vi.fn();

afterEach(() => cleanup());
