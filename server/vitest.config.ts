import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    // Quiet request logs in test output; tests that need logs capture them explicitly.
    env: { LOG_LEVEL: 'silent' },
  },
});
