import { describe, it, expect, vi, beforeEach } from 'vitest';

/**
 * Mock node:fs/promises so loadConfig() reads a minimal valid YAML
 * instead of hitting the real filesystem. All top-level keys are
 * required z.object() fields without defaults, so they must be present.
 * Inner fields all have defaults and are filled in by Zod automatically.
 */
vi.mock('node:fs/promises', () => ({
  default: {
    readFile: vi.fn().mockResolvedValue(
      'gateway: {}\nmodels: {}\nsecurity: {}\nchannels: {}\nskills: {}\nstorage: {}\nscheduler: {}\nnodes: {}\n'
    ),
  },
}));

import { loadConfig } from './loadConfig.js';

describe('Config Module', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should load config successfully', async () => {
    const config = await loadConfig();

    expect(config).toBeDefined();
    // Config structure is validated by the schema, so just check it loads
    expect(config.gateway).toBeDefined();
    expect(config.gateway.port).toBe(18789);
    expect(config.models).toBeDefined();
    expect(config.security).toBeDefined();
    expect(config.channels).toBeDefined();
    expect(config.skills).toBeDefined();
    expect(config.storage).toBeDefined();
    expect(config.scheduler).toBeDefined();
    expect(config.nodes).toBeDefined();
  });
});
