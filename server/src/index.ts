import { createServer } from 'node:http';
import { Redis } from 'ioredis';
import mongoose from 'mongoose';
import { createApp } from './app.js';
import { config, configWarnings } from './config.js';
import { productionDeps } from './deps.js';
import { logger } from './logger.js';
import { reconcileStaleJobs, startMaintenance, sweepUploads } from './maintenance.js';
import { failAnalysis } from './queue/process.js';
import { closeQueues, getAnalysisQueue, getExplainQueue } from './queue/queues.js';
import { productionProcessDeps, startWorkers } from './queue/workers.js';
import { attachSockets, emitProgress } from './sockets.js';

const LIVE_STATES = new Set(['waiting', 'delayed', 'active', 'prioritized', 'waiting-children']);

async function main(): Promise<void> {
  await mongoose.connect(config.MONGO_URL);
  const redis = new Redis(config.REDIS_URL, { maxRetriesPerRequest: 1, connectTimeout: 2000 });
  redis.on('error', (err) => logger.warn({ err }, 'redis error'));

  const deps = productionDeps(redis);
  const server = createServer(createApp(deps));
  const io = attachSockets(server, {
    analyses: deps.repos.analyses,
    corsOrigins: deps.corsOrigins,
  });
  const processDeps = productionProcessDeps({
    repos: deps.repos,
    files: deps.files,
    ml: deps.ml,
    emit: emitProgress,
  });
  const workers = startWorkers(processDeps);

  const stopMaintenance = startMaintenance(async () => {
    const stale = await reconcileStaleJobs({
      repos: deps.repos,
      maxAttempts: config.JOB_ATTEMPTS,
      staleAfterMs: config.STALE_JOB_MINUTES * 60_000,
      async hasLiveJob(a) {
        const queue = a.final ? getExplainQueue() : getAnalysisQueue();
        const jobs = await queue.getJobs([...LIVE_STATES] as never[], 0, 1000);
        return jobs.some((j) => j?.data?.analysisId === a.id);
      },
      requeueAnalysis: async (id) => {
        await getAnalysisQueue().add(
          'analyze',
          { analysisId: id },
          { jobId: `analysis-${id}-r${Date.now()}` },
        );
      },
      requeueExplanation: (id) => processDeps.enqueueExplain(id),
      fail: (id, message) => failAnalysis(processDeps, id, message),
    });
    const swept = await sweepUploads(deps.files.uploadDir, config.MEDIA_TTL_HOURS * 3600_000);
    if (stale.checked > 0 || swept > 0)
      logger.info({ stale, uploadsRemoved: swept }, 'maintenance');
  }, config.CLEANUP_INTERVAL_MINUTES * 60_000);

  server.listen(config.PORT, () => {
    logger.info(
      {
        port: config.PORT,
        mockMode: config.MOCK_MODE,
        hiveMode: config.HIVE_MODE,
        groqMode: config.GROQ_MODE,
      },
      'server listening',
    );
    for (const warning of configWarnings(config)) logger.warn(warning);
  });

  const shutdown = async (signal: string): Promise<void> => {
    logger.info({ signal }, 'shutting down');
    stopMaintenance();
    await io.close();
    await workers.close();
    await closeQueues();
    redis.disconnect();
    await mongoose.disconnect();
    process.exit(0);
  };
  process.on('SIGTERM', () => void shutdown('SIGTERM'));
  process.on('SIGINT', () => void shutdown('SIGINT'));
}

main().catch((err: unknown) => {
  logger.fatal({ err }, 'startup failed');
  process.exit(1);
});
