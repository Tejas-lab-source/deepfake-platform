import { createServer } from 'node:http';
import mongoose from 'mongoose';
import { createApp } from './app.js';
import { config } from './config.js';
import { logger } from './logger.js';
import { closeQueues } from './queue/queues.js';
import { startWorkers } from './queue/workers.js';
import { attachSockets } from './sockets.js';

async function main(): Promise<void> {
  await mongoose.connect(config.MONGO_URL);
  const server = createServer(createApp());
  const io = attachSockets(server);
  const workers = startWorkers();
  server.listen(config.PORT, () => {
    logger.info({ port: config.PORT, mockMode: config.MOCK_MODE }, 'server listening');
  });

  const shutdown = async (signal: string): Promise<void> => {
    logger.info({ signal }, 'shutting down');
    await io.close();
    await workers.close();
    await closeQueues();
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
