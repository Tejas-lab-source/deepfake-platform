import { randomUUID } from 'node:crypto';
import { mkdirSync } from 'node:fs';
import { stat, unlink } from 'node:fs/promises';
import path from 'node:path';
import { fileTypeFromFile } from 'file-type';
import multer from 'multer';
import { config } from '../config.js';
import { sha256File } from '../hash.js';
import type { MediaKind } from '../types.js';
import { HttpProblem } from './problem.js';

const MB = 1024 * 1024;

/** Accepted types, decided from the file's magic bytes, never from its name. */
const ACCEPTED: Record<string, MediaKind> = {
  'image/jpeg': 'image',
  'image/png': 'image',
  'image/webp': 'image',
  'video/mp4': 'video',
  'video/webm': 'video',
  'video/quicktime': 'video',
};

mkdirSync(config.UPLOAD_DIR, { recursive: true });

export const upload = multer({
  storage: multer.diskStorage({
    destination: config.UPLOAD_DIR,
    // Random name: the client's filename is never used on disk.
    filename: (_req, _file, cb) => cb(null, randomUUID()),
  }),
  limits: { fileSize: config.MAX_VIDEO_MB * MB, files: 1 },
});

export interface InspectedUpload {
  kind: MediaKind;
  mime: string;
  storageKey: string;
  sha256: string;
  sizeBytes: number;
}

export const storagePath = (storageKey: string): string =>
  path.join(config.UPLOAD_DIR, path.basename(storageKey));

export async function removeStored(storageKey: string): Promise<void> {
  await unlink(storagePath(storageKey)).catch(() => undefined);
}

/** Check magic bytes and per-kind size limits. Deletes the file if it is rejected. */
export async function inspectUpload(file: Express.Multer.File): Promise<InspectedUpload> {
  const reject = async (status: number, title: string, detail: string): Promise<never> => {
    await removeStored(file.filename);
    throw new HttpProblem(status, title, detail);
  };
  const type = await fileTypeFromFile(file.path);
  const kind = type ? ACCEPTED[type.mime] : undefined;
  if (!type || !kind) {
    return reject(
      415,
      'Unsupported media type',
      'Upload a JPEG, PNG or WebP image, or an MP4, WebM or MOV video.',
    );
  }
  const { size } = await stat(file.path);
  const limitMb = kind === 'image' ? config.MAX_IMAGE_MB : config.MAX_VIDEO_MB;
  if (size > limitMb * MB) {
    return reject(413, 'File too large', `The limit for a ${kind} is ${limitMb} MB.`);
  }
  return {
    kind,
    mime: type.mime,
    storageKey: file.filename,
    sha256: await sha256File(file.path),
    sizeBytes: size,
  };
}
