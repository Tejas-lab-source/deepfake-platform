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
export const ACCEPTED: Record<string, MediaKind> = {
  'image/jpeg': 'image',
  'image/png': 'image',
  'image/webp': 'image',
  'video/mp4': 'video',
  'video/webm': 'video',
  'video/quicktime': 'video',
};

export interface UploadLimits {
  maxImageMb: number;
  maxVideoMb: number;
}

export interface InspectedUpload {
  kind: MediaKind;
  mime: string;
  storageKey: string;
  sha256: string;
  sizeBytes: number;
}

/** Multer storing one file under a random name in `uploadDir` (the client's name is never used). */
export function createUpload(uploadDir: string, limits: UploadLimits): multer.Multer {
  mkdirSync(uploadDir, { recursive: true });
  return multer({
    storage: multer.diskStorage({
      destination: uploadDir,
      filename: (_req, _file, cb) => cb(null, randomUUID()),
    }),
    limits: {
      fileSize: Math.max(limits.maxImageMb, limits.maxVideoMb) * MB,
      files: 1,
      fields: 10,
      fieldSize: 1024,
    },
  });
}

/** Check magic bytes and per-kind size limits. Deletes the file if it is rejected. */
export async function inspectUpload(
  file: Express.Multer.File,
  limits: UploadLimits,
): Promise<InspectedUpload> {
  const reject = async (status: number, title: string, detail: string): Promise<never> => {
    await unlink(file.path).catch(() => undefined);
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
  if (size === 0) return reject(400, 'Empty file', 'The uploaded file is empty.');
  const limitMb = kind === 'image' ? limits.maxImageMb : limits.maxVideoMb;
  if (size > limitMb * MB) {
    return reject(413, 'File too large', `The limit for a ${kind} is ${limitMb} MB.`);
  }
  return {
    kind,
    mime: type.mime,
    storageKey: path.basename(file.filename),
    sha256: await sha256File(file.path),
    sizeBytes: size,
  };
}

/** The production limits from config. */
export const configLimits = (): UploadLimits => ({
  maxImageMb: config.MAX_IMAGE_MB,
  maxVideoMb: config.MAX_VIDEO_MB,
});
