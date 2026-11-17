/**
 * Files on the shared upload volume. Uploads live at <UPLOAD_DIR>/<random uuid>; images the ML
 * service derives (heatmaps, face crops, filtered copies) live at <UPLOAD_DIR>/derived/<32 hex>.png
 * and are referred to by the key "derived/<32 hex>.png".
 *
 * Only names matching the strict patterns below are ever turned into paths, and every path is
 * built from a fixed directory plus a validated basename, so no request can reach any other file.
 */
import { readFile, stat, unlink } from 'node:fs/promises';
import path from 'node:path';

/** A derived image key as the ML service returns it. */
export const DERIVED_KEY = /^derived\/([0-9a-f]{32})\.png$/;
/** A derived image name as it appears in a URL. */
export const DERIVED_NAME = /^[0-9a-f]{32}\.png$/;
/** Upload storage keys are random UUIDs chosen by the server. */
const STORAGE_KEY = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

const PNG_MAGIC = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
/** Derived images are small; anything bigger is not one of ours. */
const MAX_DERIVED_BYTES = 8 * 1024 * 1024;

/** Every derived image key mentioned anywhere in a value (an ML result, a comparison). */
export function collectDerivedKeys(value: unknown, out = new Set<string>()): string[] {
  if (typeof value === 'string') {
    if (DERIVED_KEY.test(value)) out.add(value);
  } else if (Array.isArray(value)) {
    for (const v of value) collectDerivedKeys(v, out);
  } else if (value && typeof value === 'object') {
    for (const v of Object.values(value)) collectDerivedKeys(v, out);
  }
  return [...out];
}

export const derivedNameOf = (key: string): string | null => {
  const m = DERIVED_KEY.exec(key);
  return m ? `${m[1]}.png` : null;
};

export type DerivedRead =
  { ok: true; bytes: Buffer } | { ok: false; reason: 'invalid_name' | 'missing' | 'corrupt' };

export interface FileStore {
  readonly uploadDir: string;
  /** Absolute path of an upload, or null for a storage key the server could not have made. */
  uploadPath(storageKey: string): string | null;
  removeUpload(storageKey: string): Promise<void>;
  readDerived(name: string): Promise<DerivedRead>;
  removeDerived(keys: string[]): Promise<number>;
}

export function diskFileStore(uploadDir: string): FileStore {
  const root = path.resolve(uploadDir);
  const derivedDir = path.join(root, 'derived');
  const uploadPath = (storageKey: string): string | null => {
    const base = path.basename(storageKey);
    return base === storageKey && STORAGE_KEY.test(base) ? path.join(root, base) : null;
  };
  return {
    uploadDir: root,
    uploadPath,
    async removeUpload(storageKey) {
      const p = uploadPath(storageKey);
      if (p) await unlink(p).catch(() => undefined);
    },
    async readDerived(name) {
      if (!DERIVED_NAME.test(name)) return { ok: false, reason: 'invalid_name' };
      const file = path.join(derivedDir, name);
      try {
        const info = await stat(file);
        if (!info.isFile() || info.size > MAX_DERIVED_BYTES || info.size < PNG_MAGIC.length) {
          return { ok: false, reason: 'corrupt' };
        }
        const bytes = await readFile(file);
        return bytes.subarray(0, PNG_MAGIC.length).equals(PNG_MAGIC)
          ? { ok: true, bytes }
          : { ok: false, reason: 'corrupt' };
      } catch {
        return { ok: false, reason: 'missing' };
      }
    },
    async removeDerived(keys) {
      let removed = 0;
      for (const key of keys) {
        const name = derivedNameOf(key);
        if (!name) continue;
        try {
          await unlink(path.join(derivedDir, name));
          removed++;
        } catch {
          // already gone
        }
      }
      return removed;
    },
  };
}
