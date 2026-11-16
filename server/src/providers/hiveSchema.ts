/**
 * Hive v3 response validation and score extraction.
 *
 * Documented shape (docs.thehive.ai, "AI-Generated and Deepfake Content Detection
 * Playground", checked 16 Nov 2026):
 *
 *   { task_id, model, metadata,
 *     output: [ { extra: [{name: "frame_index", value}, {name: "timestamp", value}],
 *                 classes: [{class, value}, ...] }, ...one entry per frame ] }
 *
 * Scores are in `value`. The legacy v2 `score` field is rejected, not silently accepted:
 * an unexpected format must be noticed and the parser adapted deliberately.
 */
import { z } from 'zod';
import type { FrameScore } from './verifier.js';

export const DEEPFAKE_CLASS = 'deepfake';
export const AI_GENERATED_CLASS = 'ai_generated';
const MAX_FRAMES = 10_000;

const classItem = z
  .object({ class: z.string().min(1), value: z.number().finite().min(0).max(1) })
  .passthrough();
const extraItem = z
  .object({ name: z.string(), value: z.union([z.number(), z.string(), z.null()]) })
  .passthrough();
const frame = z
  .object({ extra: z.array(extraItem).optional(), classes: z.array(classItem).min(1) })
  .passthrough();
const response = z
  .object({
    task_id: z.string().optional(),
    model: z.string().optional(),
    output: z.array(frame).min(1).max(MAX_FRAMES),
  })
  .passthrough();

export type HiveParse =
  | {
      ok: true;
      deepfakeScore: number;
      aiGeneratedScore: number;
      frames: FrameScore[];
      taskId: string | null;
      model: string | null;
    }
  | { ok: false; reason: string; detail: string };

function usesLegacyScoreField(json: unknown): boolean {
  if (!json || typeof json !== 'object' || !('output' in json)) return false;
  const output = (json as { output: unknown }).output;
  if (!Array.isArray(output)) return false;
  return output.some(
    (f) =>
      Array.isArray((f as { classes?: unknown }).classes) &&
      ((f as { classes: unknown[] }).classes as Record<string, unknown>[]).some(
        (c) => c && typeof c === 'object' && 'score' in c && !('value' in c),
      ),
  );
}

function extraNumber(extra: { name: string; value: unknown }[] | undefined, name: string) {
  const v = extra?.find((e) => e.name === name)?.value;
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** Validate a Hive v3 response and extract per-frame and maximum scores. */
export function parseHiveResponse(json: unknown): HiveParse {
  if (usesLegacyScoreField(json)) {
    return {
      ok: false,
      reason: 'legacy_score_field',
      detail: 'classes use the v2 "score" field; the v3 API documents "value"',
    };
  }
  const parsed = response.safeParse(json);
  if (!parsed.success) {
    const issue = parsed.error.issues[0];
    const where = issue ? issue.path.join('.') || 'response' : 'response';
    return { ok: false, reason: 'schema', detail: `${where}: ${issue?.message ?? 'invalid'}` };
  }
  const frames: FrameScore[] = [];
  for (const [i, f] of parsed.data.output.entries()) {
    const values = new Map<string, number>();
    for (const c of f.classes) {
      if (values.has(c.class)) {
        return { ok: false, reason: 'duplicate_class', detail: `frame ${i}: ${c.class} twice` };
      }
      values.set(c.class, c.value);
    }
    const deepfake = values.get(DEEPFAKE_CLASS);
    const aiGenerated = values.get(AI_GENERATED_CLASS);
    if (deepfake === undefined || aiGenerated === undefined) {
      const missing = deepfake === undefined ? DEEPFAKE_CLASS : AI_GENERATED_CLASS;
      return { ok: false, reason: 'missing_class', detail: `frame ${i}: no "${missing}" class` };
    }
    frames.push({
      frameIndex: extraNumber(f.extra, 'frame_index') ?? i,
      timestamp: extraNumber(f.extra, 'timestamp'),
      deepfake,
      aiGenerated,
    });
  }
  return {
    ok: true,
    // Video: Hive's recommended rule is "any frame" above the threshold, i.e. the maximum.
    deepfakeScore: Math.max(...frames.map((f) => f.deepfake)),
    aiGeneratedScore: Math.max(...frames.map((f) => f.aiGenerated)),
    frames,
    taskId: parsed.data.task_id ?? null,
    model: parsed.data.model ?? null,
  };
}
