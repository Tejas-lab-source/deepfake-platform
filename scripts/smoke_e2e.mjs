#!/usr/bin/env node
// End-to-end smoke test: register -> upload -> poll -> verdict + explanation.
// Usage: node scripts/smoke_e2e.mjs [API_URL]   (default http://localhost:4000)
// Needs Node 22+ and a running stack (docker compose up). No dependencies.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const base = `${process.argv[2] ?? process.env.API_URL ?? 'http://localhost:4000'}/api/v1`;
const fixture = new URL('../ml-service/tests/fixtures/sample.png', import.meta.url);
const VERDICTS = ['FAKE', 'REAL', 'UNCERTAIN'];
const BADGES = ['verified_by_2', 'second_check_disagrees', 'checks_disagree', 'hive_hint', 'single_check'];

let step = 0;
const ok = (msg) => console.log(`ok ${++step} - ${msg}`);

async function call(method, path, { token, json, form } = {}) {
  const headers = {};
  if (token) headers.authorization = `Bearer ${token}`;
  if (json) headers['content-type'] = 'application/json';
  const res = await fetch(base + path, { method, headers, body: json ? JSON.stringify(json) : form });
  const text = await res.text();
  return { status: res.status, type: res.headers.get('content-type') ?? '', body: text ? JSON.parse(text) : null };
}

async function analyse(token, bytes, type, externalCheck) {
  const form = new FormData();
  form.append('file', new Blob([bytes], { type }), 'upload.bin');
  form.append('externalCheck', String(externalCheck));
  const created = await call('POST', '/analyses', { token, form });
  if (created.status !== 202) return created;
  const { analysisId } = created.body;
  for (let i = 0; i < 60; i++) {
    const res = await call('GET', `/analyses/${analysisId}`, { token });
    if (['done', 'failed'].includes(res.body.status)) return { ...created, result: res.body };
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error(`analysis ${analysisId} did not finish within 30 s`);
}

const health = await call('GET', '/health');
assert.equal(health.status, 200);
ok(`server healthy (mockMode=${health.body.mockMode})`);

const creds = { email: `smoke-${Date.now()}@example.com`, password: 'smoke-test-pw' };
const reg = await call('POST', '/auth/register', { json: creds });
assert.equal(reg.status, 201);
const login = await call('POST', '/auth/login', { json: creds });
assert.equal(login.status, 200);
const refreshed = await call('POST', '/auth/refresh', { json: { refreshToken: login.body.refreshToken } });
assert.equal(refreshed.status, 200);
const token = refreshed.body.accessToken;
ok('register, login and refresh');

const anon = await call('GET', '/analyses');
assert.equal(anon.status, 401);
assert.match(anon.type, /application\/problem\+json/);
ok('unauthenticated request rejected with problem+json');

const png = await readFile(fixture);
const withCheck = await analyse(token, png, 'image/png', true);
assert.equal(withCheck.status, 202);
assert.ok(withCheck.body.analysisId && withCheck.body.jobId);
const r = withCheck.result;
assert.equal(r.status, 'done', r.error ?? '');
assert.ok(VERDICTS.includes(r.final.verdict));
assert.ok(BADGES.includes(r.final.badge));
assert.notEqual(r.final.badge, 'single_check');
assert.ok(VERDICTS.includes(r.ours.verdict));
assert.equal(r.external.status, 'ok');
assert.equal(typeof r.external.deepfakeScore, 'number');
assert.ok(r.ml.faces.length > 0);
ok(`image with external check: final=${r.final.verdict} badge=${r.final.badge} rule=${r.final.ruleApplied}`);

assert.ok(r.explanation.summary && r.explanation.why && r.explanation.caveats.length > 0);
const explanationText = JSON.stringify(r.explanation);
assert.match(explanationText, /automated/i);
assert.doesNotMatch(explanationText, /authentic|proven fake/i);
ok(`explanation present (fallbackUsed=${r.explanation.fallbackUsed})`);

const again = await analyse(token, png, 'image/png', true);
assert.deepEqual(again.result.final, r.final);
assert.equal(again.result.ours.probability, r.ours.probability);
ok('same file gives the same result');

const solo = await analyse(token, png, 'image/png', false);
assert.equal(solo.result.final.badge, 'single_check');
assert.equal(solo.result.final.verdict, solo.result.ours.verdict);
assert.equal(solo.result.external, null);
ok('external check off -> single_check, our verdict kept');

const bogus = await analyse(token, Buffer.from('not an image at all'), 'image/png', false);
assert.equal(bogus.status, 415);
ok('non-image upload rejected by magic bytes (415)');

const list = await call('GET', '/analyses', { token });
assert.equal(list.body.total, 3);
ok('history lists the 3 analyses');

const del = await call('DELETE', `/analyses/${solo.body.analysisId}`, { token });
assert.equal(del.status, 204);
assert.equal((await call('GET', `/analyses/${solo.body.analysisId}`, { token })).status, 404);
ok('delete removes the analysis');

console.log(`\nSMOKE TEST PASSED (${step} checks)`);
