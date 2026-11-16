# Phase 3: offline training and calibration

These scripts train the invariance head, calibrate the service, and evaluate the six head
schemes. They run on **CPU**, resume after interruption, and are tested end to end on a
generated 20-face fixture. Nothing they produce is committed to Git.

> **Status.** Every stage is implemented and runs on the fixture. **No real-data result
> exists yet**: this project has no Celeb-DF v2 access, and real-data split assignment is
> deliberately blocked until the official list's identity graph has been reviewed (see
> "Real data" below).

## Protocol

Data come from the **official Celeb-DF v2 test list** (518 videos), split 60/20/20 into
train / validation / test by **identity component**. Validation is split again into
**val-A** and **val-B** by identity group.

Every report carries this label, and it applies to every number produced:

> Internal identity-disjoint split of the Celeb-DF v2 official test list (60/20/20 by
> identity component). NOT comparable with published Celeb-DF v2 benchmark results.

| Split | Used by | For |
|---|---|---|
| train | `train_head.py` | fitting the heads |
| val-A | `train_head.py`, `calibrate.py` | early stopping, the λp choice, fusion weights |
| val-B | `calibrate.py`, `eval_lofo.py` | temperature, threshold, spread limit, margin |
| test | `eval_lofo.py` only | reporting |

`load_split()` enforces this: `train_head` and `calibrate` cannot open the test split, and
`train_head` drops val-B before training.

## Stages

Run from `ml-service/` with the ML environment (`requirements.txt`, `requirements-ml.txt`,
`requirements-offline.txt`) and the weights from `scripts/download_weights.sh`.

| # | Command | Output |
|---|---|---|
| 0 | `python -m offline.celebdf_graph --list <List_of_testing_videos.txt>` | Read-only identity-graph report |
| 1 | `python -m offline.extract_faces --data-root <dataset> --out artifacts/faces` | Crops (lossless PNG) and `manifest.json` |
| 2 | `python -m offline.cache_features --faces artifacts/faces --out artifacts/cache` | `features_{train,val,test}.npz`, `meta.json` |
| 3 | `python -m offline.train_head --cache artifacts/cache --out-dir artifacts/heads --final` | 45 LOFO heads, `index.json`, `weights/head.pt`, `final.json` |
| 4 | `python -m offline.calibrate --cache artifacts/cache` | `weights/model_card.json` |
| 5 | `python -m offline.eval_lofo --cache artifacts/cache --faces artifacts/faces` | `artifacts/eval/lofo.json` and printed tables |

Stages 1 and 2 accept `--limit N` and `--max-minutes M`: they stop cleanly and continue on
the next run. Each video's output is written atomically (temporary file, then rename) and
skipped when it already exists. Stage 2 refuses to mix outputs from a different manifest.

### The fixture

```bash
python -m offline.run_fixture --workdir artifacts/fixture_run
```

This generates the fixture, runs stages 0–5, then loads the resulting `head.pt` and
`model_card.json` into the service's real pipeline and analyses one image. Its outputs go
under `--workdir`, never into `weights/`.

**Fixture results are plumbing tests, not accuracy results.** The 20 faces are variants of
one public-domain photo (scikit-image's astronaut, NASA) and the real/fake labels are
arbitrary. The manifest, cache, heads, model card and evaluation report are all flagged
`fixture: true`, and the evaluation printout tags every criterion as meaningless.

Measured on this project's 4-core CPU VM (one run; times vary):

| Stage | Time |
|---|---|
| Extract faces (10 videos, 20 faces) | 1.7 s |
| Cache features (about 170 GenD passes) | 92 s |
| Train 45 heads + shipped head (with λp grid) | 10 s |
| Calibrate | 3 s |
| Evaluate, heatmap IoU on all 4 test faces | 189 s |
| Service check | 14 s |
| **Total** | **about 5 min** |

## The six head schemes

Exactly as defined in `CLAUDE.md` §5 item 3; see `schemes.py`.

| Scheme | Trained on | Invariance loss |
|---|---|---|
| H-a | No head (GenD as released) | — |
| H-b | Clean + bilateral smoothing only | No |
| H-c | Clean + random F1–F3 (the critical control) | No |
| H-d | Clean + generic augmentations | Yes |
| H-e | Clean + random F1–F3 | Yes |
| H-f | Clean + random F1–F3, hardest-of-K (the headline) | Yes |

- Loss: `BCE(h(z_x), y) + BCE(h(z_e), y) + λp·(h(z_x) − h(z_e))²`, class-balanced; λp = 0
  for schemes without invariance loss, 1.0 otherwise.
- Every head starts from GenD's own head (`w = W[1] − W[0]`, `b = b[1] − b[0]`) on
  L2-normalised features.
- Three folds each hold out one of F1–F3; F4 is always held out. Three seeds.
- **H-b's F1 fold is NOT HELD OUT** (its bilateral copy is part of F1) and is excluded from
  aggregate results.
- **Seeds.** Each seed sets the batch order and, for H-b to H-e, the random copy choice.
  H-a has no training, so its three seeds are identical. H-f's hardest-of-K is an argmax,
  so its only randomness is the batch order. When all training faces fit in one batch (the
  fixture: 8 faces, batch 64), every seed makes the same choices, and the heads differ only
  by floating-point rounding (at most 1.8e-7 per weight on the fixture). With real data
  (thousands of faces, batches of 64) the seed changes which faces share each step, so H-f's
  seeds genuinely differ. Tests check both cases, and that the same seed reproduces a head
  bit for bit.
- **Shipped head:** H-f on all of F1–F3, seed 0. λp is chosen from {0, 0.1, 0.3, 1, 3, 10}:
  the lowest val-A flip rate among values keeping val-A clean AUC within 1 point of H-a.
  If that rule can't be applied, λp = 1.0 and the reason is recorded.

## Feature cache inventory

All draws are seeded by (face ID, copy name), so any copy can be regenerated exactly.

| Copy | Family | Strength | Splits |
|---|---|---|---|
| `clean` | — | 0 | train, val, test |
| `F1_lo`, `F2_lo`, `F3_lo` | F1 / F2 / F3 | U(0.30, 0.65) | train, val, test |
| `F1_hi`, `F2_hi`, `F3_hi` | F1 / F2 / F3 | U(0.65, 1.00) | train, val, test |
| `F1_bilat` | F1, bilateral only | U(0.30, 1.00) | train |
| `generic` | colour jitter + blur + JPEG | random parameters | train |
| `mix` | two of F1–F3 | U(0.30, 1.00) each | val, test |
| `F4` | GFPGAN | U(0.50, 1.00) | test, only if GFPGAN is installed |

- The service's inference copies (F1, F2, F3, mix) are not recomputed: each single-family
  slot reuses `_lo` or `_hi` with probability ½, which matches the service's U(0.30, 1.00)
  strength distribution. Only `mix` is extra.
- SBI logits and `beautifiedLikelihood` are computed for val and test clean crops only.
- GenD passes per face: 9 train, 8 val, 9 test (8 without F4). Features are float16
  (about 2 KB per copy).

## Calibration (`calibrate.py`)

| Value | Data | Rule |
|---|---|---|
| Fusion weights | val-A | Logistic regression on [M1, SBI, spread, beautifiedLikelihood], C = 1, class-balanced |
| Temperature | val-B | Minimise log-loss over T in [0.05, 20] |
| Threshold | val-B | Maximise balanced accuracy within [0.05, 0.95] |
| `spread_limit` | val-B | 95th percentile of spread |
| `uncertain_margin` | val-B | Largest m in 0.00–0.30 keeping UNCERTAIN ≤ 20% (all four rules) |
| `confident_margin`, `detector_disagree_limit` | — | Defaults, recorded as not fitted |

The UNCERTAIN rate comes from the service's own `app.pipeline.decide`. Every value records
its method, data and a video-level bootstrap 95% CI.

**Insufficient data is reported, not hidden.** If either val half has fewer than 20 videos
per class or fewer than 100 faces, or a fit is impossible, the card is written with
`calibration_status: insufficient_validation_data` and the reasons in `warnings`. The
service refuses such a card, and any fixture card, unless `ALLOW_UNVALIDATED_MODEL_CARD=true`
(tests only). It also refuses a `head.pt` whose SHA-256 differs from the card's
`head_sha256`.

## Evaluation (`eval_lofo.py`)

Per scheme × fold × seed, each head gets its own temperature and threshold, fitted on
val-B using clean faces and the fold's two training families (never the held-out family).
Decisions are forced (p ≥ threshold is FAKE).

| Metric | Definition |
|---|---|
| `clean_auc` | AUC of the clean logit |
| `fpr_beautified_real` | Real faces' held-out-family copies (lo and hi) predicted FAKE |
| `fnr_beautified_fake` | Fake faces' held-out-family copies predicted REAL |
| `flip_rate` | Held-out copies whose decision differs from the clean decision |
| `uncertain_rate` | Full UNCERTAIN rule on clean test inputs, using the fold's copy set (see below) |
| `clean_error_forced` | Clean error with UNCERTAIN forced to a decision |
| `f4_*` | The same on F4 copies, when available |
| Heatmap IoU | Seed 0, top 20% regions, clean vs held-out `_hi` copy, F3 warped back; `--iou-faces 100` for real data |

**No parameter of a fold sees the held-out family.** Each fold's copy set for the
UNCERTAIN rule is clean + the two training families. The held-out family's copy is dropped,
and so is `mix`, which may contain it. Temperature, threshold, spread limit and uncertain
margin are all fitted on val-B with that set, and the test UNCERTAIN rate uses it too. The
held-out family is used only to measure FPR, FNR, flip rate and heatmap IoU. The shipped
head has no held-out family, so its calibration and ablation use the service's full
5-copy set. Regression tests scramble the held-out family's features and check that no
fitted value changes.

Results are given as seed mean ± std, with a 95% bootstrap CI that resamples whole test
videos. The run also produces the inference-time ablation of the shipped head on clean test
inputs (base → +M1 → +M2 → +calibration → +fusion) and the three pre-registered criteria as
MET, NOT MET or NOT EVALUABLE.

## Real data

1. Request Celeb-DF v2 from its authors (official page:
   `github.com/yuezunli/celeb-deepfakeforensics`). Research use only; do not redistribute
   it or anything derived from it (crops, caches).
2. Run the read-only graph report on `List_of_testing_videos.txt` and share it for review:
   connected components, largest component, and how many fakes a random identity-level
   split would discard.
3. **Real-data split assignment is not implemented yet.** `extract_faces.py` exits with
   code 3 on any non-fixture dataset. The split strategy is decided after reviewing step 2.

### Limits of running the full dataset in this environment

- **Time:** about 4–5 s per face for caching on the 4-core CPU VM, about 5–6 hours for
  roughly 4,100 faces. This is extrapolated from the fixture rate, not measured on
  Celeb-DF. Heatmap IoU on 100 faces × 18 scheme/fold pairs adds roughly an hour.
- **Container lifetime:** the cloud container is temporary. Resuming only works while the
  same container survives; if it is reclaimed, all progress and outputs are lost.
- **Persistence:** nothing derived from Celeb-DF may be pushed to GitHub, and private
  artifact storage has been deferred.
- **Download:** the dataset must be fetched into the container through its network policy
  and must fit the free disk space.
- **No GPU:** all figures are CPU only.

## Not in Phase 3

- `hive_batch.py` (moved to Phase 4).
- The `evalRuns` server endpoint: results stay as JSON files.
- F4 validation: GFPGAN is not installed, so F4 copies are skipped and recorded.
