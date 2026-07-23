# BTEH Elephant Re-ID — Production Runbook

**Scope:** Boulders Trust Elephant Herd (BTEH) re-identification pipeline.
**Selected system:** body `miewid` + ear `ear_miewid_projected` (Platt calibration, projected fusion).
**Status:** Model selection frozen. See `reports/bteh_model_selection_report.md` for full rationale.

---

## Contents

1. [Environment setup](#1-environment-setup)
2. [Configure roots](#2-configure-roots)
3. [Build image manifest](#3-build-image-manifest)
4. [Define splits](#4-define-splits)
5. [Pilot run and crop QA](#5-pilot-run-and-crop-qa)
6. [Full crop extraction](#6-full-crop-extraction)
7. [Partition assignment](#7-partition-assignment)
8. [Frozen embeddings](#8-frozen-embeddings)
9. [Calibration and eval (reference)](#9-calibration-and-eval-reference)
10. [Projection training and transform](#10-projection-training-and-transform)
11. [Selected calibration](#11-selected-calibration)
12. [Production index build](#12-production-index-build)
13. [Adding a new elephant without retraining](#13-adding-a-new-elephant-without-retraining)
14. [Refresh and rebuild triggers](#14-refresh-and-rebuild-triggers)
15. [Artifact fingerprint and version policy](#15-artifact-fingerprint-and-version-policy)
16. [Human crop QA checklist](#16-human-crop-qa-checklist)

---

## 1 · Environment Setup

### 1a · Clone repository and create conda environment

```bash
git clone <repo-url> giraffe-identification-ai-tool
cd giraffe-identification-ai-tool
conda env create -f environment.yaml
conda activate <env-name-from-yaml>
```

### 1b · Headless OpenCV caveat

On headless servers (no display), import `cv2` before any display-dependent call
or it may abort with a `DISPLAY` error.  Install the headless variant:

```bash
pip install opencv-python-headless
# Do NOT install both opencv-python and opencv-python-headless in the same env.
```

Verify:
```bash
python -c "import cv2; print(cv2.__version__)"
```

### 1c · GPU / cuDNN note

Full-body and ear detection/embedding runs on GPU by default.
**This host requires `--disable-cudnn`** for the full-crop step (see §6).
FAISS CPU index is used for production; no GPU-FAISS required.

### 1d · Install Python dependencies

```bash
pip install -r requirements_elephant.txt
# If a dependency reinstalls opencv-python, remove it and restore the pinned
# opencv-python-headless wheel from requirements_elephant.txt.
```

---

## 2 · Configure Roots

Copy `.env.template` to `.env` and set:

```dotenv
# Read-only tree of original BTEH images (never modified by pipeline).
BTEH_SOURCE_ROOT=/absolute/path/to/BTEH_images

# Writable artifact tree (pipeline writes here; versioned by schema version).
BTEH_ARTIFACT_ROOT=/absolute/path/to/BTEH_reid_artifacts
```

For commands that consume already-versioned artifacts:

```bash
export BTEH_VERSION_ROOT="$BTEH_ARTIFACT_ROOT/v1"
```

Do **not** hard-code absolute paths anywhere in pipeline scripts.
All paths are derived from these two environment variables at runtime.

Verify:
```bash
python -c "from configs.config_bteh import BTEH_SOURCE_ROOT, ARTIFACT_VERSION_ROOT; print(BTEH_SOURCE_ROOT, ARTIFACT_VERSION_ROOT)"
```

---

## 3 · Build Image Manifest

The manifest enumerates all BTEH images, assigns a stable `image_id` and
`individual_id` per row, and fingerprints the source tree.

```bash
python -m pipeline.bteh_manifest \
    --source-root "$BTEH_SOURCE_ROOT" \
    --artifact-root "$BTEH_ARTIFACT_ROOT"
```

Output: `$BTEH_ARTIFACT_ROOT/v1/manifests/bteh_image_manifest.parquet`

**Rules:**
- Directories that look like UUIDs or 32-char hex strings are automatically
  excluded as unresolved review buckets (see `config_bteh.is_uuid_dir`).
- Do **not** include images tagged "Not for AI" or from unresolved identities.
  Human review must clear all images before they enter the manifest.
- Bumping `ARTIFACT_SCHEMA_VERSION` in `config_bteh.py` invalidates all
  prior artifacts; all downstream steps must be rerun from scratch.

---

## 4 · Define Splits

Splits assign each image to `reference` (gallery) or `query` (probe) based on
leave-one-session-out (LOIO) partitioning.

```bash
python -m pipeline.bteh_splits \
    --manifest "$BTEH_VERSION_ROOT/manifests/bteh_image_manifest.parquet" \
    --output "$BTEH_VERSION_ROOT/splits/bteh_splits.parquet"
```

Output: `$BTEH_ARTIFACT_ROOT/v1/splits/bteh_splits.parquet`

Split fingerprint is embedded in every downstream artifact for traceability.
Changing the split logic requires rerunning all steps from §5 onward.

---

## 5 · Pilot Run and Crop QA

Run detection on a small pilot set to verify bounding-box quality before
committing to the full corpus.

```bash
python -m pipeline.bteh_crop_pilot sample \
    --manifest "$BTEH_VERSION_ROOT/manifests/bteh_image_manifest.parquet" \
    --splits "$BTEH_VERSION_ROOT/splits/bteh_splits.parquet" \
    --output-dir "$BTEH_VERSION_ROOT/pilot" \
    --n-pilot 120 \
    --n-review 12
```

Output: `$BTEH_ARTIFACT_ROOT/v1/pilot/`

Run the detector on the pilot manifest, then create contact sheets:

```bash
SOURCE_FP=$(python -c "import json; print(json.load(open('$BTEH_VERSION_ROOT/manifests/bteh_image_manifest.json'))['manifest_fingerprint'])")
SPLIT_FP=$(python -c "import json; print(json.load(open('$BTEH_VERSION_ROOT/splits/bteh_splits.json'))['splits_fingerprint'])")

python -m pipeline.step_1_run_detection_to_crop --bteh \
    --image-manifest "$BTEH_VERSION_ROOT/pilot/bteh_pilot_manifest.parquet" \
    --crop-manifest "$BTEH_VERSION_ROOT/pilot/crop_manifest.parquet" \
    --crops-dir "$BTEH_VERSION_ROOT/pilot/crops" \
    --source-fingerprint "$SOURCE_FP" \
    --split-fingerprint "$SPLIT_FP" \
    --disable-cudnn

python -m pipeline.bteh_crop_pilot report \
    --pilot-manifest "$BTEH_VERSION_ROOT/pilot/bteh_pilot_manifest.parquet" \
    --crop-manifest "$BTEH_VERSION_ROOT/pilot/crop_manifest.parquet" \
    --source-root "$BTEH_SOURCE_ROOT" \
    --output-dir "$BTEH_VERSION_ROOT/pilot/report" \
    --check-files
```

### Human crop QA (pilot)

See §16 for the full QA checklist.  At minimum:
- Open `pilot/pilot_visual_review.csv` in a spreadsheet.
- Inspect random sample of accepted crops for body detection quality.
- Inspect ear detection crops for alignment (ear centred, not clipped).
- Flag any `detector_status=failed` rows for investigation.
- Update `review_status` column (`accepted` / `rejected`) before proceeding.

Do **not** proceed to full crop if pilot failure rate > 5%.

---

## 6 · Full Crop Extraction

Extract body and ear crops for all accepted images.

```bash
# Use --disable-cudnn on this host to avoid CUDA errors.
SOURCE_FP=$(python -c "import json; print(json.load(open('$BTEH_VERSION_ROOT/manifests/bteh_image_manifest.json'))['manifest_fingerprint'])")
SPLIT_FP=$(python -c "import json; print(json.load(open('$BTEH_VERSION_ROOT/splits/bteh_splits.json'))['splits_fingerprint'])")

python -m pipeline.step_1_run_detection_to_crop --bteh \
    --image-manifest "$BTEH_VERSION_ROOT/manifests/bteh_image_manifest.parquet" \
    --crop-manifest "$BTEH_VERSION_ROOT/crops/crop_manifest.parquet" \
    --crops-dir "$BTEH_VERSION_ROOT/crops" \
    --source-fingerprint "$SOURCE_FP" \
    --split-fingerprint "$SPLIT_FP" \
    --disable-cudnn
```

Output: `$BTEH_ARTIFACT_ROOT/v1/crops/`

**Notes:**
- `--disable-cudnn` is required on this host due to cuDNN compatibility issues.
  Remove it only if you have verified cuDNN compatibility on a different host.
- Crop-generation manifests use local paths. The final production builder
  rewrites them relative to the artifact root for portability.
- Re-running is idempotent for already-completed crops (`detector_status` in
  `{accepted, none_detected, not_applicable}` are skipped on re-run).

---

## 7 · Partition Assignment

Assign each crop to `reference` or `query` based on the splits file.

```bash
python -m pipeline.bteh_partitions \
    --crop-manifest "$BTEH_VERSION_ROOT/crops/crop_manifest.parquet" \
    --splits "$BTEH_VERSION_ROOT/splits/bteh_splits.parquet" \
    --output-root "$BTEH_VERSION_ROOT/embeddings"
```

Output: `$BTEH_VERSION_ROOT/embeddings/{reference,query}/crop_manifest.parquet`.

---

## 8 · Frozen Embeddings

Compute embeddings for the **selected descriptors only**.  Do not run
MegaDescriptor (it received zero OOF weight and adds no signal for BTEH).

The selected descriptors for embedding are:
- `miewid` — whole-body MiewID embeddings
- `ear_miewid` — ear-crop MiewID embeddings (input to projection head)

`ear_miewid_projected` embeddings are produced in §10 (projection transform),
not directly by the embedder.

```bash
for PARTITION in reference query; do
  for DESCRIPTOR in miewid ear_miewid; do
    python -m pipeline.step_2_create_embeddings --bteh \
      --crop-manifest "$BTEH_VERSION_ROOT/embeddings/$PARTITION/crop_manifest.parquet" \
      --artifact-dir "$BTEH_VERSION_ROOT/embeddings/$PARTITION" \
      --partition "$PARTITION" \
      --descriptors "$DESCRIPTOR" \
      --source-fingerprint "$SOURCE_FP" \
      --split-fingerprint "$SPLIT_FP" \
      --model-fingerprint "$DESCRIPTOR:config-elephant-v1" \
      --disable-cudnn
  done
done
```

Output: `$BTEH_ARTIFACT_ROOT/v1/embeddings/{reference,query}/{miewid,ear_miewid}.{npy,parquet,index}`

**Do not re-embed after model selection is frozen.** Embeddings are tied to
model weights via `model_preprocess_fingerprint` in each mapping table.  Any
change to model weights or preprocessing invalidates all embeddings.

---

## 9 · Calibration and Eval (Reference)

### 9a · Train calibration

Fit per-channel calibration scalers (OOF, reference partition only).

```bash
# Isotonic (comparator)
python -m pipeline.step_4b_normalized_calibration \
    --artifact-root "$BTEH_VERSION_ROOT" \
    --splits-file "$BTEH_VERSION_ROOT/splits/bteh_splits.parquet" \
    --out-dir "$BTEH_VERSION_ROOT/calibration" \
    --channels megadescriptor miewid ear_megadescriptor ear_miewid \
    --calibration-method isotonic

# Platt non-projected (comparator)
python -m pipeline.step_4b_normalized_calibration \
    --artifact-root "$BTEH_VERSION_ROOT" \
    --splits-file "$BTEH_VERSION_ROOT/splits/bteh_splits.parquet" \
    --out-dir "$BTEH_VERSION_ROOT/calibration_platt" \
    --channels megadescriptor miewid ear_megadescriptor ear_miewid \
    --calibration-method platt
```

### 9b · Normalized eval

```bash
python -m pipeline.step_4c_normalized_eval \
    --artifact-root "$BTEH_VERSION_ROOT" \
    --splits-file "$BTEH_VERSION_ROOT/splits/bteh_splits.parquet" \
    --calibration-dir "$BTEH_VERSION_ROOT/calibration" \
    --out-dir "$BTEH_VERSION_ROOT/reports/calibrated_eval"

python -m pipeline.step_4c_normalized_eval \
    --artifact-root "$BTEH_VERSION_ROOT" \
    --splits-file "$BTEH_VERSION_ROOT/splits/bteh_splits.parquet" \
    --calibration-dir "$BTEH_VERSION_ROOT/calibration_platt" \
    --out-dir "$BTEH_VERSION_ROOT/reports/calibrated_eval_platt"
```

---

## 10 · Projection Training and Transform

Train the identity-adapter projection head for ear embeddings.

### 10a · Train projection head

```bash
python -m pipeline.train_miewid_projection \
    --artifact-root "$BTEH_VERSION_ROOT" \
    --out-dir "$BTEH_VERSION_ROOT/checkpoints/ear_miewid_identity_adapter" \
    --descriptor ear_miewid \
    --out-dim 2152 \
    --hidden-dim 0 \
    --loss triplet \
    --epochs 40 \
    --seed 42
```

Output: `$BTEH_ARTIFACT_ROOT/v1/checkpoints/ear_miewid_identity_adapter/`

Verify adoption gate:
```bash
python -c "
import json
with open('$BTEH_ARTIFACT_ROOT/v1/checkpoints/ear_miewid_identity_adapter/training_manifest.json') as f:
    m = json.load(f)
print('gate.adopted:', m['gate']['adopted'])
print('gate.reason:', m['gate']['reason'])
"
```

**If `gate.adopted=False`, do not proceed to production.** Re-examine
hyperparameters or training data.  The random 512-dim projection
(`ear_miewid_projection`) is an example of a rejected checkpoint; do not use it.

### 10b · Apply projection transform

```bash
python -m pipeline.transform_miewid_projection \
    --artifact-root "$BTEH_VERSION_ROOT" \
    --checkpoint "$BTEH_VERSION_ROOT/checkpoints/ear_miewid_identity_adapter/best_projection.pt" \
    --src-descriptor ear_miewid \
    --out-descriptor ear_miewid_projected \
    --partitions reference query
```

Output: `$BTEH_ARTIFACT_ROOT/v1/embeddings/{reference,query}/ear_miewid_projected.{npy,parquet,index}`
Transform manifest: `$BTEH_ARTIFACT_ROOT/v1/embeddings/transform_manifest.json`

---

## 11 · Selected Calibration

Fit the production Platt calibration on projected channels only.

```bash
python -m pipeline.step_4b_normalized_calibration \
    --artifact-root "$BTEH_VERSION_ROOT" \
    --splits-file "$BTEH_VERSION_ROOT/splits/bteh_splits.parquet" \
    --out-dir "$BTEH_VERSION_ROOT/calibration_projected" \
    --channels miewid ear_miewid_projected \
    --calibration-method platt
```

Run eval to confirm metrics match report:
```bash
python -m pipeline.step_4c_normalized_eval \
    --artifact-root "$BTEH_VERSION_ROOT" \
    --splits-file "$BTEH_VERSION_ROOT/splits/bteh_splits.parquet" \
    --calibration-dir "$BTEH_VERSION_ROOT/calibration_projected" \
    --out-dir "$BTEH_VERSION_ROOT/reports/calibrated_eval_projected" \
    --channels miewid ear_miewid_projected
```

Verify against expected values in `reports/bteh_model_selection_report.json`:
- `known_top1 ≈ 0.384`, `known_mAP ≈ 0.473`, `calibration_ece ≈ 0.120`

---

## 12 · Production Index Build

Merge reference and query partitions into a single production catalog.

```bash
python -m pipeline.build_production_index \
    --artifact-root "$BTEH_VERSION_ROOT" \
    --build-tag "$(date -u +%Y%m%dT%H%M%SZ)"
```

This will:
1. Validate all fingerprints (source, split, model) match across partitions.
2. Detect and fail on duplicate `crop_id`, identity mismatches, and
   cross-partition contamination.
3. Merge reference + query into a single contiguous matrix for each channel.
4. Rebuild FAISS index (`IndexFlatIP`, L2-normalised cosine).
5. Write to `$BTEH_ARTIFACT_ROOT/v1/production/<build-tag>/`.
6. Write `production_manifest.json` referencing calibration files by path
   (files are not duplicated).

**Verify output:**
```bash
python -c "
import json
import faiss, numpy as np
build_tag = '<build-tag>'
root = '$BTEH_ARTIFACT_ROOT/v1/production/' + build_tag
with open(root + '/production_manifest.json') as f:
    m = json.load(f)
print('auto_accept_enabled:', m['auto_accept_policy']['enabled'])  # must be False
for ch, stats in m['channel_stats'].items():
    print(ch, stats)
idx = faiss.read_index(root + '/miewid.index')
print('miewid ntotal:', idx.ntotal)
"
```

**Do not run on real artifacts during code review; execute only after sign-off.**

### 12a · Dry-run validation (safe to run any time)

```bash
python -m pipeline.build_production_index \
    --artifact-root "$BTEH_ARTIFACT_ROOT/v1" \
    --dry-run
```

---

## 13 · Adding a New Elephant Without Retraining

### Prerequisites

1. **Verified identity**: Individual must be confirmed by a BTEH expert before
   any images enter the system.  Do not add unresolved, anonymous, or
   "Not for AI" individuals.
2. **Multiple sessions and views**: Collect body crops (whole animal) and ear
   crops from at least 2 sessions.  Single-session individuals degrade
   onboarding performance.
3. **Human crop QA**: All crops must pass the §16 QA checklist before embedding.

### Onboarding steps

1. Add images beneath a verified named identity directory in the source tree.
2. Re-run §§3–7. Crop extraction is resumable, so completed images are skipped.
3. Re-run §8 for both partitions using the unchanged published MiewID weights.
4. Reapply the already-adopted projection checkpoint with §10b. Do **not**
   retrain it for each new elephant.
5. Re-run selected Platt calibration/evaluation (§11), because the source and
   split fingerprints changed. Investigate material score-distribution drift.
6. Build a new immutable production tag with §12. Prior builds remain available
   for rollback.

The current safe onboarding path performs a full manifest/index refresh rather
than an in-place append. This is deliberate: every production row is rebuilt
under one source/split fingerprint, preventing mixed-generation indexes.

### What NOT to do

- ❌ Do not retrain the MiewID backbone.
- ❌ Do not retrain the projection head unless catalog has grown substantially
  (see §14 for triggers).
- ❌ Do not set a score threshold for automatic identity acceptance.
- ❌ Do not add images from sessions where the individual's identity is disputed.

### Calibration monitoring

After each batch of ≥5 new individuals:
- Run the full calibration eval (§11) on the updated catalog.
- Compare ECE and open-set FAR/FRR against baseline values in the selection report.
- If ECE degrades by > 0.05 or FAR increases by > 10 percentage points,
  recalibration is required (§14).

---

## 14 · Refresh and Rebuild Triggers

| Trigger | Action Required |
|---------|-----------------|
| Catalog grows by ≥ 20% in image count | Rebuild production index (§12) |
| New individual batch ≥ 5 | Rerun calibration eval (§11); compare metrics |
| ECE degrades > 0.05 from baseline | Refit calibration (§11) and rebuild production index |
| Open-set FAR increases > 10 pp | Refit calibration; escalate to expert review before deploying |
| Any model weight change | Full rerun from §8 onward; prior production indexes invalid |
| Schema version bump in `config_bteh.py` | Full rerun from §3 onward |
| Source fingerprint changes | Verify manifest integrity; rerun from §3 |
| Split fingerprint changes | Rerun from §5 onward; all embeddings invalidated |
| Projection checkpoint updated | Rerun from §10b onward |

**Model retraining (backbone or projection) is not routine.**
Retraining is warranted only when retrieval performance has verifiably degraded
on a held-out evaluation set after catalog refresh.

---

## 15 · Artifact Fingerprint and Version Policy

Every artifact carries cryptographic fingerprints for traceability:

| Fingerprint | What It Covers | Where Used |
|-------------|---------------|------------|
| `source_fingerprint` | Content hash of the source image tree | Manifest, splits, embeddings, calibration |
| `split_fingerprint` | Hash of reference/query assignment table | Embeddings, calibration, eval, production manifest |
| `model_preprocess_fingerprint` | Model ID + preprocessing config | Each channel's mapping parquet |
| `checkpoint_fingerprint` | SHA-256 of `.pt` checkpoint file | Training manifest, production manifest |

### Rules

1. **Never modify artifacts in-place** after they are written.  All updates
   produce a new versioned artifact (new `build_tag` or schema bump).
2. **Production indexes are immutable** once deployed.  Append-only updates
   produce a new `build_tag`.
3. Any fingerprint mismatch between a newly embedded crop and the existing
   catalog must cause a hard failure and must not be silently dropped.
4. `ARTIFACT_SCHEMA_VERSION` in `config_bteh.py` is the master version gate.
   Bump it when the manifest schema changes incompatibly.
5. Do not symlink or copy large artifact files across versions unnecessarily.
   Calibration files are referenced by absolute path in the production manifest.

---

## 16 · Human Crop QA Checklist

Apply this checklist to every batch of crops before embedding:

**Body crops (`crop_kind=body`)**

- [ ] Animal is fully visible (not clipped at edges) or minimally clipped (< 20%)
- [ ] Correct individual in frame (verify against source image metadata)
- [ ] No occlusion of diagnostic markings (ear notches, tail, body markings)
- [ ] Image is in focus (not heavily blurred)
- [ ] No duplicate crops from the same detection event

**Ear crops (`crop_kind=ear`)**

- [ ] At least one ear is clearly visible and centred in the crop
- [ ] Ear notches (if present) are visible and unobstructed
- [ ] Crop is not clipped to a thin strip (minimum bounding box: 30×30 px)
- [ ] Correct side (left/right) if laterality matters for the protocol

**Identity QA**

- [ ] `individual_id` matches the source directory name
- [ ] Individual is not on the "Not for AI" list
- [ ] Individual's identity is resolved (not a UUID or hex-string directory)
- [ ] No images from disputed or anonymous sightings

**Failed crops**

- [ ] Flag `review_status=rejected` for any crop failing the above
- [ ] Do not proceed to embedding for rejected crops
- [ ] Investigate detector failures (`detector_status=failed`) — may indicate
  unexpected image format, orientation, or detector quality regression

---

## Appendix A · Selected Artifact Paths (relative to versioned root)

| Artifact | Relative Path |
|----------|--------------|
| Image manifest | `manifests/bteh_image_manifest.parquet` |
| Splits | `splits/bteh_splits.parquet` |
| Reference miewid embeddings | `embeddings/reference/miewid.npy` |
| Reference ear_miewid_projected embeddings | `embeddings/reference/ear_miewid_projected.npy` |
| Reference miewid FAISS index | `embeddings/reference/miewid.index` |
| Selected calibration dir | `calibration_projected/` |
| Selected calibration manifest | `calibration_projected/calibration_manifest.json` |
| Adopted projection checkpoint | `checkpoints/ear_miewid_identity_adapter/best_projection.pt` |
| Selected eval summary | `reports/calibrated_eval_projected/normalized_eval_summary.json` |
| Production output dir | `production/<build-tag>/` |
| Production manifest | `production/<build-tag>/production_manifest.json` |

## Appendix B · Key Metric Targets (from frozen evaluation)

| Metric | Target (reference value) | Source |
|--------|--------------------------|--------|
| Known top-1 | ≈ 0.384 | `calibrated_eval_projected` summary |
| Known mAP | ≈ 0.473 | `calibrated_eval_projected` summary |
| Calibration ECE | ≤ 0.125 | `calibrated_eval_projected` summary |
| Open-set FAR | documented only (≈27%) | disabled in production |
| Open-set FRR | documented only (≈48%) | disabled in production |

Any significant degradation from these values after a catalog refresh warrants
investigation before deploying the updated production index.
