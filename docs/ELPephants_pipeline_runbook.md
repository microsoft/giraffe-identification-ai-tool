# ELPephants Pipeline Runbook

ELPephants is a data source for the shared normalized elephant
re-identification pipeline. Its source adapter builds the canonical manifest;
all subsequent processing is source-agnostic and uses the same commands and
artifact contract as BTEH. Source images are immutable, and generated artifacts
live under a separate versioned root.

## Current status

Cleaned source snapshot:

- 2,074 images across 274 source identities.
- 2,042 directly included images.
- 16 retained representatives of same-identity exact-duplicate groups.
- 16 redundant exact-duplicate copies excluded.
- 0 rows requiring identity review.
- Original source labels: 1,570 train and 504 validation images.

The padded duplicate identities `00183` and `001566` were removed after review;
their unpadded counterparts `183` and `1566` are authoritative. Across BTEH and
ELPephants there are 331 namespaced identities (57 + 274).

Current generated artifacts:

- Manifest fingerprint:
  `2595a0dd10e3c4f0650de873bf84684f24308d183076cc0ab01f01aa748fdbe5`
- Split fingerprint:
  `dfc035ced1decb30cf0139277c8d9f40f6ccabe9727949ec1a7466c9dcce4fde`
- Splits: 1,344 gallery, 316 probe, 324 held-out gallery,
  74 held-out probe, and 16 excluded.

Manifest and split generation are complete. A deterministic 120-image crop
pilot spanning 93 identities has also completed:

- Body detector coverage and reviewed precision: 100%.
- At least one detected ear: 98.33% of images.
- Reviewed ear precision: 96.32% (131 accepted, 5 rejected).
- Two detected ears: 15% of images.
- Overall reviewed crop precision: 98.05% (251/256).

The five rejected ear crops were a trunk, a background/non-target object, a
front-body/legs crop, an ambiguous multi-elephant body patch, and one severely
clipped ambiguous ear. The pilot exceeds the 95% precision gate, so resumable
full-corpus crop extraction is approved. Embeddings and matching remain pending.

Pilot artifacts:

- `$ELPEPHANTS_VERSION_ROOT/pilot/bteh_pilot_manifest.parquet`
- `$ELPEPHANTS_VERSION_ROOT/pilot/pilot_visual_review.csv`
- `$ELPEPHANTS_VERSION_ROOT/pilot/report/bteh_pilot_crop_report.json`
- `$ELPEPHANTS_VERSION_ROOT/pilot/report/contact_sheets/`

Full extraction and frozen descriptor baselines:

- 2,058 eligible images processed; 2,055 body crops and 2,242 ear crops accepted.
- Reference partition: 1,668 images; query partition: 390 images.
- Raw temporal top-1: MegaDescriptor 1.58%, MiewID 5.38%,
  ear-MegaDescriptor 2.88%, ear-MiewID 14.06%.
- Raw held-out onboarding top-1: MegaDescriptor 1.35%, MiewID 2.70%,
  ear-MegaDescriptor 1.37%, ear-MiewID 9.59%.

Ear-MiewID is the strongest frozen baseline, but absolute retrieval quality is
not yet production-ready. Platt calibration is rejected because all four
hard-negative OOF fits have negative slopes, which would reverse similarity
rankings.

The split-safe ear-MiewID projection adapter improved inner validation mAP by
10.44 points and inner top-1 by 7.28 points, but failed the untouched external
gate: temporal top-1 fell from 14.06% to 12.46%, and held-out top-1 fell from
9.59% to 4.11%. The adapter is rejected and stored under
`experiments/rejected_ear_miewid_projection/`; it must not enter calibration or
production. Raw ear-MiewID remains the selected ELPephants baseline.

The archive's original train/validation split gives ear-MiewID 29.01% top-1,
showing a large same-era/session advantage over the leakage-safe temporal
protocol. Pre-specified raw fusion did not help: equal body+ear MiewID reached
10.13% temporal top-1, and equal four-channel fusion reached 6.96%.

**Current decision:** keep raw ear-MiewID as the research baseline; do not build
an ELPephants production matcher or auto-accept threshold. Calibration must be
revisited only after retrieval improves.

Reproduce the rejected experiment only under its isolated namespace:

```bash
python -m pipeline.train_miewid_projection \
  --artifact-root "$ELPEPHANTS_VERSION_ROOT" \
  --splits-file "$ELPEPHANTS_VERSION_ROOT/splits/elpephants_splits.parquet" \
  --out-dir "$ELPEPHANTS_VERSION_ROOT/experiments/rejected_ear_miewid_projection/checkpoint" \
  --descriptor ear_miewid \
  --out-dim 2152 \
  --hidden-dim 0 \
  --loss triplet \
  --epochs 40 \
  --seed 42 \
  --device cuda
```

## Run the crop-quality pilot

```bash
python -m pipeline.elephant_crop_pilot sample \
  --manifest "$ELPEPHANTS_VERSION_ROOT/manifests/elpephants_image_manifest.parquet" \
  --splits "$ELPEPHANTS_VERSION_ROOT/splits/elpephants_splits.parquet" \
  --output-dir "$ELPEPHANTS_VERSION_ROOT/pilot" \
  --source-fingerprint "$SOURCE_FP" \
  --split-fingerprint "$SPLIT_FP" \
  --n-pilot 120 \
  --n-review 0 \
  --seed 42
```

The deterministic sample is stratified across identities, evaluation splits,
years, session provenance, image size/aspect, and viewpoints.

## Configure roots

Set these values in the local environment:

```dotenv
ELPEPHANTS_SOURCE_ROOT=/absolute/path/to/ELPephants
ELPEPHANTS_ARTIFACT_ROOT=/absolute/path/to/ELPephants_reid_artifacts
```

```bash
export ELPEPHANTS_VERSION_ROOT="$ELPEPHANTS_ARTIFACT_ROOT/v1"
```

The source tree must contain `images/`, `train.txt`, `val.txt`,
`class_mapping.txt`, and `LICENSE.txt`.

## Build the canonical manifest

```bash
python -m pipeline.elpephants_manifest \
  --source-root "$ELPEPHANTS_SOURCE_ROOT" \
  --artifact-root "$ELPEPHANTS_ARTIFACT_ROOT"
```

The manifest preserves source class IDs, class indexes, and original
train/validation labels. Exact duplicates within an identity are deduplicated;
exact duplicates assigned to different identities require review.

Output:
`$ELPEPHANTS_VERSION_ROOT/manifests/elpephants_image_manifest.parquet`

## Generate evaluation splits

```bash
python -m pipeline.elephant_splits \
  --manifest "$ELPEPHANTS_VERSION_ROOT/manifests/elpephants_image_manifest.parquet" \
  --output "$ELPEPHANTS_VERSION_ROOT/splits/elpephants_splits.parquet"
```

These are leakage-aware temporal and held-out-identity splits. The source
train/validation labels remain provenance and are not used as re-identification
evaluation partitions.

## Generate body and ear crops

```bash
SOURCE_FP=$(python -c \
  "import json; print(json.load(open('$ELPEPHANTS_VERSION_ROOT/manifests/elpephants_image_manifest.json'))['manifest_fingerprint'])")
SPLIT_FP=$(python -c \
  "import json; print(json.load(open('$ELPEPHANTS_VERSION_ROOT/splits/elpephants_splits.json'))['splits_fingerprint'])")

python -m pipeline.step_1_run_detection_to_crop --normalized \
  --image-manifest "$ELPEPHANTS_VERSION_ROOT/manifests/elpephants_image_manifest.parquet" \
  --crop-manifest "$ELPEPHANTS_VERSION_ROOT/crops/crop_manifest.parquet" \
  --crops-dir "$ELPEPHANTS_VERSION_ROOT/crops" \
  --source-root "$ELPEPHANTS_SOURCE_ROOT" \
  --source-fingerprint "$SOURCE_FP" \
  --split-fingerprint "$SPLIT_FP" \
  --disable-cudnn
```

## Partition crops

```bash
python -m pipeline.elephant_partitions \
  --crop-manifest "$ELPEPHANTS_VERSION_ROOT/crops/crop_manifest.parquet" \
  --splits "$ELPEPHANTS_VERSION_ROOT/splits/elpephants_splits.parquet" \
  --output-root "$ELPEPHANTS_VERSION_ROOT/embeddings"
```

## Build embeddings and indexes

Run each descriptor and partition separately so every artifact records the
specific model/preprocessing fingerprint:

```bash
for PARTITION in reference query; do
  for DESCRIPTOR in megadescriptor miewid ear_megadescriptor ear_miewid; do
    python -m pipeline.step_2_create_embeddings --normalized \
      --crop-manifest "$ELPEPHANTS_VERSION_ROOT/embeddings/$PARTITION/crop_manifest.parquet" \
      --artifact-dir "$ELPEPHANTS_VERSION_ROOT/embeddings/$PARTITION" \
      --partition "$PARTITION" \
      --descriptors "$DESCRIPTOR" \
      --source-fingerprint "$SOURCE_FP" \
      --split-fingerprint "$SPLIT_FP" \
      --model-fingerprint "$DESCRIPTOR:config-elephant-v1"
  done
done
```

## Run normalized matching

```bash
python -m pipeline.step_3_run_initial_matching --normalized \
  --query-artifact-dir "$ELPEPHANTS_VERSION_ROOT/embeddings/query" \
  --reference-artifact-dir "$ELPEPHANTS_VERSION_ROOT/embeddings/reference" \
  --query-crop-manifest "$ELPEPHANTS_VERSION_ROOT/embeddings/query/crop_manifest.parquet" \
  --output "$ELPEPHANTS_VERSION_ROOT/reports/initial_matches.parquet" \
  --source-fingerprint "$SOURCE_FP" \
  --split-fingerprint "$SPLIT_FP"
```
