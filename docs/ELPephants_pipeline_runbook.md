# ELPephants Pipeline Runbook

ELPephants is a data source for the shared normalized elephant
re-identification pipeline. Its source adapter builds the canonical manifest;
all subsequent processing is source-agnostic and uses the same commands and
artifact contract as BTEH. Source images are immutable, and generated artifacts
live under a separate versioned root.

## Current status

Cleaned source snapshot:

- 2,074 images across 274 source identities.
- 2,022 directly included images.
- 26 retained representatives of same-identity exact-duplicate groups.
- 26 redundant exact-duplicate copies excluded: 16 byte-identical and
  10 metadata/encoding variants with identical decoded RGB pixels.
- 0 rows requiring identity review.
- Original source labels: 1,570 train and 504 validation images.

The padded duplicate identities `00183` and `001566` were removed after review;
their unpadded counterparts `183` and `1566` are authoritative. Across BTEH and
ELPephants there are 331 namespaced identities (57 + 274).

Current generated artifacts:

- Manifest fingerprint:
  `057b26b56bf75aa016dec000cded0cb8113941162b15bced995d6d1839e1a702`
- Split fingerprint:
  `13e1c435deda1524986573fbc870096065952fbdccd3593d260c295656238f3c`
- Splits: 1,336 gallery, 317 probe, 320 held-out gallery,
  75 held-out probe, and 26 excluded.

The July 24 forensic audit corrected three ingestion/preprocessing defects:

- Undated files no longer create lexicographically latest `"unknown"` sessions;
  16 such files had incorrectly become probe images.
- Four known filename date typos now parse correctly.
- Deduplication now uses exact decoded-pixel hashes after byte hashes.
- Normalized body embeddings again flip right views to the canonical left
  orientation, and the preprocessing is included in model fingerprints.

Pre-correction manifests, splits, embeddings, reports, pilot artifacts, and
rejected experiments are retained under
`experiments/pre_ingestion_fix_20260724/`.

Manifest and split generation are complete. A deterministic 120-image crop
pilot spanning 93 identities completed before the session correction; its crop
quality findings remain valid because image pixels and detector code did not
change:

- Body detector coverage and reviewed precision: 100%.
- At least one detected ear: 98.33% of images.
- Reviewed ear precision: 96.32% (131 accepted, 5 rejected).
- Two detected ears: 15% of images.
- Overall reviewed crop precision: 98.05% (251/256).

The five rejected ear crops were a trunk, a background/non-target object, a
front-body/legs crop, an ambiguous multi-elephant body patch, and one severely
clipped ambiguous ear. The pilot exceeds the 95% precision gate, so resumable
full-corpus crop extraction was approved and has completed.

Pilot artifacts:

- `$ELPEPHANTS_VERSION_ROOT/experiments/pre_ingestion_fix_20260724/pilot/`

Full extraction and frozen descriptor baselines:

- 2,048 eligible images processed; 2,045 body crops and 2,232 ear crops accepted.
- Reference partition: 1,656 images; query partition: 392 images.
- Raw temporal top-1: MegaDescriptor 1.89%, MiewID 5.36%,
  ear-MegaDescriptor 1.59%, ear-MiewID 13.33%.
- Raw held-out onboarding top-1: MegaDescriptor 6.67%, MiewID 13.33%,
  ear-MegaDescriptor 5.41%, ear-MiewID 17.57%.
- Raw temporal top-5 for selected ear-MiewID: 23.17%.
- Raw held-out top-5 for selected ear-MiewID: 35.14%.

Ear-MiewID is the strongest frozen baseline, but absolute retrieval quality is
not yet production-ready. Platt calibration is rejected because all four
hard-negative OOF fits have negative slopes, which would reverse similarity
rankings.

The pre-correction split-safe ear-MiewID projection experiment passed an inner
validation gate but regressed on its untouched query. It remains archived under
`experiments/pre_ingestion_fix_20260724/rejected_ear_miewid_projection/` and is
not valid for the corrected split. Raw ear-MiewID remains the selected
ELPephants baseline.

The archive's original train/validation split gives ear-MiewID 28.22% top-1,
but 307/495 validation images share a session with training. MegaDescriptor
reaches 15.96% on that contaminated split, close to the published 13.66%
global-only result, while falling to 1.89% under the leakage-safe temporal
protocol. This sanity check supports the embedding implementation and shows the
large same-session advantage in the canonical benchmark.

**Current decision:** keep raw ear-MiewID as the research baseline; do not build
an ELPephants production matcher or auto-accept threshold. Calibration must be
revisited only after retrieval improves.

One unresolved source-label warning remains: class IDs `3817` and `3819` are
both named `Sheena III`. Their source images are not duplicate pixels, and each
protocol slice still contains a same-ID reference, so this cannot explain the
overall low accuracy. Do not merge them without authoritative catalog review.

## Forensic accuracy audit

The July 24 audit independently reconstructed every stored rank from the raw
embedding matrices. It found no row-alignment, cosine-direction, tie-order,
FAISS, identity-aggregation, or query/reference mapping defect. All vectors are
finite and L2-normalized, and every query identity is represented in the
protocol-appropriate reference.

The low accuracy is primarily genuine:

- ELPephants spans roughly 14 years and has weak visual identity cues.
- For raw ear-MiewID, positive and negative similarity distributions overlap
  heavily (Cohen's d approximately 0.37).
- The original archive split is image-random: 307/495 usable validation images
  share a session with training, inflating top-1 to 28.22%.
- Published global-only MegaDescriptor performance on the random benchmark is
  also low (13.66% top-1), close to this pipeline's corrected 15.96%.
- Published methods reach approximately 49–54% top-1 only after adding local
  features and geometric verification, on the easier random split.

Authoritative dataset paper:
<https://openaccess.thecvf.com/content_ICCVW_2019/html/CVWC/Korschens_ELPephants_A_Fine-Grained_Dataset_for_Elephant_Re-Identification_ICCVW_2019_paper.html>

Therefore the corrected 13.33% temporal top-1 is not evidence of a remaining
gross coding failure. It is a global-descriptor baseline on a substantially
harder, session-disjoint protocol. Local evidence and geometric verification
are the most justified next research direction.

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
