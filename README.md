# ASL Fingerspelling Recognition

Real-time American Sign Language fingerspelling recognition from a webcam.
MediaPipe extracts hand landmarks; a small sklearn MLP classifies the letter;
a debounced UI builds words on screen.

![Python](https://img.shields.io/badge/python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/sklearn-1.9-orange)
![MediaPipe](https://img.shields.io/badge/mediapipe-1.0.1-green)

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download the hand landmarker model (~7.5 MB)
wget -P models/ https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

python src/live_demo.py          # ASL mode
python src/live_demo.py --lang isl   # ISL mode (press 'l' to toggle)
```

**Keys:** `q` quit, `c` clear, `backspace` delete, `l` switch language, `h` swap handedness

## Architecture

```
webcam -> MediaPipe HandLandmarker (frozen, 7.5 MB)
       -> 21 keypoints (x, y, z) per hand
       -> normalize (wrist-center, scale to unit hand size)
       -> feature transform (coordinates + 210 pairwise distances = 273 floats)
       -> sklearn MLP (256, 128) with StandardScaler -> letter prediction
       -> probability averaging over 5 frames -> debounce (hold 12 frames to commit)
```

The model is 1.3 MB and trains in ~2 minutes. MediaPipe does the hard
perception work; the MLP head learns which hand shapes correspond to which
letters.

## The interesting part: evaluation

**Test accuracy on this project is misleading and ranks models backwards.**

Both source datasets (Kaggle ASL Alphabet, ASL-HG) consist of consecutive video
frames, so a random train/test split puts near-duplicates on both sides. Every
model scores 98-99% on a random split, but that number predicted the wrong
winner every time.

The honest metric is **leave-one-signer-out (LOSO)**: ASL-HG labels which of
its 10 participants produced each image, so each signer takes a turn as the
held-out test set. Nothing about the test person's anatomy appears in training.

### Feature engineering results (10-fold LOSO, 36 classes, 35,982 samples)

| Features | Dims | Mean | Std | Worst signer |
|---|---|---|---|---|
| Raw coordinates | 63 | 86.0% | — | 71.4% |
| Pairwise distances | 210 | 87.5% | 5.2% | 76.4% |
| **Raw + distances** | **273** | **89.1%** | **3.7%** | **83.0%** |
| Raw + distances + augmentation x3 | 273 | **91.4%** | 4.8% | **83.5%** |

Distances encode *handshape* ("thumb tip touching index tip") rather than
landmark position, so different signers' finger-length ratios become a smooth
rescaling instead of displaced points.

**Rotation invariance was tested and rejected.** Re-expressing each hand in a
canonical frame helps alone (+2.8 points), but *loses* when combined with
distances — ASL uses palm orientation to distinguish letters, so discarding it
costs more class information than nuisance variance.

### Cross-dataset progression

| Training data | Test acc | Live (own hands) | Unseen signer |
|---|---|---|---|
| Kaggle, 1 signer | 98.9% | poor (12.6x dist ratio) | 46.8% |
| + own captures, x20 duplicated | 98.9% | 88% | — |
| Kaggle capped + own | 98.0% | 99% | — |
| ASL-HG 10 signers, raw | — | — | 86.0% |
| **ASL-HG + own, raw+dist + augment** | **99.9%** | — | **91.4%** |

## Project structure

```
src/
  extract_landmarks.py   # MediaPipe -> normalized landmark vectors (.npz)
  features.py            # Feature variants: distances, rotation, augmentation
  train.py               # Train MLP with --features, --augment, --cap, --holdout-signer
  evaluate.py            # Leave-one-signer-out across feature variants
  tune.py                # Hyperparameter search scored by GroupKFold on signer
  diagnose.py            # Compare live hand against training distribution
  capture_data.py        # Record your own landmarks with guided prompts
  live_demo.py           # Webcam demo with ASL/ISL toggle

models/
  balanced.joblib        # ASL, raw features, 28 classes (previous best)
  final_v2.joblib        # ASL, raw+dist features, 38 classes (current best)
  isl_classifier.joblib  # ISL, raw features, 35 classes (single-signer, unvalidated)

data/processed/          # Landmark vectors (committed, small)
  asl_hg.npz             # ASL-HG: 10 signers, 36 classes, 35,982 samples
  own_asl.npz            # Own captures: 28 classes, 5,600 samples
  isl_landmarks.npz      # Kaggle ISL: 35 classes, 42,229 samples
  landmarks.npz          # Kaggle ASL Alphabet (superseded by ASL-HG)

tests/
  test_features.py       # 45 invariance tests (translation, scale, rotation, two-hand)

results/                 # LOSO outputs and sweep results
```

## Data sources

- [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) — 87k images, 1 signer (used for initial dev, superseded)
- [ASL-HG](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet) — 36k images, 10 signers, filename-encoded participant IDs (primary training data)
- [Indian Sign Language](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl) — 42k images, 1 signer, mixed one/two-handed

## ISL (Indian Sign Language)

ISL support is built but partially validated. The two-hand encoding packs both
hands into a 127-float vector in a shared coordinate frame anchored on the
primary hand. Cross-hand pairwise distances (441 floats) are available as
features for two-handed letters.

To improve ISL, record yourself and friends:

```bash
python src/capture_data.py --lang isl --signer yourname
python src/evaluate.py --data data/processed/isl_landmarks.npz data/processed/own_isl.npz \
    --modes raw dist raw+dist cross raw+cross --quick
```

## Known limitations

- **Static signs only.** J and Z require motion in ASL; the model can only match
  a single pose for them.
- **91.4% on unseen signers** is decent, not striking — and with 10 signers
  the confidence interval on that mean is roughly +/-1.5%.
- **ISL model is single-signer** and will generalize poorly until multi-person
  data is captured.
- **Left-handed signers** are handled for ASL (mirrored onto right) but not
  for ISL (two-handed signs have asymmetric roles).
