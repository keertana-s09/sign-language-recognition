# Measured results

Every number here comes from a script in `src/`, and the raw output is in this
folder. Reproduce with the commands shown.

## The metric that matters

Test accuracy on this project is **worthless and actively misleading**. Both
source datasets are consecutive frames from video recordings, so a random split
puts near-duplicate frames on both sides. Every model here scores 98-99.9% on a
random split, and that number ranked the models *backwards* against real use.

The honest metric is **leave-one-signer-out (LOSO)**: ASL-HG labels which of its
10 participants produced each image, so each signer takes a turn as the test set
while the model trains on the other nine. Nothing about the test person's
anatomy is in training. That predicts a stranger walking up to the demo.

```
python src/evaluate.py --data data/processed/asl_hg.npz --modes raw dist raw+dist
```

## Feature engineering (10-fold LOSO, ASL-HG, 36 classes, 35,982 samples)

| features | dims | mean | std | worst signer |
|---|---|---|---|---|
| `raw` — coordinates only | 63 | 82.6% | 7.6% | 71.4% |
| `dist` — pairwise distances | 210 | 87.5% | 5.2% | 76.4% |
| `raw+dist` | 273 | **89.1%** | **3.7%** | **83.0%** |
| `raw+dist` + augment x3 | 273 | **91.4%** | 4.8% | **83.5%** |

`raw` scores 86.0% without the StandardScaler that the mixed feature sets
require, so the fair comparison against the previously shipping model is
**86.0% -> 91.4%**.

The worst-signer column improved more than the mean (71.4% -> 83.5%): the gain
lands where the model was actually failing, not on the signers it already knew.

### Why distances help

Coordinates encode *where landmarks sit*; pairwise distances encode *handshape*
- "thumb tip touching index tip" - which is what defines a letter. Signers
differ in finger-length ratios, and in distance space that appears as a smooth
rescaling rather than as displaced points. Cross-signer std nearly halved.

### Why rotation-invariance was rejected

Re-expressing each hand in a self-anchored frame (`rot`, `src/features.py`)
removes 3D rotation entirely. Alone it helps a lot: 82.6% -> 85.4%, worst signer
71.4% -> 77.6%, confirming orientation sensitivity was real.

But combined with distances it *loses*: `rot+dist` 86.8% vs `raw+dist` 88.8%
(quick sweep). ASL uses palm orientation to distinguish letters, so discarding it
costs more class information than the nuisance variance it removes. Keeping
orientation and adding shape beats normalizing orientation away.

## Cross-dataset history

| training data | random-split test | live (own hands) | unseen signer |
|---|---|---|---|
| Kaggle, 1 signer | 98.9% | poor (12.6x dist ratio) | 46.8% |
| + own captures, x20 duplicated | 98.9% | 88% (3.8x) | - |
| Kaggle capped + own (`balanced`) | 98.0% | 99% (2.2x) | - |
| ASL-HG 10 signers, `raw` | - | - | 86.0% |
| ASL-HG + own, `raw+dist` + augment (`final_v2`) | 99.9% | not yet measured | **91.4%** |

Oversampling the scarce source by duplication made things *worse*: duplicates add
weight but no diversity, and they leak into sklearn's internal early-stopping
split, halting training at 46 epochs instead of 81. Subsampling the majority
source (`--cap`) is the correct fix.

## Files

- `loso_quick_sweep.txt` — 5 feature variants, subsampled
- `loso_full.txt` — 3 variants, full data
- `loso_augment.txt` — winner + augmentation
- `train_final_v2.txt` — shipping model fit
