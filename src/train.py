"""Train a classifier on the extracted hand landmarks.

    python src/train.py --data data/processed/landmarks.npz --out models/
"""

import argparse
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

import features as F


def print_confusion(cm, classes, top=12):
    """Print only the confusions that actually matter, not a 29x29 wall of zeros."""
    pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], cm[i].sum(), classes[i], classes[j]))
    pairs.sort(reverse=True)

    if not pairs:
        print("  (no misclassifications)")
        return

    print(f"  {'true':>8} -> {'predicted':<10} {'count':>6}  {'rate':>6}")
    for count, total, true_label, pred_label in pairs[:top]:
        print(f"  {true_label:>8} -> {pred_label:<10} {count:>6}  {count / total:>5.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, nargs="+",
                        help="One or more .npz files; they are concatenated.")
    parser.add_argument("--oversample", type=int, default=1,
                        help="Repeat the LAST --data file N times. Your own captured "
                             "data is tiny next to 66k Kaggle rows, so without this it "
                             "gets drowned out.")
    parser.add_argument("--cap", type=int,
                        help="Randomly subsample every file except the last to this "
                             "many rows. Preferred over --oversample: no duplicates.")
    parser.add_argument("--out", default="models")
    parser.add_argument("--model", choices=["mlp", "forest"], default="mlp")
    parser.add_argument("--name", default="classifier",
                        help="Output filename stem, e.g. 'isl_classifier'.")
    parser.add_argument("--min-samples", type=int, default=50,
                        help="Drop classes with fewer surviving samples than this.")
    parser.add_argument("--holdout-signer",
                        help="Signer id (e.g. P7) to exclude from training and test on "
                             "instead of a random split. This is the honest estimate of "
                             "accuracy on a person the model has never seen.")
    parser.add_argument("--features", default="raw",
                        help="Feature set. 'raw+dist' for ASL (89.1%% LOSO). "
                             "For ISL try 'raw+dist' or 'raw+cross'.")
    parser.add_argument("--augment", type=int, default=1,
                        help="Synthesize N-1 rotated/jittered copies of the "
                             "training rows. x3 added +2.3 points cross-signer.")
    args = parser.parse_args()

    parts_X, parts_y, parts_origin, parts_group = [], [], [], []
    for i, path in enumerate(args.data):
        blob = np.load(path, allow_pickle=True)
        Xi, yi = blob["X"], blob["y"]

        # Balance by throwing away surplus rows rather than duplicating scarce
        # ones. Duplicates add weight but no diversity, and they leak into
        # sklearn's internal early-stopping split, which halts training early.
        if args.cap and i < len(args.data) - 1 and len(yi) > args.cap:
            pick = np.random.RandomState(42).choice(len(yi), args.cap, replace=False)
            Xi, yi = Xi[pick], yi[pick]
            print(f"  (capped {path} to {args.cap})")

        parts_X.append(Xi)
        parts_y.append(yi)
        parts_origin.append(np.full(len(yi), i))
        # Files without signer ids (Kaggle, your own captures) are each treated
        # as one signer, named after the file.
        gi = blob["groups"] if "groups" in blob.files else np.full(
            len(yi), os.path.splitext(os.path.basename(path))[0])
        parts_group.append(gi)
        print(f"  {path}: {Xi.shape[0]} samples")

    widths = {p.shape[1] for p in parts_X}
    if len(widths) > 1:
        raise SystemExit(f"Feature widths differ across files: {widths}. "
                         "Did you mix ASL (63) and ISL (127) data?")

    X = np.concatenate(parts_X)
    y_raw = np.concatenate(parts_y)
    origin = np.concatenate(parts_origin)
    signer = np.concatenate(parts_group)

    # 'nothing' is empty frames, so the detector finds no hand and the class
    # survives only as a handful of false positives. Training on those teaches
    # the model noise. The live app treats "no hand" as a UI state, not a class.
    labels, counts = np.unique(y_raw, return_counts=True)
    too_few = labels[counts < args.min_samples]
    if len(too_few):
        print(f"Dropping {list(too_few)} (< {args.min_samples} samples)")
        keep = ~np.isin(y_raw, too_few)
        X, y_raw, origin, signer = X[keep], y_raw[keep], origin[keep], signer[keep]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    classes = encoder.classes_

    print(f"{X.shape[0]} samples, {X.shape[1]} features, {len(classes)} classes")

    if args.holdout_signer:
        held = signer == args.holdout_signer
        if not held.any():
            raise SystemExit(f"No samples from signer {args.holdout_signer}. "
                             f"Available: {sorted(set(signer))}")
        # No shared signer between the splits, so nothing about this person's
        # anatomy can leak into training. This is the number that predicts
        # performance on a stranger.
        X_train, y_train, origin_train = X[~held], y[~held], origin[~held]
        X_test, y_test = X[held], y[held]
        print(f"Holding out signer {args.holdout_signer}: "
              f"{held.sum()} test / {(~held).sum()} train samples")
    else:
        # Stratify so every letter appears in both splits.
        X_train, X_test, y_train, y_test, origin_train, _ = train_test_split(
            X, y, origin, test_size=0.2, random_state=42, stratify=y
        )

    # Oversample AFTER splitting. Duplicating first would put copies of the same
    # frame on both sides of the split and inflate test accuracy - the exact
    # leakage this project already ran into once with the Kaggle data.
    if args.oversample > 1:
        last = len(args.data) - 1
        extra = origin_train == last
        if extra.any():
            X_train = np.concatenate([X_train] + [X_train[extra]] * (args.oversample - 1))
            y_train = np.concatenate([y_train] + [y_train[extra]] * (args.oversample - 1))
            share = extra.sum() * args.oversample / len(y_train)
            print(f"Oversampled {args.data[last]} x{args.oversample} "
                  f"-> {share:.0%} of the training set ({len(y_train)} rows)")

    # Augment before the feature transform, since rotation only makes sense on
    # raw coordinates. Training rows only - the test split stays untouched, or
    # the score stops meaning anything.
    if args.augment > 1:
        dummy = np.zeros(len(y_train))
        X_train, y_train, _ = F.augment(X_train, y_train, dummy, args.augment)
        print(f"Augmented training set x{args.augment} -> {len(y_train)} rows")

    if args.features != "raw":
        X_train = F.transform(X_train, args.features)
        X_test = F.transform(X_test, args.features)
        print(f"Features: {args.features} -> {X_train.shape[1]} dimensions")

    if args.model == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=400,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=42,
            verbose=True,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=42
        )

    # Scaling matters once coordinates and distances are mixed: they sit on
    # different magnitude scales and the MLP would underweight one block.
    # Measured to *hurt* plain coordinates (86.0%% -> 82.6%% LOSO), so it is
    # applied only where it earns its place.
    if args.features != "raw":
        model = make_pipeline(StandardScaler(), model)

    print(f"\nTraining {args.model}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest accuracy: {accuracy:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=classes, digits=3))

    print("\nTop confusions (M/N/S and A/E/T are the usual offenders):")
    print_confusion(confusion_matrix(y_test, y_pred), classes)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.name}.joblib")
    # The feature mode travels with the model. Inference reads it from here
    # rather than from a flag, so a model can never be fed the wrong features.
    joblib.dump({"model": model, "classes": classes,
                 "features": args.features}, path)

    size_kb = os.path.getsize(path) / 1024
    print(f"\nSaved -> {path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
