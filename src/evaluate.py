"""Leave-one-signer-out evaluation, and a sweep over feature variants.

Every signer takes a turn as the test set while the model trains on the other
nine. Nothing about the test person's anatomy is in training, so this is the
number that predicts a stranger walking up to the demo - unlike the 99.9% test
accuracy, which mostly measures how similar consecutive video frames are.

    python src/evaluate.py --data data/processed/asl_hg.npz
    python src/evaluate.py --data data/processed/asl_hg.npz \
        --modes raw rot dist rot+dist --quick

--quick subsamples and shortens training so a five-variant sweep takes minutes;
re-run the winner without it before believing the exact number.
"""

import argparse
import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

import features as F


def build_model(quick):
    # StandardScaler matters more than it looks: pairwise distances and raw
    # coordinates live on different scales, and an MLP fed both without scaling
    # will effectively ignore the smaller-magnitude block.
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=60 if quick else 400,
            early_stopping=not quick,
            n_iter_no_change=15,
            random_state=42,
        ),
    )


def loso(X, y, groups, quick):
    """Returns per-signer accuracy, in signer order."""
    scores = {}
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        held = groups[test_idx][0]
        model = build_model(quick)
        model.fit(X[train_idx], y[train_idx])
        scores[held] = accuracy_score(y[test_idx], model.predict(X[test_idx]))
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--modes", nargs="+", default=["raw"])
    parser.add_argument("--augment", type=int, default=1,
                        help="Synthesize N-1 rotated/jittered copies of the "
                             "training signers. Test signer is never augmented.")
    parser.add_argument("--quick", action="store_true",
                        help="Subsample to 300/class and cap training epochs.")
    parser.add_argument("--min-samples", type=int, default=50)
    args = parser.parse_args()

    parts_X, parts_y, parts_g = [], [], []
    for path in args.data:
        blob = np.load(path, allow_pickle=True)
        parts_X.append(blob["X"])
        parts_y.append(blob["y"])
        parts_g.append(blob["groups"] if "groups" in blob.files
                       else np.full(len(blob["y"]), path))

    X = np.concatenate(parts_X)
    y_raw = np.concatenate(parts_y)
    groups = np.concatenate(parts_g)

    labels, counts = np.unique(y_raw, return_counts=True)
    too_few = labels[counts < args.min_samples]
    if len(too_few):
        keep = ~np.isin(y_raw, too_few)
        X, y_raw, groups = X[keep], y_raw[keep], groups[keep]
        print(f"dropped {list(too_few)} (< {args.min_samples} samples)")

    if args.quick:
        rng = np.random.RandomState(0)
        keep = np.concatenate([
            rng.permutation(np.flatnonzero(y_raw == c))[:300]
            for c in np.unique(y_raw)
        ])
        X, y_raw, groups = X[keep], y_raw[keep], groups[keep]

    y = LabelEncoder().fit_transform(y_raw)
    signers = sorted(set(groups))
    print(f"{len(X)} samples, {len(set(y))} classes, {len(signers)} signers"
          f"{' [quick]' if args.quick else ''}\n")

    if len(signers) < 2:
        raise SystemExit(
            "LOSO needs at least 2 signers. asl_hg.npz has 10; a file without "
            "signer ids collapses to one group and cannot be split this way."
        )

    summary = []
    for mode in args.modes:
        started = time.time()
        Xm = F.transform(X, mode)

        if args.augment > 1:
            # Augment inside each fold, so the held-out signer stays untouched.
            scores = {}
            for train_idx, test_idx in LeaveOneGroupOut().split(Xm, y, groups):
                Xa, ya, _ = F.augment(X[train_idx], y[train_idx],
                                      groups[train_idx], args.augment)
                model = build_model(args.quick)
                model.fit(F.transform(Xa, mode), ya)
                scores[groups[test_idx][0]] = accuracy_score(
                    y[test_idx], model.predict(Xm[test_idx]))
        else:
            scores = loso(Xm, y, groups, args.quick)

        values = np.array([scores[s] for s in signers])
        summary.append((mode, Xm.shape[1], values, time.time() - started))

        print(f"{mode:>9} ({Xm.shape[1]:>3}f)  " +
              "  ".join(f"{s}:{scores[s]:.3f}" for s in signers))

    print("\n" + "=" * 62)
    print(f"{'mode':>10} {'feats':>6} {'mean':>8} {'std':>7} {'worst':>8} {'mins':>6}")
    baseline = summary[0][2].mean()
    for mode, width, values, elapsed in summary:
        delta = values.mean() - baseline
        marker = "" if mode == summary[0][0] else f"  ({delta:+.1%})"
        print(f"{mode:>10} {width:>6} {values.mean():>7.1%} "
              f"{values.std():>6.1%} {values.min():>7.1%} {elapsed / 60:>6.1f}{marker}")

    best = max(summary, key=lambda r: r[2].mean())
    print(f"\nbest: {best[0]} at {best[2].mean():.1%} mean across signers")
    print("The worst-signer column matters as much as the mean - it is the "
          "person your demo fails in front of.")


if __name__ == "__main__":
    main()
