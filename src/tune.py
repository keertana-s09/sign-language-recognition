"""Hyperparameter search scored the only way that counts: across signers.

An ordinary GridSearchCV would split randomly, and on this data a random split
leaks near-duplicate video frames between train and validation - so it would
happily pick whichever configuration memorizes hardest. GroupKFold on the signer
id keeps every one of a person's frames on the same side of the split, so the
search optimizes for generalizing to a new person instead.

    python src/tune.py --data data/processed/asl_hg.npz
    python src/tune.py --data data/processed/asl_hg.npz --full   # slow, confirms
"""

import argparse
import json
import time

import numpy as np
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

import features as F

# alpha is the L2 penalty and is the knob that most directly trades memorizing
# these ten people against generalizing to an eleventh, so it gets the widest
# sweep. Two hidden layers vs one is the other axis worth spending fits on.
SEARCH_SPACE = {
    "mlp__hidden_layer_sizes": [(128,), (256,), (256, 128), (512, 256), (384, 192, 96)],
    "mlp__alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    "mlp__learning_rate_init": [3e-4, 1e-3, 3e-3],
    "mlp__batch_size": [128, 256, 512],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--features", default="raw+dist",
                        help="Feature mode. Auto-detected width picks valid modes.")
    parser.add_argument("--n-iter", type=int, default=24,
                        help="Configurations to sample from the space.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--per-class", type=int, default=300,
                        help="Subsample for speed. --full disables it.")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--out", default="results/tuning.json")
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

    if not args.full:
        rng = np.random.RandomState(0)
        keep = np.concatenate([
            rng.permutation(np.flatnonzero(y_raw == c))[:args.per_class]
            for c in np.unique(y_raw)
        ])
        X, y_raw, groups = X[keep], y_raw[keep], groups[keep]

    X = F.transform(X, args.features)
    y = LabelEncoder().fit_transform(y_raw)

    n_signers = len(set(groups))
    folds = min(args.folds, n_signers)
    print(f"{len(X)} samples, {X.shape[1]} features, {len(set(y))} classes, "
          f"{n_signers} signers, {folds}-fold GroupKFold")
    print(f"sampling {args.n_iter} configurations\n")

    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(max_iter=400, early_stopping=True,
                              n_iter_no_change=12, random_state=42)),
    ])

    search = RandomizedSearchCV(
        pipeline,
        SEARCH_SPACE,
        n_iter=args.n_iter,
        cv=GroupKFold(n_splits=folds),
        scoring="accuracy",
        n_jobs=-1,
        random_state=42,
        verbose=1,
        return_train_score=False,
    )

    started = time.time()
    search.fit(X, y, groups=groups)
    elapsed = (time.time() - started) / 60

    results = search.cv_results_
    order = np.argsort(-results["mean_test_score"])

    print(f"\n{'rank':>4}  {'cross-signer acc':>17}  {'std':>6}  config")
    for rank, i in enumerate(order[:10], 1):
        params = {k.replace("mlp__", ""): v for k, v in results["params"][i].items()}
        print(f"{rank:>4}  {results['mean_test_score'][i]:>16.1%}  "
              f"{results['std_test_score'][i]:>5.1%}  {params}")

    best = {k.replace("mlp__", ""): v for k, v in search.best_params_.items()}
    print(f"\nbest cross-signer accuracy: {search.best_score_:.1%}")
    print(f"best config: {best}")
    print(f"search took {elapsed:.1f} min")

    # Worth knowing whether the search actually found anything: if the spread
    # between best and worst is small, the defaults were already fine and the
    # honest conclusion is that tuning did not help.
    spread = results["mean_test_score"][order[0]] - results["mean_test_score"][order[-1]]
    print(f"spread across {len(order)} configs: {spread:.1%}")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump({"best_params": {k: list(v) if isinstance(v, tuple) else v
                                   for k, v in best.items()},
                   "best_score": float(search.best_score_),
                   "features": args.features,
                   "subsampled": not args.full,
                   "spread": float(spread)}, handle, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
