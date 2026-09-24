"""Find out *why* live accuracy is worse than test accuracy.

Prompts you to hold specific letters, then compares the live feature vectors
against the training distribution.

    python src/diagnose.py --lang asl --letters A B C L V

The number that matters is the distance ratio. Every live frame is measured
against its nearest training example of the same letter, and compared to how
far apart training examples of that letter are from each other:

  ratio ~1-2x  -> live features look like training data. The model should work;
                  if it doesn't, the problem is thresholds, not data.
  ratio >3x    -> distribution shift. Your hand sits outside what the model has
                  ever seen, and no amount of tuning fixes that. Capture your own.

It also reports accuracy with the hand held still vs rotated, which separates
orientation sensitivity from general distribution shift.
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
from sklearn.neighbors import NearestNeighbors

import features as F
from extract_landmarks import normalize, normalize_two

FRAMES_PER_LETTER = 45  # ~1.5s of holding


def build_features(result, language, swap_handedness):
    if not result.hand_landmarks:
        return None

    def label_of(i):
        name = (result.handedness[i][0].category_name
                if result.handedness else "Unknown")
        if swap_handedness and name in ("Left", "Right"):
            name = "Right" if name == "Left" else "Left"
        return name

    if language == "isl":
        hands = [([(p.x, p.y, p.z) for p in lm], label_of(i))
                 for i, lm in enumerate(result.hand_landmarks)]
        return normalize_two(hands)

    coords = [(p.x, p.y, p.z) for p in result.hand_landmarks[0]]
    if label_of(0) == "Left":
        coords = [(-x, y, z) for x, y, z in coords]
    return normalize(coords)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["asl", "isl"], default="asl")
    parser.add_argument("--letters", nargs="+", default=["A", "B", "C", "L", "V"])
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--swap-handedness", action="store_true", default=True)
    parser.add_argument("--model", help="Defaults to the model for --lang.")
    parser.add_argument("--data", nargs="+",
                        help="Reference set(s) for the distance ratio. Pass every file "
                             "the model was trained on, or the ratio compares your hand "
                             "against data the model never used.")
    args = parser.parse_args()

    model_path = args.model or ("models/classifier.joblib" if args.lang == "asl"
                                else "models/isl_classifier.joblib")
    data_paths = args.data or [("data/processed/landmarks.npz" if args.lang == "asl"
                                else "data/processed/isl_landmarks.npz")]

    bundle = joblib.load(model_path)
    classifier, classes = bundle["model"], bundle["classes"]
    feature_mode = bundle.get("features", "raw")

    print(f"model     : {model_path}  (features: {feature_mode})")
    print(f"reference : {', '.join(data_paths)}\n")

    blobs = [np.load(p, allow_pickle=True) for p in data_paths]
    X_train = np.concatenate([b["X"] for b in blobs])
    y_train = np.concatenate([b["y"] for b in blobs])

    unknown = [l for l in args.letters if l not in set(classes)]
    if unknown:
        sys.exit(f"Not classes in this model: {unknown}\nAvailable: {list(classes)}")

    # Per-letter neighbour index, plus the intra-class spread that sets the baseline.
    neighbours, baseline = {}, {}
    for letter in args.letters:
        subset = X_train[y_train == letter]
        nn = NearestNeighbors(n_neighbors=2).fit(subset)
        neighbours[letter] = NearestNeighbors(n_neighbors=1).fit(subset)
        sample = subset[np.random.RandomState(0).choice(len(subset),
                                                        min(500, len(subset)), replace=False)]
        # distance to nearest *other* training example of the same letter
        d, _ = nn.kneighbors(sample)
        baseline[letter] = float(np.median(d[:, 1]))

    detector = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
        )
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera}")

    results = {}

    for letter in args.letters:
        for phase in ("still", "rotated"):
            hint = ("hold it steady" if phase == "still"
                    else "now slowly TILT and ROTATE your hand while holding it")
            collected = []

            deadline = time.time() + 3.0
            while time.time() < deadline:  # countdown
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                left = deadline - time.time()
                cv2.putText(frame, f"Sign '{letter}' - {hint}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 180, 250), 2)
                cv2.putText(frame, f"{left:.0f}", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
                cv2.imshow("diagnose", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cap.release(); cv2.destroyAllWindows(); return

            while len(collected) < FRAMES_PER_LETTER:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                    int(time.time() * 1000))

                features = build_features(result, args.lang, args.swap_handedness)
                if features is not None:
                    # Predict in the model's feature space, but measure the
                    # distance ratio in raw landmark space - that keeps the
                    # ratio comparable to every earlier run and to the
                    # thresholds the verdict below is calibrated against.
                    vector = ([features] if feature_mode == "raw"
                              else F.transform([features], feature_mode))
                    probabilities = classifier.predict_proba(vector)[0]
                    index = int(np.argmax(probabilities))
                    distance, _ = neighbours[letter].kneighbors([features])
                    collected.append((classes[index], float(probabilities[index]),
                                      float(distance[0][0])))

                cv2.putText(frame, f"RECORDING '{letter}' ({phase}) {len(collected)}/{FRAMES_PER_LETTER}",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 230, 120), 2)
                cv2.imshow("diagnose", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cap.release(); cv2.destroyAllWindows(); return

            results[(letter, phase)] = collected

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 74)
    print(f"DIAGNOSTIC REPORT - {args.lang.upper()}")
    print("=" * 74)
    print(f"\n{'letter':>7} {'phase':>8} {'correct':>8} {'conf':>7} "
          f"{'live dist':>10} {'train dist':>11} {'ratio':>7}")

    ratios, still_acc, rot_acc = [], [], []
    for (letter, phase), rows in results.items():
        if not rows:
            print(f"{letter:>7} {phase:>8} {'NO HAND DETECTED':>36}")
            continue
        correct = sum(1 for p, _, _ in rows if p == letter) / len(rows)
        confidence = float(np.mean([c for _, c, _ in rows]))
        live = float(np.median([d for _, _, d in rows]))
        ratio = live / baseline[letter] if baseline[letter] > 0 else float("nan")
        ratios.append(ratio)
        (still_acc if phase == "still" else rot_acc).append(correct)
        print(f"{letter:>7} {phase:>8} {correct:>7.0%} {confidence:>7.2f} "
              f"{live:>10.3f} {baseline[letter]:>11.3f} {ratio:>6.1f}x")

        wrong = defaultdict(int)
        for p, _, _ in rows:
            if p != letter:
                wrong[p] += 1
        if wrong:
            top = sorted(wrong.items(), key=lambda kv: -kv[1])[:3]
            print(f"{'':>16} mistaken for: " + ", ".join(f"{k} x{v}" for k, v in top))

    print("\n" + "-" * 74)
    median_ratio = float(np.median(ratios)) if ratios else float("nan")
    print(f"median distance ratio : {median_ratio:.1f}x")
    if still_acc:
        print(f"accuracy held still   : {np.mean(still_acc):.0%}")
    if rot_acc:
        print(f"accuracy rotated      : {np.mean(rot_acc):.0%}")

    print("\nVERDICT")
    if median_ratio > 3:
        print("  DISTRIBUTION SHIFT. Your hand is far outside the training data.")
        print("  Tuning will not fix this - capture your own samples.")
    elif still_acc and rot_acc and np.mean(still_acc) - np.mean(rot_acc) > 0.25:
        print("  ORIENTATION SENSITIVITY. Fine held still, breaks when rotated.")
        print("  Fix normalization to be rotation-invariant; no new data needed.")
    elif still_acc and np.mean(still_acc) > 0.7:
        print("  Model is largely working. Tune MIN_CONFIDENCE / HOLD_FRAMES.")
    else:
        print("  Mixed. Check the per-letter rows above for which letters fail.")


if __name__ == "__main__":
    main()
