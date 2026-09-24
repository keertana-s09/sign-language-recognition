"""Run MediaPipe Hands over the ASL Alphabet images and save normalized landmarks.

This is the one-time expensive pass. Everything downstream trains on the tiny
.npz this produces (63 floats per image) instead of the 87k JPEGs.

    python src/extract_landmarks.py --input data/raw/asl_alphabet_train --out data/processed/landmarks.npz
"""

import argparse
import os
import re
from multiprocessing import Pool

import numpy as np

# 21 keypoints, (x, y, z) each
N_LANDMARKS = 21
WRIST = 0
MIDDLE_MCP = 9  # knuckle of the middle finger - stable reference for hand scale


def normalize(landmarks):
    """Make landmarks invariant to where the hand is, how big it is, and which hand it is.

    Without this the model memorizes hand position in frame and falls apart the
    moment you move closer to the camera.
    """
    lm = np.asarray(landmarks, dtype=np.float32).reshape(N_LANDMARKS, 3)

    # 1. Translate: wrist becomes the origin.
    lm = lm - lm[WRIST]

    # 2. Scale: wrist-to-middle-knuckle becomes unit length, so distance from
    #    the camera stops mattering.
    scale = np.linalg.norm(lm[MIDDLE_MCP])
    if scale < 1e-6:
        return None
    lm = lm / scale

    return lm.flatten()


def normalize_two(hands):
    """Pack one *or* two hands into a fixed 127-float vector.

    ISL mixes one-handed signs (all digits, plus I/J/L/N/O/U/V) with two-handed
    ones (A/B/D/E/F/...), so a single representation has to cover both.

    Crucially, both hands are normalized in ONE shared frame anchored on the
    primary hand. Normalizing each hand independently would discard where the
    hands sit relative to each other - which is precisely what distinguishes
    many two-handed letters.

    Layout: [primary 63][secondary 63][secondary_present 1]

    `hands` is a list of (coords, handedness_label).
    """
    if not hands:
        return None

    # Primary = the signer's right hand when we can tell, else the only hand.
    # Deterministic ordering matters: swapping the roles between training and
    # inference would silently scramble every two-handed class.
    primary_idx = 0
    for i, (_, handed) in enumerate(hands):
        if handed == "Right":
            primary_idx = i
            break

    primary = np.asarray(hands[primary_idx][0], dtype=np.float32).reshape(N_LANDMARKS, 3)

    origin = primary[WRIST].copy()
    scale = np.linalg.norm(primary[MIDDLE_MCP] - origin)
    if scale < 1e-6:
        return None

    vector = np.zeros(N_LANDMARKS * 3 * 2 + 1, dtype=np.float32)
    vector[:63] = ((primary - origin) / scale).flatten()

    others = [h for i, h in enumerate(hands) if i != primary_idx]
    if others:
        secondary = np.asarray(others[0][0], dtype=np.float32).reshape(N_LANDMARKS, 3)
        # Same origin and scale as the primary hand - keeps relative placement.
        vector[63:126] = ((secondary - origin) / scale).flatten()
        vector[126] = 1.0

    return vector


def _worker_init(model_path, two_hands):
    global _detector, _two_hands
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
    from mediapipe.tasks.python.vision import RunningMode

    _two_hands = two_hands
    _detector = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=2 if two_hands else 1,
            min_hand_detection_confidence=0.3,  # low: dataset crops are tight and awkward
        )
    )


def group_of(path, pattern):
    """Which signer produced this image.

    ASL-HG encodes it in the filename (P10_F_980.jpg -> P10). Knowing this is
    what makes a leave-one-signer-out test possible, which is the only honest
    way to predict accuracy on a person the model has never seen.
    """
    if not pattern:
        return "unknown"
    match = re.search(pattern, os.path.basename(path))
    return match.group(1) if match else "unknown"


def _process(task):
    """Returns (features, label) or (None, label) when no hand is found."""
    path, label = task
    import cv2
    import mediapipe as mp

    image = cv2.imread(path)
    if image is None:
        return None, label

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = _detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.hand_landmarks:
        return None, label

    if _two_hands:
        # No mirroring here: in two-handed signs the hands play different roles,
        # so flipping one onto the other would destroy the distinction.
        hands = [
            ([(p.x, p.y, p.z) for p in lm],
             result.handedness[i][0].category_name if result.handedness else "Unknown")
            for i, lm in enumerate(result.hand_landmarks)
        ]
        return normalize_two(hands), label

    coords = [(p.x, p.y, p.z) for p in result.hand_landmarks[0]]

    # Mirror left hands onto right so both spell the same letter identically.
    if result.handedness and result.handedness[0][0].category_name == "Left":
        coords = [(-x, y, z) for x, y, z in coords]

    return normalize(coords), label


def _process_with_group(task):
    path, label, group = task
    vector, _ = _process((path, label))
    return vector, label, group


def collect_tasks(input_dir):
    tasks = []
    classes = sorted(
        d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))
    )
    for label in classes:
        folder = os.path.join(input_dir, label)
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                tasks.append((os.path.join(folder, fname), label))
    return tasks, classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit-per-class", type=int, default=0,
                        help="Cap images per class. Use ~200 for a fast smoke test.")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--model", default="models/hand_landmarker.task")
    parser.add_argument("--two-hands", action="store_true",
                        help="127-feature mode for ISL, which mixes one- and two-handed signs.")
    parser.add_argument("--group-regex", default=None,
                        help=r"Capture the signer id from the filename, e.g. '^(P\d+)_' "
                             r"for ASL-HG. Enables leave-one-signer-out evaluation.")
    args = parser.parse_args()

    tasks, classes = collect_tasks(args.input)

    if args.limit_per_class:
        capped, seen = [], {}
        for path, label in tasks:
            if seen.get(label, 0) < args.limit_per_class:
                capped.append((path, label))
                seen[label] = seen.get(label, 0) + 1
        tasks = capped

    print(f"{len(classes)} classes, {len(tasks)} images, {args.workers} workers")

    grouped = [(path, label, group_of(path, args.group_regex)) for path, label in tasks]

    features, labels, groups, misses = [], [], [], {}
    with Pool(args.workers, initializer=_worker_init,
              initargs=(args.model, args.two_hands)) as pool:
        for i, (vec, label, group) in enumerate(
                pool.imap_unordered(_process_with_group, grouped, chunksize=64)):
            if vec is None:
                misses[label] = misses.get(label, 0) + 1
            else:
                features.append(vec)
                labels.append(label)
                groups.append(group)

            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(tasks)}  kept={len(features)}", flush=True)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels)
    g = np.array(groups)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, groups=g, classes=np.array(classes))

    if args.group_regex:
        found = sorted(set(groups))
        print(f"\nsigners: {len(found)} -> {found}")

    print(f"\nSaved {X.shape} to {args.out}")

    # A class that loses most of its images here will never train well - the
    # detector simply can't see the hand in those poses. Worth knowing now.
    if misses:
        print("\nImages where no hand was detected:")
        for label in sorted(misses, key=lambda k: -misses[k])[:10]:
            total = sum(1 for _, l in tasks if l == label)
            print(f"  {label:>8}: {misses[label]:>5} / {total} ({misses[label] / total:.0%})")


if __name__ == "__main__":
    main()
