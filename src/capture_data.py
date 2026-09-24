"""Record your own landmark samples from the webcam.

The Kaggle datasets are single-signer and generalize poorly - measured at ~12x
the intra-class distance for a new hand. This closes that gap.

    python src/capture_data.py --lang asl --signer keertana
    python src/capture_data.py --lang isl --signer keertana
    python src/capture_data.py --lang isl --signer friend1   # second person

Each person runs it with their own --signer name. All recordings merge into one
file with signer labels, which enables leave-one-signer-out evaluation.

Saves after EVERY class, so quitting partway keeps what you already recorded,
and re-running merges into the same file.

Keys during capture:  s skip this class   q save and quit
"""

import argparse
import glob
import os
import time

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

from extract_landmarks import normalize, normalize_two

ASL_CLASSES = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["space", "del"]
ISL_CLASSES = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [str(d) for d in range(1, 10)]

# Cycled through while recording. Variety is the entire point - samples all taken
# from one pose at one distance reproduce the exact problem we are fixing.
PROMPTS = [
    "hold steady",
    "TILT your hand left and right",
    "move CLOSER to the camera",
    "move BACK from the camera",
    "ROTATE your wrist slowly",
    "shift your hand around the frame",
]


def build_features(result, language, swap_handedness=True):
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


def load_reference(reference_dir, label):
    """A training image of this sign, shown as a thumbnail so you can copy the pose."""
    if not reference_dir:
        return None
    for pattern in (label, label.upper()):
        matches = glob.glob(os.path.join(reference_dir, pattern, "*"))
        if matches:
            image = cv2.imread(matches[0])
            if image is not None:
                return cv2.resize(image, (160, 160))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["asl", "isl"], default="asl")
    parser.add_argument("--classes", nargs="+")
    parser.add_argument("--samples", type=int, default=200,
                        help="Frames per class. 200 takes about 8 seconds each.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--reference-dir",
                        help="Dataset dir with per-class folders, for pose thumbnails.")
    parser.add_argument("--signer",
                        help="Signer id (e.g. 'keertana', 'friend1'). Each "
                             "person uses their own name. Enables LOSO eval.")
    parser.add_argument("--out")
    args = parser.parse_args()

    classes = args.classes or (ASL_CLASSES if args.lang == "asl" else ISL_CLASSES)
    out_path = args.out or f"data/processed/own_{args.lang}.npz"

    signer_id = args.signer
    if not signer_id:
        signer_id = input("Enter signer name/id (e.g. your first name): ").strip()
        if not signer_id:
            raise SystemExit("A signer id is required for multi-person evaluation.")
    print(f"Recording as signer: {signer_id}")

    features, labels, groups = [], [], []
    if os.path.exists(out_path):
        blob = np.load(out_path, allow_pickle=True)
        features, labels = list(blob["X"]), list(blob["y"])
        groups = (list(blob["groups"]) if "groups" in blob.files
                  else ["unknown"] * len(labels))
        signers = sorted(set(groups))
        done = {l: labels.count(l) for l in set(labels)}
        print(f"Resuming {out_path}: {len(labels)} samples from signers {signers}")
        print("  " + ", ".join(f"{k}={v}" for k, v in sorted(done.items())))

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
        raise SystemExit(f"Could not open camera {args.camera}")

    def save():
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path,
                            X=np.array(features, dtype=np.float32),
                            y=np.array(labels),
                            groups=np.array(groups),
                            classes=np.array(sorted(set(labels))))

    quit_early = False

    for position, label in enumerate(classes, 1):
        if quit_early:
            break

        reference = load_reference(args.reference_dir, label)
        collected = 0

        # Countdown, so you have time to form the handshape.
        deadline = time.time() + 4.0
        while time.time() < deadline:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            remaining = deadline - time.time()
            cv2.putText(frame, f"[{position}/{len(classes)}]  Get ready: '{label}'",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90, 180, 250), 2)
            cv2.putText(frame, f"{remaining:.0f}", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 4)
            cv2.putText(frame, "s skip   q save and quit", (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
            if reference is not None:
                frame[20:180, frame.shape[1] - 180:frame.shape[1] - 20] = reference
            cv2.imshow("capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                quit_early = True
                break
            if key == ord("s"):
                deadline = 0
                collected = args.samples  # skip straight past recording
                break

        while collected < args.samples and not quit_early:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                int(time.time() * 1000))

            vector = build_features(result, args.lang)
            if vector is not None:
                features.append(vector)
                labels.append(label)
                groups.append(signer_id)
                collected += 1

            # Change the instruction every ~1/6 of the way through.
            step = max(1, args.samples // len(PROMPTS))
            prompt = PROMPTS[min(collected // step, len(PROMPTS) - 1)]

            found = bool(result.hand_landmarks)
            cv2.putText(frame, f"'{label}'  {collected}/{args.samples}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (80, 230, 120) if found else (60, 60, 220), 2)
            cv2.putText(frame, prompt if found else "NO HAND DETECTED", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255) if found else (60, 60, 220), 2)

            width = int((frame.shape[1] - 40) * collected / args.samples)
            cv2.rectangle(frame, (20, 110), (20 + width, 122), (80, 230, 120), -1)
            if reference is not None:
                frame[20:180, frame.shape[1] - 180:frame.shape[1] - 20] = reference

            cv2.imshow("capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                quit_early = True
            elif key == ord("s"):
                break

        save()  # after every class, so nothing is ever lost
        print(f"  {label}: {labels.count(label)} total samples")

    cap.release()
    cv2.destroyAllWindows()
    save()

    signers = sorted(set(groups))
    print(f"\nSaved {len(labels)} samples across {len(set(labels))} classes -> {out_path}")
    print(f"Signers: {signers} ({[groups.count(s) for s in signers]} samples each)")
    missing = [c for c in classes if c not in set(labels)]
    if missing:
        print(f"Still missing: {missing}")


if __name__ == "__main__":
    main()
