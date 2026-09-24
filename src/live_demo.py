"""Live webcam fingerspelling -> text, for ASL and ISL.

    python src/live_demo.py            # starts in ASL
    python src/live_demo.py --lang isl

Keys:  q quit   c clear   BACKSPACE delete last   l switch language
       h swap handedness convention (fixes scrambled two-handed ISL letters)

The hard part here isn't prediction, it's *debouncing*. The classifier fires ~30
times a second, so naively appending every prediction gives you "AAAAAAAAAA".
A letter is only committed once it has been held steadily for HOLD_FRAMES, and
then the same letter is locked out until you change handshape.
"""

import argparse
import os
import time
from collections import deque

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

import features as F
from extract_landmarks import normalize, normalize_two

# Tuning knobs - these are what you adjust when the demo feels wrong.
MIN_CONFIDENCE = 0.70   # below this, show the guess greyed out but don't commit
HOLD_FRAMES = 12        # ~0.4s at 30fps: how long to hold a sign to commit it
SMOOTHING = 5           # majority vote over this many frames, kills single-frame flicker

# Landmark pairs to draw, so the overlay looks like a hand and not a dot cloud.
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (5, 9), (9, 10), (10, 11), (11, 12),      # middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (0, 17),                                  # palm base
]


def draw_hand(frame, landmarks, color):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 3, color, -1)


def draw_ui(frame, letter, confidence, progress, sentence, fps, language):
    h, w = frame.shape[:2]

    # Top banner: current prediction + how close it is to committing.
    cv2.rectangle(frame, (0, 0), (w, 90), (24, 24, 24), -1)

    if letter is None:
        cv2.putText(frame, "no hand", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (120, 120, 120), 2)
    else:
        committed = confidence >= MIN_CONFIDENCE
        color = (80, 230, 120) if committed else (120, 120, 120)
        cv2.putText(frame, letter, (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 3)
        cv2.putText(frame, f"{confidence:.0%}", (110, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Progress bar toward commit - makes the hold time legible instead of mysterious.
        if committed and progress > 0:
            bar_w = int(200 * progress)
            cv2.rectangle(frame, (200, 34), (400, 56), (60, 60, 60), -1)
            cv2.rectangle(frame, (200, 34), (200 + bar_w, 56), (80, 230, 120), -1)

    cv2.putText(frame, f"{fps:.0f} fps", (w - 110, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    cv2.putText(frame, language.upper(), (w - 110, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 180, 250), 2)

    # Bottom: the sentence being built.
    cv2.rectangle(frame, (0, h - 80), (w, h), (24, 24, 24), -1)
    shown = sentence[-32:] if len(sentence) > 32 else sentence
    cv2.putText(frame, shown + "_", (20, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, "q quit   c clear   backspace delete   l language   h swap hands",
                (20, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 130), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asl-model", default="models/balanced.joblib")
    parser.add_argument("--isl-model", default="models/isl_classifier.joblib")
    parser.add_argument("--landmarker", default="models/hand_landmarker.task")
    parser.add_argument("--lang", choices=["asl", "isl"], default="asl")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    models = {}
    for name, path in (("asl", args.asl_model), ("isl", args.isl_model)):
        if os.path.exists(path):
            bundle = joblib.load(path)
            models[name] = (bundle["model"], bundle["classes"],
                            bundle.get("features", "raw"))
        else:
            print(f"note: no {name.upper()} model at {path}, that language is disabled")

    if not models:
        raise SystemExit("No models found - train at least one first.")

    language = args.lang if args.lang in models else next(iter(models))

    # Two hands always: ISL needs both, and ASL simply ignores the extra one.
    detector = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=args.landmarker),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
        )
    )

    # MediaPipe reports handedness assuming a mirrored (selfie) image. The ISL
    # training images were not mirrored, so live labels come out opposite to the
    # convention the ISL model learned, and we flip them back. Only the *primary
    # hand choice* depends on this, so if two-handed letters look scrambled,
    # press 'h' to swap the convention rather than retraining.
    swap_handedness = True

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    prob_window = deque(maxlen=SMOOTHING)
    sentence = ""
    held_letter, held_count, last_committed = None, 0, None
    frame_times = deque(maxlen=30)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Selfie view. Handedness normalization in normalize() makes this safe.
        frame = cv2.flip(frame, 1)
        start = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
            int(time.time() * 1000),
        )

        letter, confidence = None, 0.0

        if result.hand_landmarks:
            classifier, classes, feature_mode = models[language]

            def label_of(i):
                name = (result.handedness[i][0].category_name
                        if result.handedness else "Unknown")
                if swap_handedness and name in ("Left", "Right"):
                    name = "Right" if name == "Left" else "Left"
                return name

            if language == "isl":
                hands = [([(p.x, p.y, p.z) for p in lm], label_of(i))
                         for i, lm in enumerate(result.hand_landmarks)]
                features = normalize_two(hands)
                landmarks = result.hand_landmarks[0]
            else:
                landmarks = result.hand_landmarks[0]
                coords = [(p.x, p.y, p.z) for p in landmarks]
                if label_of(0) == "Left":
                    coords = [(-x, y, z) for x, y, z in coords]
                features = normalize(coords)

            if features is not None:
                vector = ([features] if feature_mode == "raw"
                          else F.transform([features], feature_mode))
                probabilities = classifier.predict_proba(vector)[0]
                prob_window.append(probabilities)

                avg = np.mean(prob_window, axis=0)
                index = int(np.argmax(avg))
                letter = classes[index]
                confidence = float(avg[index])
                color = (80, 230, 120) if confidence >= MIN_CONFIDENCE else (110, 110, 110)
                # ISL is often two-handed, so draw every hand the model saw.
                for lm in (result.hand_landmarks if language == "isl"
                           else result.hand_landmarks[:1]):
                    draw_hand(frame, lm, color)
        else:
            prob_window.clear()
            last_committed = None

        # Commit logic: hold a confident letter steady, then it lands once.
        if letter is not None and confidence >= MIN_CONFIDENCE:
            if letter == held_letter:
                held_count += 1
            else:
                held_letter, held_count = letter, 1

            if held_count >= HOLD_FRAMES and held_letter != last_committed:
                if held_letter == "space":
                    sentence += " "
                elif held_letter == "del":
                    sentence = sentence[:-1]
                else:
                    sentence += held_letter
                last_committed = held_letter
                held_count = 0
        else:
            held_letter, held_count = None, 0

        frame_times.append(time.time() - start)
        fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0

        progress = held_count / HOLD_FRAMES if held_letter else 0.0
        draw_ui(frame, letter, confidence, progress, sentence, fps, language)

        cv2.imshow("Fingerspelling", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            sentence = ""
        elif key == 8:  # backspace
            sentence = sentence[:-1]
        elif key == ord("l") and len(models) > 1:
            language = "isl" if language == "asl" else "asl"
            prob_window.clear()
            held_letter, held_count, last_committed = None, 0, None
        elif key == ord("h"):
            swap_handedness = not swap_handedness
            prob_window.clear()

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinal: {sentence!r}")


if __name__ == "__main__":
    main()
