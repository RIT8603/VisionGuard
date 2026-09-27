"""Measure face-recognition accuracy against labelled images.

The pipeline measures FPS continuously and accuracy not at all, because
accuracy needs ground truth. This supplies it.

Usage::

    python -m tools.evaluate_recognition --faces data/faces \\
        --detector computer_vision/Model/face/face_detection_yunet_2023mar.onnx \\
        --embedder computer_vision/Model/face/face_recognition_sface_2021dec.onnx

Expected layout - one directory per person::

    data/faces/
      alice/  img01.jpg img02.jpg img03.jpg ...
      bob/    img01.jpg ...

Images are split per person: the first ``--enroll`` of each go into the
identity store, the rest become the test set. Someone with only enrolment
images contributes nothing to the score, which is reported rather than hidden.

What it reports, and why each matters for a surveillance deployment:

* **Accuracy** - overall correctness. The least useful number on its own.
* **False accept rate** - how often a stranger is mistaken for an enrolled
  person. In an access-control setting this is the dangerous error.
* **False reject rate** - how often a known person is called unknown. This is
  the annoying error, and the one users complain about.
* **Threshold sweep** - the two above trade against each other. The right
  threshold depends on your cameras and how many people are enrolled, so pick
  it from this table rather than trusting the default.

Exit codes: 0 success, 1 bad usage or unreadable data, 2 nothing to evaluate.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Sample:
    """One test image and what it should be recognised as."""

    person: str
    path: Path
    embedding: np.ndarray


@dataclass
class Outcome:
    """Scores at one threshold."""

    threshold: float
    correct: int = 0
    false_accept: int = 0
    false_reject: int = 0
    wrong_identity: int = 0
    total: int = 0
    #: Test images whose subject was never enrolled, so a match is wrong.
    strangers: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def false_accept_rate(self) -> float:
        return self.false_accept / self.strangers if self.strangers else 0.0

    @property
    def false_reject_rate(self) -> float:
        known = self.total - self.strangers
        return self.false_reject / known if known else 0.0


def find_people(root: Path) -> Dict[str, List[Path]]:
    """Map person name -> their image files."""
    if not root.is_dir():
        raise SystemExit(f"[error] Not a directory: {root}")

    people: Dict[str, List[Path]] = {}
    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        images = sorted(
            f for f in person_dir.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES
        )
        if images:
            people[person_dir.name] = images
    return people


def embed_images(
    paths: Sequence[Path], detector, embedder, verbose: bool = False
) -> List[Tuple[Path, np.ndarray]]:
    """Detect the largest face in each image and embed it."""
    import cv2

    results: List[Tuple[Path, np.ndarray]] = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            if verbose:
                print(f"  ! unreadable: {path.name}")
            continue

        faces = detector.detect(image)
        if not faces:
            if verbose:
                print(f"  ! no face found: {path.name}")
            continue

        # Largest face: in an enrolment photo the subject is the big one.
        face = max(faces, key=lambda f: (f.xyxy[2] - f.xyxy[0]) * (f.xyxy[3] - f.xyxy[1]))
        try:
            results.append((path, embedder.embed(image, face)))
        except Exception as exc:  # noqa: BLE001 - one bad image is not fatal
            if verbose:
                print(f"  ! could not embed {path.name}: {exc}")
    return results


def evaluate(
    store, samples: Sequence[Sample], threshold: float, enrolled: set
) -> Outcome:
    """Score every test sample at one threshold."""
    outcome = Outcome(threshold=threshold)

    for sample in samples:
        outcome.total += 1
        is_stranger = sample.person not in enrolled
        if is_stranger:
            outcome.strangers += 1

        match = store.match(sample.embedding, threshold=threshold)

        if is_stranger:
            if match.matched:
                outcome.false_accept += 1
            else:
                outcome.correct += 1
        else:
            if not match.matched:
                outcome.false_reject += 1
            elif match.name != sample.person:
                outcome.wrong_identity += 1
            else:
                outcome.correct += 1

    return outcome


def sweep(store, samples, thresholds: Sequence[float], enrolled: set) -> List[Outcome]:
    return [evaluate(store, samples, t, enrolled) for t in thresholds]


def print_report(
    best: Outcome, outcomes: Sequence[Outcome], enrolled: set, skipped: Dict[str, int]
) -> None:
    print("\n" + "=" * 66)
    print("FACE RECOGNITION ACCURACY")
    print("=" * 66)
    print(f"Enrolled identities : {len(enrolled)}  ({', '.join(sorted(enrolled))})")
    print(f"Test images         : {best.total}  ({best.strangers} of unenrolled people)")
    if skipped:
        total_skipped = sum(skipped.values())
        print(f"Skipped             : {total_skipped} (no detectable face)")

    print(f"\nAt threshold {best.threshold:.3f}:")
    print(f"  accuracy            {best.accuracy:6.1%}")
    print(f"  correct             {best.correct:6d}")
    print(f"  false rejects       {best.false_reject:6d}   ({best.false_reject_rate:.1%} of known)")
    print(f"  wrong identity      {best.wrong_identity:6d}")
    print(f"  false accepts       {best.false_accept:6d}   ({best.false_accept_rate:.1%} of strangers)")

    print("\nThreshold sweep  (FAR = stranger accepted, FRR = known rejected)")
    print(f"  {'thresh':>7} {'acc':>7} {'FAR':>7} {'FRR':>7} {'wrong':>6}")
    print("  " + "-" * 38)
    for outcome in outcomes:
        marker = " <-" if outcome.threshold == best.threshold else ""
        print(
            f"  {outcome.threshold:7.3f} {outcome.accuracy:7.1%} "
            f"{outcome.false_accept_rate:7.1%} {outcome.false_reject_rate:7.1%} "
            f"{outcome.wrong_identity:6d}{marker}"
        )

    print(
        "\nPick the threshold from this table, not the default: raising it "
        "\ntrades false accepts for false rejects. For access control favour "
        "\na low FAR; for convenience favour a low FRR."
    )
    print("=" * 66)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evaluate_recognition",
        description="Measure face recognition accuracy on labelled images.",
    )
    parser.add_argument("--faces", required=True, help="Directory of <person>/<images>")
    parser.add_argument("--detector", required=True, help="YuNet ONNX path")
    parser.add_argument("--embedder", required=True, help="SFace ONNX path")
    parser.add_argument(
        "--enroll", type=int, default=2, help="Images per person to enrol (default 2)"
    )
    parser.add_argument(
        "--hold-out",
        default="",
        help="Comma-separated people to NOT enrol, to measure false accepts",
    )
    parser.add_argument("--threshold", type=float, default=0.363)
    parser.add_argument("--save-store", help="Write the enrolled store here")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    from computer_vision.identity import IdentityStore
    from computer_vision.inference.opencv_face import SFaceEmbedder, YuNetFaceDetector
    from computer_vision.utils.logger import setup_logging

    setup_logging("WARNING")

    people = find_people(Path(args.faces))
    if not people:
        print(f"[error] No <person>/<images> directories under {args.faces}", file=sys.stderr)
        return 1

    detector = YuNetFaceDetector({"model": args.detector})
    embedder = SFaceEmbedder({"model": args.embedder})
    detector.load()
    embedder.load()

    hold_out = {n.strip() for n in args.hold_out.split(",") if n.strip()}
    store = IdentityStore(threshold=args.threshold)
    samples: List[Sample] = []
    enrolled: set = set()
    skipped: Dict[str, int] = {}

    print(f"Processing {len(people)} people from {args.faces} ...")
    for person, images in people.items():
        if args.verbose:
            print(f"  {person}: {len(images)} image(s)")

        embedded = embed_images(images, detector, embedder, args.verbose)
        missing = len(images) - len(embedded)
        if missing:
            skipped[person] = missing

        if person in hold_out:
            # Never enrolled: every one of their images is a stranger.
            samples.extend(Sample(person, p, e) for p, e in embedded)
            continue

        for index, (path, embedding) in enumerate(embedded):
            if index < args.enroll:
                store.enroll(person, embedding)
                enrolled.add(person)
            else:
                samples.append(Sample(person, path, embedding))

    if not samples:
        print(
            "[error] No test images left after enrolment. Give each person "
            f"more than --enroll ({args.enroll}) images, or use --hold-out.",
            file=sys.stderr,
        )
        return 2
    if store.is_empty():
        print("[error] Nothing was enrolled - no faces detected?", file=sys.stderr)
        return 2

    thresholds = [round(0.20 + 0.05 * i, 3) for i in range(11)]
    if args.threshold not in thresholds:
        thresholds.append(args.threshold)
    thresholds.sort()

    outcomes = sweep(store, samples, thresholds, enrolled)
    best = next(o for o in outcomes if o.threshold == args.threshold)
    print_report(best, outcomes, enrolled, skipped)

    if args.save_store:
        path = store.save(args.save_store)
        print(f"\nSaved identity store: {path}")

    detector.unload()
    embedder.unload()
    return 0


if __name__ == "__main__":
    sys.exit(main())
