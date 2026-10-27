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
    parser.add_argument("--faces", help="Directory of <person>/<images>")
    parser.add_argument("--detector", help="YuNet ONNX path")
    parser.add_argument("--embedder", help="SFace ONNX path")
    parser.add_argument("--registry", help="Read-only existing face registry database")
    parser.add_argument("--photo-manifest", help="Labelled face-photo export manifest.csv; diagnostic, not held-out footage")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--json-output", help="Registry diagnostic report path")
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

    if args.registry:
        if not args.photo_manifest or not args.json_output:
            parser.error('--registry needs --photo-manifest and --json-output')
        return evaluate_registry_photos(args.registry, args.photo_manifest, args.config, args.json_output)
    if not args.faces or not args.detector or not args.embedder:
        parser.error('--faces, --detector and --embedder are required without --registry')

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


def evaluate_registry_photos(registry: str, manifest: str, config_path: str, output: str) -> int:
    """Score supplied labels against real enrolled templates without editing them.

    Enrollment exports are leakage-prone, so these numbers are explicitly
    diagnostics. No enrollment photo is claimed to be independent CCTV evidence.
    Every supplied photo stays in the denominator, including no-face failures.
    """
    import csv
    import json
    import sqlite3
    import time
    import cv2
    from computer_vision.core.config import load_config
    from computer_vision.inference.face_factory import create_face_detector, create_face_embedder
    from tools.benchmark_cv import registry_store
    from tools.cv_metrics import timing

    config = load_config(config_path, use_env=False)
    detection = config.module_config('face_detection').params
    recognition = config.module_config('face_recognition').params
    threshold, margin = float(recognition.get('threshold', 0.363)), float(recognition.get('margin', 0.05))
    store = registry_store(registry, threshold)
    with sqlite3.connect(Path(registry).resolve().as_uri() + '?mode=ro', uri=True) as db:
        users = list(db.execute('SELECT name, employee_id FROM users WHERE active=1'))
    by_employee = {employee.casefold(): name for name, employee in users if employee}
    by_name = {name.casefold(): name for name, _ in users}
    detector = create_face_detector(detection.get('backend', 'yunet'), detection)
    embedder = create_face_embedder(recognition.get('backend', 'sface'), recognition)
    detector.load()
    embedder.load()
    rows, durations = [], []
    source = Path(manifest)
    with source.open(encoding='utf-8-sig', newline='') as handle:
        labels = list(csv.DictReader(handle))
    try:
        for row in labels:
            expected = by_employee.get(row.get('employee_id', '').casefold()) or by_name.get(row['name'].casefold(), row['name'])
            path = source.parent / row['file']
            record = {'file': row['file'], 'expected': expected, 'enrolled': expected in store.names,
                      'predicted': None, 'similarity': None, 'max_own_template_cosine': None}
            started = time.perf_counter()
            image = cv2.imread(str(path))
            try:
                if image is None:
                    raise ValueError('unreadable image')
                faces = detector.detect(image)
                record['faces'] = len(faces)
                if len(faces) != 1:
                    raise ValueError(f'expected exactly one face, detected {len(faces)}')
                vector = embedder.embed(image, faces[0])
                match = store.match(vector, threshold=threshold, margin=margin)
                record.update(predicted=match.name if match.matched else None, similarity=float(match.similarity))
                identity = store.get(expected)
                if identity is not None:
                    unit = vector / max(float(np.linalg.norm(vector)), 1e-12)
                    record['max_own_template_cosine'] = max(float(np.dot(unit, v / max(float(np.linalg.norm(v)), 1e-12))) for v in identity.embeddings)
            except Exception as exc:
                record['error'] = f'{type(exc).__name__}: {exc}'
            durations.append((time.perf_counter() - started) * 1000)
            rows.append(record)
    finally:
        detector.unload()
        embedder.unload()

    def summarize(records):
        known = sum(r['enrolled'] for r in records)
        unknown = len(records) - known
        correct = sum(r['enrolled'] and r['predicted'] == r['expected'] for r in records)
        wrong = sum(r['enrolled'] and r['predicted'] is not None and r['predicted'] != r['expected'] for r in records)
        rejected = sum(r['enrolled'] and r['predicted'] is None for r in records)
        accepted_unknown = sum(not r['enrolled'] and r['predicted'] is not None for r in records)
        accepted = correct + wrong + accepted_unknown
        return {'samples': len(records), 'known_samples': known, 'unknown_samples': unknown,
                'correct_known': correct, 'wrong_identity': wrong, 'false_rejects': rejected,
                'false_accepts_of_unknown': accepted_unknown,
                'precision': correct / accepted if accepted else None,
                'recall': correct / known if known else None,
                'false_accept_rate': accepted_unknown / unknown if unknown else None,
                'false_reject_rate': rejected / known if known else None,
                'wrong_identity_rate': wrong / known if known else None}

    report = {'protocol': 'Single-image registered-gallery diagnostic on supplied enrollment-export labels',
              'independent_cctv_accuracy': 'not_measured',
              'limitations': ['Export labels are supplied metadata, not independently verified names.',
                             'Enrollment-export photos may overlap registry training samples or the same capture session.',
                             'Removing near-identical templates does not make the remaining photos independent.',
                             'Production temporal confirmation, body continuity and face quality gates are not evaluated by this single-image diagnostic.'],
              'registry': registry, 'identities': len(store), 'templates': store.sample_count,
              'threshold': threshold, 'margin': margin,
              'detector_parameters': detection, 'embedder_parameters': recognition,
              'all_export_photos': summarize(rows),
              'near_identical_template_photos': sum((r['max_own_template_cosine'] or 0) >= 0.9995 for r in rows),
              'non_identical_export_subset': summarize([r for r in rows if (r['max_own_template_cosine'] or 0) < 0.9995]),
              'image_decode_detect_embed_match_latency': timing(durations), 'predictions': rows}
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k in {'identities', 'templates', 'all_export_photos', 'near_identical_template_photos', 'independent_cctv_accuracy'}}))
    return 0 if rows else 2


if __name__ == "__main__":
    sys.exit(main())
