"""Recognition rechecks, honest face states, names carried on bodies, registry changes.

Uses the face-registry test fixtures: a fake embedder whose output the test
sets, synthetic face boxes, and the live shared identity store.
"""

from __future__ import annotations

import pytest

from computer_vision.core.context import BBox, Detection, FrameContext, Track
from computer_vision.identity.shared import replace_shared_store
from computer_vision.inference.face_base import RawFace
from tools.tests.test_face_registry import (
    _FLAT,
    BODY,
    BOX,
    FakeEmbedder,
    frame_with_face,
    landmarks,
    make_module,
    run,
    unit,
)


def timed_module(**params):
    """Recognition with the timed rules on (the shared fixture switches them off)."""
    embedder = FakeEmbedder()
    module = make_module(embedder, revalidate_seconds=3.0, body_gap_reverify_seconds=0.5, **params)
    module._live = True  # the fixture skips initialize(); this is the live registry
    return module, embedder


def at(frame_id: int, t: float, **kwargs) -> FrameContext:
    ctx = frame_with_face(frame_id, **kwargs)
    ctx.monotonic_ts = t
    return ctx


def name_asha(module, embedder, persons=()):
    embedder.vector = unit(1)
    run(module, at(1, 0.0, persons=persons))
    det = run(module, at(2, 0.1, persons=persons))
    assert det.label == "Asha"


# ------------------------------------------------------------- rechecking ---


def test_a_confirmed_name_is_revalidated_and_withdrawn_when_it_stops_matching():
    replace_shared_store([("Asha", [unit(1)])])
    module, embedder = timed_module()
    name_asha(module, embedder)

    embedder.vector = unit(9)  # it was never Asha: matches nobody
    # Before revalidate_seconds nothing re-runs: the name stands.
    assert run(module, at(3, 1.0)).label == "Asha"
    # Revalidation: one disagreeing check is not enough...
    assert run(module, at(4, 3.2)).label == "Asha"
    # ...confirm_matches (2) in a row withdraw it. The face now reads "Unknown".
    det = run(module, at(5, 3.3))
    assert det.label == "unknown" and det.extra["state"] == "unknown"


def test_a_wrong_name_is_replaced_by_the_person_it_really_is():
    replace_shared_store([("Asha", [unit(1)]), ("Ravi", [unit(2)])])
    module, embedder = timed_module()
    name_asha(module, embedder)
    embedder.vector = unit(2)
    run(module, at(3, 3.5))
    assert run(module, at(4, 3.6)).label == "Ravi"


def test_a_check_that_still_resembles_the_named_person_is_not_held_against_it():
    replace_shared_store([("Asha", [unit(1)])])
    module, embedder = timed_module(margin=0.05)
    name_asha(module, embedder)
    # A near-identical second identity is enrolled: from now on every check of
    # this face is "too close to call" between the two - yet it is still
    # plainly Asha-like, so it must not count as evidence against the name.
    replace_shared_store([("Asha", [unit(1)]), ("Asha2", [unit(1, 0.02, base=1)])])
    for k in range(6):
        det = run(module, at(10 + k, 5.0 + k * 0.2))
    subject = next(iter(module._by_face.values()))
    assert subject.name == "Asha"            # never withdrawn
    assert det.extra["state"] == "checking"   # but not shown until it is unambiguous again


def test_one_sharp_match_is_never_enough_to_show_a_name():
    replace_shared_store([("Asha", [unit(1)])])
    module, embedder = timed_module()
    embedder.vector = unit(1)
    det = run(module, at(1, 0.0, box=(100, 100, 300, 360)))  # a large, sharp face
    assert det.label == "checking" and det.extra["recognised"] is False


# ------------------------------------------------------------------ states ---


def test_unclear_checking_and_unknown_are_different_states():
    replace_shared_store([("Asha", [unit(1)])])
    module, embedder = timed_module()
    embedder.vector = unit(9)
    unclear = run(module, at(1, 0.0, frame=_FLAT))           # too blurred to compare
    assert unclear.extra["state"] == "unclear" and unclear.label == "face"
    assert unclear.extra["verified"] is False

    stranger = run(module, at(2, 0.1))                        # compared: matches nobody
    assert stranger.extra["state"] == "unknown" and stranger.extra["verified"] is True

    module2, embedder2 = timed_module()
    embedder2.vector = unit(1)
    pending = run(module2, at(1, 0.0))                        # Asha, not yet confirmed
    assert pending.extra["state"] == "checking" and pending.extra["verified"] is False


def test_unknown_person_alerts_only_on_a_real_unknown_not_on_checking():
    from computer_vision.modules.recognition.unknown_person import UnknownPersonModule

    replace_shared_store([("Asha", [unit(1)])])
    module, embedder = timed_module()
    embedder.vector = unit(1)
    ctx = at(1, 0.0)
    module.process(ctx)                                       # "checking" Asha
    alert = UnknownPersonModule({"min_frames": 1, "cooldown_seconds": 0})
    alert.initialize()
    assert alert.process(ctx).data["unknown_faces"] == 0


# --------------------------------------------------------- body continuity ---


BODY_A = (40, 40, 330, 490)


def test_the_name_stays_on_the_body_while_the_face_is_turned_away():
    replace_shared_store([("Asha", [unit(1)])])
    module, embedder = timed_module()
    name_asha(module, embedder, persons=[(5, BODY_A)])
    for k in range(10):  # back turned for ~1 s, body seen every frame
        ctx = at(3 + k, 0.2 + k * 0.1, persons=[(5, BODY_A)], faces=False)
        module.process(ctx)
        assert ctx.metadata["person_identities"][5] == {"name": "Asha", "state": "recognised"}
        assert ctx.get_tracks("person_tracking")[0].extra["identity"] == "Asha"


def test_a_body_that_overlaps_someone_else_must_be_reverified():
    replace_shared_store([("Asha", [unit(1)])])
    module, embedder = timed_module()
    name_asha(module, embedder, persons=[(5, BODY_A)])
    crowd = [(5, BODY_A), (6, (60, 40, 350, 490))]           # someone walks right through
    ctx = at(3, 0.2, persons=crowd, faces=False)
    module.process(ctx)
    entry = ctx.metadata["person_identities"][5]
    assert entry == {"name": None, "state": "checking"}
    assert ctx.get_tracks("person_tracking")[0].extra["identity"] is None
    # The face comes back and matches: the name is shown again.
    embedder.vector = unit(1)
    det = run(module, at(4, 0.3, persons=[(5, BODY_A)]))
    assert det.label == "Asha"


def test_a_body_lost_for_a_while_must_be_reverified():
    replace_shared_store([("Asha", [unit(1)])])
    module, embedder = timed_module()
    name_asha(module, embedder, persons=[(5, BODY_A)])
    module.process(at(3, 0.2, faces=False))                   # body not seen...
    ctx = at(4, 1.5, persons=[(5, BODY_A)], faces=False)      # ...back after 1.3 s
    module.process(ctx)
    assert ctx.metadata["person_identities"][5]["state"] == "checking"


# ---------------------------------------------------------- registry edits ---


def test_deleting_a_user_withdraws_their_name_at_once():
    replace_shared_store([("Asha", [unit(1)]), ("Ravi", [unit(2)])])
    module, embedder = timed_module()
    name_asha(module, embedder, persons=[(5, BODY_A)])
    replace_shared_store([("Ravi", [unit(2)])])               # Asha deleted in the dashboard
    ctx = at(3, 0.2, persons=[(5, BODY_A)], faces=False)
    module.process(ctx)
    assert ctx.metadata.get("person_identities", {}).get(5, {}).get("name") is None


def test_any_registry_change_makes_names_be_confirmed_again():
    replace_shared_store([("Asha", [unit(1)])])
    module, embedder = timed_module()
    name_asha(module, embedder, persons=[(5, BODY_A)])
    replace_shared_store([("Asha", [unit(1), unit(1, 0.1, base=1)])])  # re-enrolled
    ctx = at(3, 0.2, persons=[(5, BODY_A)], faces=False)
    module.process(ctx)
    assert ctx.metadata["person_identities"][5]["state"] == "checking"
    embedder.vector = unit(1)
    assert run(module, at(4, 0.3, persons=[(5, BODY_A)])).label == "Asha"


# ---------------------------------------------------------- gate counting ---


def test_the_gate_forgets_a_name_recognition_withdrew():
    from computer_vision.modules.analysis.gate import GateCounterModule

    gate = GateCounterModule({"line": [[0.0, 0.5], [1.0, 0.5]]})
    gate.initialize()
    ctx = FrameContext(frame=_FLAT, frame_id=1, source_id="t", monotonic_ts=0.0)
    ctx.metadata["person_identities"] = {5: {"name": "Asha", "state": "recognised"}}
    gate._learn_names(ctx, 0.0)
    assert gate._name_of(5) == "Asha"
    ctx.metadata["person_identities"] = {5: {"name": None, "state": "checking"}}
    gate._learn_names(ctx, 0.1)
    assert gate._name_of(5) is None
