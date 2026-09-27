# VisionGuard AI — Development Workflow

How this project is built, and what is left to build. Read alongside
`Roadmap.md` (what order) and `Agent.md` (what rules).

---

## 1. The data flow

Everything in VisionGuard is one of five things, and each has its own lane in
the shared `FrameContext`:

```
  Camera ──▶ FramePacket ──▶ FrameContext
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
   DETECTION               RECOGNITION                TRACKING
   "what is here"          "who is it"              "same one as before"
   context.detections      context.detections       context.tracks
        │                        │                        │
        └────────────────────────┴────────────────────────┘
                                 ▼
                             ANALYSIS
                          "what happened"
                           context.events
                                 │
                                 ▼
                    OutputSink  →  window / dashboard
```

**A module may only read what an earlier module wrote.** No module calls
another module. That single rule is what lets any subset be enabled.

| Layer | Produces | May consume | Runs a model? |
|---|---|---|---|
| Detection | `detections` | frame only | yes |
| Recognition | `detections` | frame + detections | yes |
| Tracking | `tracks` | detections | no |
| Analysis | `events` | detections + tracks | no |

Analysis modules never run models. If a new module needs inference, it is a
detection module and something else consumes it.

---

## 2. Adding a module — the standard loop

Every module so far followed these seven steps. Do not skip step 1 or step 7.

### Step 1 — Decide the layer and the dependency

Ask: *what does it consume?* That fixes the layer, the `requires` tuple and
the execution order. Getting this wrong is expensive later.

> Example: "fall detection" consumes poses, so it is **analysis**, requires
> `pose_detection`, and runs after it. It does **not** get its own model.

### Step 2 — Check the model actually exists

Before writing code, verify the weights are obtainable:

```bash
.venv/Scripts/python.exe -c "from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES; print([n for n in GITHUB_ASSETS_NAMES if 'pose' in n])"
```

Three outcomes, and they change the deliverable:

| Outcome | What to build |
|---|---|
| Stock model exists | Working module, verifiable today |
| Needs custom weights | Working module with `requires_own_model = True`, fails at startup with the URL |
| No model at all | Do not fake it — say so |

**Never ship a module that silently detects nothing.** In a safety system that
is worse than not shipping it.

### Step 3 — Write the module

```python
@register-free: add to BUILTIN_MODULES instead
class ThingModule(BaseAIModule):       # or a shared base
    module_name = "thing"
    module_type = ModuleType.ANALYSIS
    version = "1.0.0"
    requires = ("source_module",)

    def initialize(self): ...          # load, or parse config
    def process(self, ctx): ...        # read ctx, write ctx, return ModuleResult
    def cleanup(self): ...             # release
```

Reuse an existing base where one fits:

- `LabelFilteredDetectionModule` — anything that runs a detector and keeps
  some labels
- `BaseTrackingModule` — anything that assigns ids to a detector's output
- Plain `BaseAIModule` — analysis modules

### Step 4 — Register and configure

1. One line in `BUILTIN_MODULES` (`computer_vision/modules/registry.py`)
2. One config block in `configs/config.yaml`, **`enabled: false`**

New modules ship disabled. The default config must stay runnable on a laptop.

### Step 5 — Test against fakes

Tests must not need a camera, display, model or network. Use the fake backends
in `tests/conftest.py`. Cover:

- the happy path
- every config knob actually taking effect
- the dependency being enforced (`ModuleDependencyError`)
- failure containment (`safe_process` catches it)
- the result payload being JSON-safe

### Step 6 — Verify against reality

Fakes prove logic; they do not prove integration. Run the real model on a real
image and **look at the output**:

```bash
.venv/Scripts/python.exe -m backend.main --no-window --max-frames 60
```

Several real bugs in this project were only visible in a rendered frame — the
zone colour collision, the feet-vs-centre reference point, id-space overlap.

### Step 7 — Update docs and version

`README.md` module table, `app/__init__.py` version, and any measured numbers.
Numbers in the README are measured, never estimated.

---

## 3. Measuring before deciding

Several defaults in this project were chosen by measurement, not intuition,
and three of them contradicted the obvious answer:

| Question | Assumption | Measured | Decision |
|---|---|---|---|
| Windows camera backend | DirectShow is faster | DSHOW 10 fps, MSMF 30.8 fps | MSMF |
| Restrict model to classes? | Saves time | +0.2% (noise) | Filter by label instead |
| MJPG pixel format | Lifts fps cap | No change; hurt DSHOW | Opt-in, not default |

Before hard-coding a performance default, spend two minutes measuring it.

```bash
# per-module cost lands in the final stats line
... | grep module_latency_ms
```

---

## 4. Definition of done

A feature is done when all of these hold:

- [ ] Module registered, configured, **disabled by default**
- [ ] Unit tests pass with no camera/display/model
- [ ] Verified once against the real model or a real stream
- [ ] Dependency enforced and tested
- [ ] Failure contained — a broken module cannot stop the pipeline
- [ ] Per-module latency measured and recorded
- [ ] README updated with measured numbers
- [ ] Nothing in the pipeline, sources or context changed to accommodate it

That last one is the architectural check. If adding a feature forced a change
to `computer_vision/core/pipeline.py`, the abstraction was wrong — fix that first.

---

## 5. Current state

### Done

| Phase | What |
|---|---|
| 1 | Structure, config, logging, context, registry, pipeline |
| 2 | Person detection |
| 3 | Person tracking + `IoUTracker` |
| 4 | Video file source |
| 5 | RTSP source |
| 6 | Object / vehicle / crowd detection, PPE / fire / smoke scaffolding |
| — | Shared inference (2.15× faster with several detectors) |
| 7 | Face detection, recognition, unknown person, identity store |
| 8 | Object and vehicle tracking |
| 9 | Restricted zone |
| 10 | Read-only dashboard |

### In progress

| Feature | Consumes | Blocker |
|---|---|---|
| Pose detection | frame | none — `yolo11n-pose.pt` is a stock model |
| Fall detection | `pose_detection` | none |
| Attendance | `face_recognition` | none |
| Recognition accuracy harness | identity store | needs labelled faces to produce numbers |

### Blocked

| Feature | Blocked on |
|---|---|
| PPE violation | PPE weights (no stock model has helmet/vest) |
| Fire / smoke | Fire and smoke weights |
| Face recognition end-to-end | YuNet + SFace ONNX files |
| Dashboard control | Authentication — no mutation endpoints until then |

### Not started

Loitering, intrusion analysis, cloud (Phase 11), agentic vision (Phase 12).

---

## 6. Accuracy, not just speed

FPS is measured continuously; **accuracy is not measured at all** by the
pipeline, and cannot be — it needs ground truth.

Use `tools/evaluate_recognition.py` against a labelled face directory:

```
data/faces/
  alice/  img01.jpg img02.jpg ...
  bob/    img01.jpg ...
```

It enrols some images per person, tests the rest, and reports accuracy,
precision, recall and a threshold sweep. **Tune `threshold` from that sweep,
not from the default**, because the right value depends on your cameras,
lighting and how many people are enrolled.

Rule: a detection or recognition default may only be changed on the basis of a
measurement over data representative of the deployment.

---

## 7. Privacy checkpoints

Before enabling anything under "Recognition":

- [ ] Is identity processing actually required, or is a person box enough?
- [ ] Who enrolled the identities, and can they be shown consent?
- [ ] Is `computer_vision/Model/face/identities.npz` backed up and access-controlled? It is
      personal data.
- [ ] Is there a process for `store.remove(name)` on request?
- [ ] Is the dashboard on `127.0.0.1`? It streams live camera with no auth.

`Agent.md` §24 is a constraint, not advice: identity processing stays off
unless a camera's configuration explicitly turns it on.
