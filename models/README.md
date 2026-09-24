# models/

Model **weight files** live here (`.pt`, `.onnx`, `.engine`, ...). They are
git-ignored: weights are downloaded or trained, never committed.

Do not confuse this directory with `app/models/`, which holds the **Python
adapters** that wrap an inference runtime.

Empty in Phase 1 — no AI model is loaded yet.
