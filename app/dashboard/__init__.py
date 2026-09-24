"""Web dashboard.

A read-only view of a running pipeline: live annotated video, which modules
are active, what they are producing, and the events they have raised.

It is deliberately an *output sink* (``WebSink``) plus a small HTTP server,
not a second pipeline. The dashboard observes; it does not drive. That keeps
the Vision Core unaware that a dashboard exists - Phase 1 predicted this
exact shape, and nothing in the pipeline changed to accommodate it.
"""
