"""Cross-camera batching: several cameras' frames, one model call.

With several cameras sharing one model, each camera used to take the model's
lock, run one frame, and release it, so the cameras queued behind each other.
Every call also spends most of its time on the CPU (resizing, building the
tensor, non-max suppression) while the GPU waits - so the GPU sat mostly idle
and every camera got slower as cameras were added.

:class:`BatchCombiner` merges those calls ("flat combining"): a camera that
wants a result queues its frame, and whichever thread gets the model next runs
*every* frame queued so far as one batch and hands each caller its result.
There is no artificial waiting: a lone camera runs immediately, and the batch
grows on its own exactly when cameras would otherwise have been queueing. One
batched forward pass costs far less than the same frames one at a time, and it
keeps the GPU busy.
"""

from __future__ import annotations

import threading
from typing import Callable, Generic, List, Optional, Sequence, TypeVar

import numpy as np

R = TypeVar("R")


class _Request(Generic[R]):
    __slots__ = ("frame", "result", "error", "done")

    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.result: Optional[R] = None
        self.error: Optional[BaseException] = None
        self.done = False


class BatchCombiner(Generic[R]):
    """Run ``run_batch(frames) -> results`` over whatever frames are waiting."""

    def __init__(self, run_batch: Callable[[Sequence[np.ndarray]], List[R]], max_batch: int = 8) -> None:
        self._run_batch = run_batch
        self._max_batch = max(1, int(max_batch))
        self._queue: List[_Request[R]] = []
        self._queue_lock = threading.Lock()
        self._model_lock = threading.Lock()
        #: How many batches ran, and how many frames they held (for stats).
        self.batches = 0
        self.frames = 0

    def submit(self, frame: np.ndarray) -> R:
        request: _Request[R] = _Request(frame)
        with self._queue_lock:
            self._queue.append(request)

        while True:
            with self._model_lock:
                if request.done:
                    break  # another thread already ran our frame in its batch
                with self._queue_lock:
                    batch = self._queue[: self._max_batch]
                    del self._queue[: len(batch)]
                if batch:
                    self._run(batch)
            if request.done:
                break

        if request.error is not None:
            raise request.error
        return request.result  # type: ignore[return-value]

    def _run(self, batch: List[_Request[R]]) -> None:
        try:
            results = self._run_batch([r.frame for r in batch])
            if len(results) != len(batch):
                raise RuntimeError(
                    f"Batched inference returned {len(results)} results for {len(batch)} frames"
                )
        except BaseException as exc:  # noqa: BLE001 - every caller gets the error
            for request in batch:
                request.error, request.done = exc, True
            return
        for request, result in zip(batch, results):
            request.result, request.done = result, True
        self.batches += 1
        self.frames += len(batch)

    @property
    def average_batch(self) -> float:
        return self.frames / self.batches if self.batches else 0.0
