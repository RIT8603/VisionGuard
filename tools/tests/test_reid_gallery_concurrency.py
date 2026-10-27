"""A shared gallery must not lose one camera's update during another's sync."""
import multiprocessing
import time

import numpy as np

from computer_vision.modules.recognition.person_reid import Gallery, PersonReIdModule


def _sync_camera(path, gate, index):
    original = Gallery.load
    def slow_read(self):
        original(self)
        time.sleep(0.3)  # widen the real read/merge/write race between worker processes
    Gallery.load = slow_read
    module = PersonReIdModule({})
    module._gallery = Gallery(path)
    module._gallery.enrol(np.eye(2, dtype=np.float32)[index], time.time(), f'camera-{index}')
    module._sync_seconds, module._last_sync = 0.001, 0
    gate.wait(timeout=15)
    module._sync(time.time(), force=True)


def test_two_camera_processes_keep_both_gallery_updates(tmp_path):
    context = multiprocessing.get_context('spawn')
    gate = context.Barrier(2)
    path = str(tmp_path / 'shared-gallery.npz')
    workers = [context.Process(target=_sync_camera, args=(path, gate, n)) for n in range(2)]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=25)
            assert worker.exitcode == 0
        gallery = Gallery(path)
        gallery.load()
        assert len(gallery.identities) == 2, 'A concurrent camera update was lost'
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
