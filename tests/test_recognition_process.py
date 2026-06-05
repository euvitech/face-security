import queue

import numpy as np

from app.recognition_process import RecognitionProcess, recognition_process_loop


class ImmediateRecognitionService:
    def __init__(self):
        self.calls = []

    def recognize(self, frame):
        self.calls.append(frame)
        return {
            "recognized": True,
            "identity": "rayston",
            "name": "rayston",
            "status": "AUTHORIZED",
            "confidence": 0.8,
            "distance": 0.2,
            "matched_image": "data/known_faces/rayston_01.jpg",
        }


class FailingRecognitionService:
    def recognize(self, frame):
        raise RuntimeError("recognition failure")


class FakeQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)

    def get(self, timeout=None):
        if not self.items:
            raise queue.Empty
        return self.items.pop(0)

    def get_nowait(self):
        return self.get(timeout=0)


class FakeProcess:
    def __init__(self, target=None, args=(), daemon=False):
        self.target = target
        self.args = args
        self.daemon = daemon
        self.started = False
        self.terminated = False
        self.joined = False
        self._alive = False

    def start(self):
        self.started = True
        self._alive = True

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.terminated = True
        self._alive = False

    def join(self, timeout=None):
        self.joined = True


class FakeContext:
    def __init__(self):
        self.queues = []
        self.processes = []

    def Queue(self, maxsize=0):
        new_queue = FakeQueue()
        self.queues.append(new_queue)
        return new_queue

    def Process(self, target=None, args=(), daemon=False):
        process = FakeProcess(target=target, args=args, daemon=daemon)
        self.processes.append(process)
        return process


def test_recognition_process_loop_sends_job_result_through_queue():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    job_queue = FakeQueue()
    result_queue = FakeQueue()
    job_queue.put({"type": "recognize", "job_id": "job-1", "frame": frame})
    job_queue.put({"type": "shutdown"})

    recognition_process_loop(
        job_queue=job_queue,
        result_queue=result_queue,
        recognition_service_factory=ImmediateRecognitionService,
    )

    result = result_queue.get_nowait()
    assert result["job_id"] == "job-1"
    assert result["result"]["status"] == "AUTHORIZED"
    assert result["result"]["name"] == "rayston"


def test_recognition_process_loop_handles_errors_safely():
    job_queue = FakeQueue()
    result_queue = FakeQueue()
    job_queue.put({
        "type": "recognize",
        "job_id": "job-1",
        "frame": np.zeros((20, 20, 3), dtype=np.uint8),
    })
    job_queue.put({"type": "shutdown"})

    recognition_process_loop(
        job_queue=job_queue,
        result_queue=result_queue,
        recognition_service_factory=FailingRecognitionService,
    )

    result = result_queue.get_nowait()
    assert result["job_id"] == "job-1"
    assert result["result"]["status"] == "UNRECOGNIZED"
    assert "error" in result["result"]


def test_recognition_process_wrapper_starts_process_and_sends_job():
    context = FakeContext()
    worker = RecognitionProcess(context=context)

    worker.start()
    accepted = worker.submit(
        frame=np.zeros((20, 20, 3), dtype=np.uint8),
        sensitive_area=False,
    )

    assert context.processes[0].started is True
    assert accepted is True
    assert worker.is_processing is True
    assert context.queues[0].items[0]["type"] == "recognize"


def test_recognition_process_wrapper_does_not_accept_second_job_while_busy():
    context = FakeContext()
    worker = RecognitionProcess(context=context)
    worker.start()

    first = worker.submit(np.zeros((20, 20, 3), dtype=np.uint8))
    second = worker.submit(np.zeros((20, 20, 3), dtype=np.uint8))

    assert first is True
    assert second is False
    assert len(context.queues[0].items) == 1


def test_recognition_process_wrapper_returns_result_and_clears_busy():
    context = FakeContext()
    worker = RecognitionProcess(context=context)
    worker.start()
    worker.submit(np.zeros((20, 20, 3), dtype=np.uint8))
    context.queues[1].put({
        "job_id": worker.current_job_id,
        "result": {"status": "AUTHORIZED", "name": "rayston"},
    })

    result = worker.poll_result()

    assert result == {"status": "AUTHORIZED", "name": "rayston"}
    assert worker.is_processing is False


def test_recognition_process_wrapper_shuts_down_cleanly():
    context = FakeContext()
    worker = RecognitionProcess(context=context)
    worker.start()

    worker.shutdown()

    assert context.queues[0].items[-1] == {"type": "shutdown"}
    assert context.processes[0].joined is True


def test_recognition_process_loop_limits_tensorflow_and_opencv_threads(monkeypatch, mocker):
    monkeypatch.delenv("TF_NUM_INTRAOP_THREADS", raising=False)
    monkeypatch.delenv("TF_NUM_INTEROP_THREADS", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    set_num_threads = mocker.patch("app.recognition_process.cv2.setNumThreads")
    job_queue = FakeQueue()
    result_queue = FakeQueue()
    job_queue.put({"type": "shutdown"})

    recognition_process_loop(
        job_queue=job_queue,
        result_queue=result_queue,
        recognition_service_factory=ImmediateRecognitionService,
    )

    assert set_num_threads.call_args.args == (1,)
    assert __import__("os").environ["TF_NUM_INTRAOP_THREADS"] == "1"
    assert __import__("os").environ["TF_NUM_INTEROP_THREADS"] == "1"
    assert __import__("os").environ["OMP_NUM_THREADS"] == "1"
