import threading
import time

import numpy as np

from app.recognition_worker import RecognitionWorker


def _recognized_result(name="rayston"):
    return {
        "recognized": True,
        "identity": name,
        "name": name,
        "status": "AUTHORIZED",
        "confidence": 0.7,
        "distance": 0.3,
        "matched_image": f"data/known_faces/{name}_01.jpg",
    }


def _unrecognized_result():
    return {
        "recognized": False,
        "identity": "unknown",
        "name": "unknown",
        "status": "UNRECOGNIZED",
        "confidence": None,
        "distance": None,
        "matched_image": None,
    }


class BlockingRecognitionService:
    def __init__(self, result):
        self.result = result
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = []

    def recognize(self, frame):
        self.calls.append(frame)
        self.started.set()
        assert self.release.wait(timeout=1)
        return self.result


def test_worker_starts_recognition_in_background_and_does_not_block_caller(mocker):
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    recognition_service = BlockingRecognitionService(_recognized_result("rayston"))
    worker = RecognitionWorker(
        recognition_service=recognition_service,
        event_logger=mocker.Mock(),
        clock=lambda: 0,
    )

    started_at = time.monotonic()
    accepted = worker.start(frame)
    elapsed = time.monotonic() - started_at

    assert accepted is True
    assert elapsed < 0.05
    assert recognition_service.started.wait(timeout=0.2)
    assert worker.is_processing is True
    assert worker.get_display_result()["status"] == "PROCESSING"

    recognition_service.release.set()
    worker.wait(timeout=1)

    assert worker.is_processing is False
    assert worker.last_result["status"] == "AUTHORIZED"
    assert worker.last_result["name"] == "rayston"


def test_worker_does_not_start_second_job_while_processing(mocker):
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    recognition_service = BlockingRecognitionService(_unrecognized_result())
    worker = RecognitionWorker(
        recognition_service=recognition_service,
        event_logger=mocker.Mock(),
        clock=lambda: 0,
    )

    assert worker.start(frame) is True
    assert recognition_service.started.wait(timeout=0.2)
    assert worker.start(frame.copy()) is False

    recognition_service.release.set()
    worker.wait(timeout=1)

    assert len(recognition_service.calls) == 1
    assert worker.last_result["status"] == "UNRECOGNIZED"


def test_worker_updates_last_result_and_logs_when_finished(mocker):
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    recognition_service = mocker.Mock()
    recognition_service.recognize.return_value = _recognized_result("ana")
    event_logger = mocker.Mock()
    worker = RecognitionWorker(
        recognition_service=recognition_service,
        event_logger=event_logger,
        clock=lambda: 10,
    )

    assert worker.start(frame) is True
    worker.wait(timeout=1)

    assert worker.last_result["status"] == "AUTHORIZED"
    assert worker.last_result["identity"] == "ana"
    event_logger.save_unknown_evidence.assert_not_called()
    event_logger.register_event.assert_called_once_with(
        name="ana",
        status="AUTHORIZED",
        attention_level="LOW",
        evidence_path=None,
    )


def test_worker_saves_unrecognized_evidence_with_cooldown_service(mocker):
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    recognition_service = mocker.Mock()
    recognition_service.recognize.return_value = _unrecognized_result()
    event_logger = mocker.Mock()
    event_logger.save_unknown_evidence.return_value = "data/unknown_faces/unknown.jpg"
    worker = RecognitionWorker(
        recognition_service=recognition_service,
        event_logger=event_logger,
        clock=lambda: 10,
    )

    worker.start(frame)
    worker.wait(timeout=1)

    assert worker.last_result["status"] == "UNRECOGNIZED"
    assert worker.last_result["evidence_path"] == "data/unknown_faces/unknown.jpg"
    event_logger.save_unknown_evidence.assert_called_once_with(
        frame=recognition_service.recognize.call_args.args[0],
        name="unknown",
        now=10,
    )
    event_logger.register_event.assert_called_once_with(
        name="unknown",
        status="UNRECOGNIZED",
        attention_level="ATTENTION",
        evidence_path="data/unknown_faces/unknown.jpg",
    )


def test_worker_handles_recognition_errors_safely(mocker):
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    recognition_service = mocker.Mock()
    recognition_service.recognize.side_effect = RuntimeError("DeepFace failure")
    worker = RecognitionWorker(
        recognition_service=recognition_service,
        event_logger=mocker.Mock(),
        clock=lambda: 0,
    )

    assert worker.start(frame) is True
    worker.wait(timeout=1)

    assert worker.is_processing is False
    assert worker.last_result["status"] == "UNRECOGNIZED"
    assert worker.last_result["name"] == "unknown"
    assert "error" in worker.last_result
