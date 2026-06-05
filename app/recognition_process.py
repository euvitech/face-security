import multiprocessing
import os
import queue
from uuid import uuid4

try:
    import cv2
except ImportError:
    class _UnavailableCv2:
        @staticmethod
        def setNumThreads(count):
            return None

    cv2 = _UnavailableCv2()

from app.risk_analyzer import analyze_risk


def _initial_result():
    return {
        "recognized": False,
        "identity": "none",
        "name": "no_face_detected",
        "status": "NO_FACE_DETECTED",
        "attention_level": "NONE",
        "confidence": None,
        "distance": None,
        "matched_image": None,
        "evidence_path": None,
    }


def _processing_result(previous_result):
    return {
        "recognized": previous_result.get("recognized", False),
        "identity": previous_result.get("identity", "unknown"),
        "name": previous_result.get("name", "unknown"),
        "status": "PROCESSING",
        "attention_level": previous_result.get("attention_level", "PROCESSING"),
        "confidence": previous_result.get("confidence"),
        "distance": previous_result.get("distance"),
        "matched_image": previous_result.get("matched_image"),
        "evidence_path": previous_result.get("evidence_path"),
        "previous_result": previous_result.copy(),
    }


def _error_result(exc):
    return {
        "recognized": False,
        "identity": "unknown",
        "name": "unknown",
        "status": "UNRECOGNIZED",
        "attention_level": "ATTENTION",
        "confidence": None,
        "distance": None,
        "matched_image": None,
        "evidence_path": None,
        "error": str(exc),
    }


def configure_worker_runtime():
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    if hasattr(cv2, "setNumThreads"):
        cv2.setNumThreads(1)


def build_recognition_service():
    from app.face_recognition_service import FaceRecognitionService

    return FaceRecognitionService(
        require_face_detection=False,
        rebuild_face_cache=False,
    )


def build_event_logger(unknown_evidence_cooldown_seconds=5.0):
    from app.logger import EventLogger

    return EventLogger(
        logs_dir=os.getenv("LOGS_DIR", "data/logs"),
        evidence_dir=os.getenv("UNKNOWN_FACES_DIR", "data/unknown_faces"),
        unknown_evidence_cooldown_seconds=unknown_evidence_cooldown_seconds,
    )


def recognition_process_loop(
    job_queue,
    result_queue,
    recognition_service_factory=None,
    event_logger_factory=None,
    unknown_evidence_cooldown_seconds=5.0,
    save_unknown_evidence=True,
):
    configure_worker_runtime()
    recognition_service_factory = recognition_service_factory or build_recognition_service
    recognition_service = recognition_service_factory()
    event_logger = None
    if event_logger_factory is not None:
        event_logger = event_logger_factory(unknown_evidence_cooldown_seconds)

    while True:
        message = job_queue.get()
        if message.get("type") == "shutdown":
            break
        if message.get("type") != "recognize":
            continue

        job_id = message.get("job_id")
        frame = message.get("frame")
        sensitive_area = bool(message.get("sensitive_area", False))
        try:
            recognition = recognition_service.recognize(frame)
            result = _result_from_recognition(
                recognition=recognition,
                frame=frame,
                sensitive_area=sensitive_area,
                event_logger=event_logger,
                save_unknown_evidence=save_unknown_evidence,
            )
        except Exception as exc:
            result = _error_result(exc)
            _register_result(event_logger, result)

        result_queue.put({"job_id": job_id, "result": result})


def _result_from_recognition(
    recognition,
    frame,
    sensitive_area=False,
    event_logger=None,
    save_unknown_evidence=True,
):
    recognition = recognition or {}
    if recognition.get("status") == "NO_FACE_DETECTED":
        return _initial_result()

    identity = recognition.get("identity") or recognition.get("name") or "unknown"
    risk = analyze_risk(
        is_recognized=bool(recognition.get("recognized")),
        sensitive_area=sensitive_area,
    )
    evidence_path = None
    if risk["status"] == "UNRECOGNIZED" and save_unknown_evidence:
        evidence_path = _save_unknown_evidence(
            event_logger=event_logger,
            frame=frame,
            name=identity,
        )

    result = {
        "recognized": bool(recognition.get("recognized")),
        "identity": identity,
        "name": identity,
        "status": risk["status"],
        "attention_level": risk["attention_level"],
        "confidence": recognition.get("confidence"),
        "distance": recognition.get("distance"),
        "matched_image": recognition.get("matched_image"),
        "evidence_path": evidence_path,
    }
    _register_result(event_logger, result)
    return result


def _save_unknown_evidence(event_logger, frame, name):
    if event_logger is None:
        return None
    if hasattr(event_logger, "save_unknown_evidence"):
        return event_logger.save_unknown_evidence(frame=frame, name=name)

    return event_logger.save_evidence(
        frame=frame,
        status="UNRECOGNIZED",
        name=name,
    )


def _register_result(event_logger, result):
    if event_logger is None or not hasattr(event_logger, "register_event"):
        return

    event_logger.register_event(
        name=result.get("name", "unknown"),
        status=result.get("status", "UNRECOGNIZED"),
        attention_level=result.get("attention_level", "ATTENTION"),
        evidence_path=result.get("evidence_path"),
    )


class RecognitionProcess:
    def __init__(
        self,
        unknown_evidence_cooldown_seconds=5.0,
        save_unknown_evidence=True,
        context=None,
        recognition_service_factory=None,
        event_logger_factory=None,
    ):
        self.unknown_evidence_cooldown_seconds = float(
            unknown_evidence_cooldown_seconds
        )
        self.save_unknown_evidence = save_unknown_evidence
        self.context = context or multiprocessing.get_context("spawn")
        self.recognition_service_factory = (
            recognition_service_factory or build_recognition_service
        )
        self.event_logger_factory = event_logger_factory or build_event_logger
        self.job_queue = self.context.Queue(maxsize=1)
        self.result_queue = self.context.Queue()
        self.process = None
        self.current_job_id = None
        self.last_result = _initial_result()

    @property
    def is_processing(self):
        return self.current_job_id is not None

    def start(self):
        if self.process is not None and self.process.is_alive():
            return

        self.process = self.context.Process(
            target=recognition_process_loop,
            args=(
                self.job_queue,
                self.result_queue,
                self.recognition_service_factory,
                self.event_logger_factory,
                self.unknown_evidence_cooldown_seconds,
                self.save_unknown_evidence,
            ),
            daemon=True,
        )
        self.process.start()

    def submit(self, frame, sensitive_area=False):
        self.poll_result()
        if self.is_processing:
            return False

        self.start()
        job_id = uuid4().hex
        self.current_job_id = job_id
        self.job_queue.put(
            {
                "type": "recognize",
                "job_id": job_id,
                "frame": self._copy_frame(frame),
                "sensitive_area": sensitive_area,
            }
        )
        return True

    def poll_result(self):
        try:
            while True:
                message = self.result_queue.get_nowait()
                if message.get("job_id") != self.current_job_id:
                    continue

                self.last_result = message.get("result") or _error_result(
                    RuntimeError("empty recognition result")
                )
                self.current_job_id = None
                return self.last_result.copy()
        except queue.Empty:
            return None

    def get_display_result(self):
        self.poll_result()
        if self.is_processing:
            return _processing_result(self.last_result)

        return self.last_result.copy()

    def shutdown(self, timeout=1.0):
        if self.process is None:
            return

        try:
            self.job_queue.put({"type": "shutdown"})
        except Exception:
            pass

        self.process.join(timeout=timeout)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=timeout)

    def _copy_frame(self, frame):
        if hasattr(frame, "copy"):
            return frame.copy()

        return frame
