import threading

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


class RecognitionWorker:
    def __init__(
        self,
        recognition_service,
        event_logger,
        unknown_evidence_cooldown_seconds=5.0,
        save_unknown_evidence=True,
        clock=None,
    ):
        self.recognition_service = recognition_service
        self.event_logger = event_logger
        self.unknown_evidence_cooldown_seconds = float(
            unknown_evidence_cooldown_seconds
        )
        self.save_unknown_evidence = save_unknown_evidence
        self.clock = clock
        self.last_result = _initial_result()
        self._is_processing = False
        self._thread = None
        self._lock = threading.Lock()

        if hasattr(self.event_logger, "unknown_evidence_cooldown_seconds"):
            self.event_logger.unknown_evidence_cooldown_seconds = (
                self.unknown_evidence_cooldown_seconds
            )

    @property
    def is_processing(self):
        with self._lock:
            return self._is_processing

    def start(self, frame, sensitive_area=False):
        with self._lock:
            if self._is_processing:
                return False

            frame_for_worker = self._copy_frame(frame)
            self._is_processing = True
            self._thread = threading.Thread(
                target=self._run,
                args=(frame_for_worker, sensitive_area),
                daemon=True,
            )
            self._thread.start()
            return True

    def wait(self, timeout=None):
        thread = None
        with self._lock:
            thread = self._thread

        if thread is not None:
            thread.join(timeout=timeout)

    def get_display_result(self):
        with self._lock:
            if self._is_processing:
                return _processing_result(self.last_result)

            return self.last_result.copy()

    def _run(self, frame, sensitive_area):
        try:
            recognition = self.recognition_service.recognize(frame)
            result = self._result_from_recognition(
                recognition=recognition,
                frame=frame,
                sensitive_area=sensitive_area,
            )
        except Exception as exc:
            result = _error_result(exc)
            self._register_result(result)

        with self._lock:
            self.last_result = result
            self._is_processing = False

    def _result_from_recognition(self, recognition, frame, sensitive_area=False):
        recognition = recognition or {}
        status = recognition.get("status")
        if status == "NO_FACE_DETECTED":
            return _initial_result()

        identity = recognition.get("identity") or recognition.get("name") or "unknown"
        risk = analyze_risk(
            is_recognized=bool(recognition.get("recognized")),
            sensitive_area=sensitive_area,
        )
        evidence_path = None
        if risk["status"] == "UNRECOGNIZED" and self.save_unknown_evidence:
            evidence_path = self._save_unknown_evidence(frame=frame, name=identity)

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
        self._register_result(result)
        return result

    def _save_unknown_evidence(self, frame, name):
        now = self._now()
        if hasattr(self.event_logger, "save_unknown_evidence"):
            return self.event_logger.save_unknown_evidence(
                frame=frame,
                name=name,
                now=now,
            )

        return self.event_logger.save_evidence(
            frame=frame,
            status="UNRECOGNIZED",
            name=name,
        )

    def _register_result(self, result):
        if not hasattr(self.event_logger, "register_event"):
            return

        self.event_logger.register_event(
            name=result.get("name", "unknown"),
            status=result.get("status", "UNRECOGNIZED"),
            attention_level=result.get("attention_level", "ATTENTION"),
            evidence_path=result.get("evidence_path"),
        )

    def _now(self):
        if self.clock is None:
            return None

        return self.clock()

    def _copy_frame(self, frame):
        if hasattr(frame, "copy"):
            return frame.copy()

        return frame
