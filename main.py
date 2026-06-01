import os
import time

try:
    import cv2
except ImportError:
    class _UnavailableCv2:
        @staticmethod
        def imshow(*args, **kwargs):
            raise RuntimeError("OpenCV is not available to display the webcam.")

        @staticmethod
        def waitKey(delay):
            return ord("q")

        @staticmethod
        def destroyAllWindows():
            return None

    cv2 = _UnavailableCv2()

from app.alert_service import apply_alert_overlay, build_alert_text, draw_face_overlays
from app.camera import Camera, CameraError
from app.face_detector import FaceDetector
from app.face_recognition_service import FaceRecognitionService
from app.logger import EventLogger
from app.risk_analyzer import analyze_risk


def _env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    value = os.getenv(name)
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _no_face_result():
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


class RealTimeRecognitionFlow:
    def __init__(
        self,
        face_detector,
        recognition_service,
        event_logger,
        recognition_interval_frames=30,
        recognition_interval_seconds=1.5,
        unknown_evidence_cooldown_seconds=5,
        save_unknown_evidence=True,
        clock=None,
    ):
        self.face_detector = face_detector
        self.recognition_service = recognition_service
        self.event_logger = event_logger
        self.recognition_interval_frames = max(1, int(recognition_interval_frames))
        self.recognition_interval_seconds = float(recognition_interval_seconds)
        self.unknown_evidence_cooldown_seconds = float(
            unknown_evidence_cooldown_seconds
        )
        self.save_unknown_evidence = save_unknown_evidence
        self.clock = clock or time.monotonic
        self.frame_number = 0
        self.last_recognition_frame = None
        self.last_recognition_time = None
        self.last_unknown_evidence_time = None
        self.last_result = _no_face_result()

    def process_frame(self, frame, sensitive_area=False):
        now = self.clock()
        self.frame_number += 1
        boxes = self.face_detector.detect(frame)

        if not boxes:
            self.last_result = _no_face_result()
            frame_with_overlay = draw_face_overlays(
                frame=frame,
                boxes=boxes,
                recognition=self.last_result,
            )
            return {**self.last_result, "frame": frame_with_overlay, "boxes": boxes}

        if self._should_run_recognition(now):
            recognition = self.recognition_service.recognize(frame)
            self.last_result = self._result_from_recognition(
                recognition=recognition,
                frame=frame,
                now=now,
                sensitive_area=sensitive_area,
            )
            self.last_recognition_frame = self.frame_number
            self.last_recognition_time = now

        frame_with_overlay = draw_face_overlays(
            frame=frame,
            boxes=boxes,
            recognition=self.last_result,
        )
        return {**self.last_result, "frame": frame_with_overlay, "boxes": boxes}

    def _should_run_recognition(self, now):
        if self.last_recognition_frame is None or self.last_recognition_time is None:
            return True

        frames_since = self.frame_number - self.last_recognition_frame
        seconds_since = now - self.last_recognition_time
        return (
            frames_since >= self.recognition_interval_frames
            or seconds_since >= self.recognition_interval_seconds
        )

    def _result_from_recognition(self, recognition, frame, now, sensitive_area=False):
        if recognition.get("status") == "NO_FACE_DETECTED":
            return _no_face_result()

        identity = recognition.get("identity") or recognition.get("name", "unknown")
        risk = analyze_risk(
            is_recognized=recognition["recognized"],
            sensitive_area=sensitive_area,
        )
        evidence_path = None
        if risk["status"] == "UNRECOGNIZED" and self._should_save_unknown(now):
            evidence_path = self.event_logger.save_evidence(
                frame=frame,
                status=risk["status"],
                name=identity,
            )
            self.last_unknown_evidence_time = now

        self.event_logger.register_event(
            name=identity,
            status=risk["status"],
            attention_level=risk["attention_level"],
            evidence_path=evidence_path,
        )
        return {
            "identity": identity,
            "name": identity,
            "recognized": recognition["recognized"],
            "status": risk["status"],
            "attention_level": risk["attention_level"],
            "evidence_path": evidence_path,
            "confidence": recognition.get("confidence"),
            "distance": recognition.get("distance"),
            "matched_image": recognition.get("matched_image"),
        }

    def _should_save_unknown(self, now):
        if not self.save_unknown_evidence:
            return False
        if self.last_unknown_evidence_time is None:
            return True

        return (
            now - self.last_unknown_evidence_time
            >= self.unknown_evidence_cooldown_seconds
        )


def process_frame(
    frame,
    recognition_service,
    event_logger,
    sensitive_area=False,
    save_unknown_evidence=None,
):
    recognition = recognition_service.recognize(frame)
    identity = recognition.get("identity") or recognition.get("name", "unknown")
    if recognition.get("status") == "NO_FACE_DETECTED":
        attention_level = "NONE"
        alert_text = build_alert_text(
            name=identity,
            status="NO_FACE_DETECTED",
            attention_level=attention_level,
        )
        frame_with_alert = apply_alert_overlay(frame, alert_text)
        return {
            "frame": frame_with_alert,
            "identity": identity,
            "name": recognition.get("name", "no_face_detected"),
            "status": "NO_FACE_DETECTED",
            "attention_level": attention_level,
            "evidence_path": None,
            "confidence": recognition.get("confidence"),
            "distance": recognition.get("distance"),
            "matched_image": recognition.get("matched_image"),
        }

    risk = analyze_risk(
        is_recognized=recognition["recognized"],
        sensitive_area=sensitive_area,
    )
    alert_text = build_alert_text(
        name=identity,
        status=risk["status"],
        attention_level=risk["attention_level"],
    )
    frame_with_alert = apply_alert_overlay(frame, alert_text)
    should_save_unknown = _env_bool("SAVE_UNKNOWN_EVIDENCE", True)
    if save_unknown_evidence is not None:
        should_save_unknown = save_unknown_evidence

    evidence_path = None
    if risk["status"] == "UNRECOGNIZED" and should_save_unknown:
        evidence_path = event_logger.save_evidence(
            frame=frame,
            status=risk["status"],
            name=identity,
        )

    event_logger.register_event(
        name=identity,
        status=risk["status"],
        attention_level=risk["attention_level"],
        evidence_path=evidence_path,
    )

    return {
        "frame": frame_with_alert,
        "identity": identity,
        "name": identity,
        "status": risk["status"],
        "attention_level": risk["attention_level"],
        "evidence_path": evidence_path,
        "confidence": recognition.get("confidence"),
        "distance": recognition.get("distance"),
        "matched_image": recognition.get("matched_image"),
    }


def main(max_frames=None):
    show_camera_window = _env_bool("SHOW_CAMERA_WINDOW", True)
    camera = Camera(index=_env_int("CAMERA_INDEX", 0))
    recognition_service = FaceRecognitionService(require_face_detection=False)
    event_logger = EventLogger(
        logs_dir=os.getenv("LOGS_DIR", "data/logs"),
        evidence_dir=os.getenv("UNKNOWN_FACES_DIR", "data/unknown_faces"),
    )
    flow = RealTimeRecognitionFlow(
        face_detector=FaceDetector(),
        recognition_service=recognition_service,
        event_logger=event_logger,
        recognition_interval_frames=_env_int("RECOGNITION_INTERVAL_FRAMES", 30),
        recognition_interval_seconds=_env_float(
            "RECOGNITION_INTERVAL_SECONDS",
            1.5,
        ),
        unknown_evidence_cooldown_seconds=_env_float(
            "UNKNOWN_EVIDENCE_COOLDOWN_SECONDS",
            5,
        ),
        save_unknown_evidence=_env_bool("SAVE_UNKNOWN_EVIDENCE", True),
    )
    frames_processed = 0

    try:
        camera.open()
        while True:
            frame = camera.read_frame()
            result = flow.process_frame(frame)
            frames_processed += 1

            if show_camera_window:
                cv2.imshow("Face Security", result["frame"])

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                print(
                    f"{result['status']}: {result['name']} | "
                    f"Attention: {result['attention_level']}"
                )

            if max_frames is not None and frames_processed >= max_frames:
                break
    except CameraError as exc:
        print(f"Face Security could not start the webcam: {exc}")
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        if show_camera_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
