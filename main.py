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
from app.recognition_process import RecognitionProcess
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
        recognition_process,
        recognition_interval_seconds=3.0,
        max_recognition_image_width=320,
        clock=None,
    ):
        self.face_detector = face_detector
        self.recognition_process = recognition_process
        self.recognition_interval_seconds = float(recognition_interval_seconds)
        self.max_recognition_image_width = max(1, int(max_recognition_image_width))
        self.clock = clock or time.monotonic
        self.frame_number = 0
        self.last_recognition_time = None
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

        if self._should_start_recognition(now):
            face_frame = self._copy_face_frame(frame, boxes[0])
            started = self.recognition_process.submit(
                frame=face_frame,
                sensitive_area=sensitive_area,
            )
            if started:
                self.last_recognition_time = now

        display_result = self.recognition_process.get_display_result()
        if display_result.get("status") != "PROCESSING":
            self.last_result = display_result

        frame_with_overlay = draw_face_overlays(
            frame=frame,
            boxes=boxes,
            recognition=display_result,
        )
        return {**display_result, "frame": frame_with_overlay, "boxes": boxes}

    def _should_start_recognition(self, now):
        if self.recognition_process.is_processing:
            return False
        if self.last_recognition_time is None:
            return True

        return now - self.last_recognition_time >= self.recognition_interval_seconds

    def _copy_face_frame(self, frame, box):
        try:
            height, width = frame.shape[:2]
            x = max(0, int(box["x"]))
            y = max(0, int(box["y"]))
            w = max(0, int(box["w"]))
            h = max(0, int(box["h"]))
            x2 = min(width, x + w)
            y2 = min(height, y + h)
            if x2 <= x or y2 <= y:
                face_frame = frame.copy()
            else:
                face_frame = frame[y:y2, x:x2].copy()

            return self._resize_recognition_frame(face_frame)
        except Exception:
            if hasattr(frame, "copy"):
                return self._resize_recognition_frame(frame.copy())

            return frame

    def _resize_recognition_frame(self, frame):
        try:
            height, width = frame.shape[:2]
        except Exception:
            return frame

        if width <= self.max_recognition_image_width:
            return frame

        resized_height = max(1, int(height * (self.max_recognition_image_width / width)))
        return cv2.resize(frame, (self.max_recognition_image_width, resized_height))

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
    frame_width = _env_int("FRAME_WIDTH", 640)
    frame_height = _env_int("FRAME_HEIGHT", 480)
    camera = Camera(index=_env_int("CAMERA_INDEX", 0))
    recognition_process = RecognitionProcess(
        unknown_evidence_cooldown_seconds=_env_float(
            "UNKNOWN_EVIDENCE_COOLDOWN_SECONDS",
            5.0,
        ),
        save_unknown_evidence=_env_bool("SAVE_UNKNOWN_EVIDENCE", True),
    )
    flow = RealTimeRecognitionFlow(
        face_detector=FaceDetector(),
        recognition_process=recognition_process,
        recognition_interval_seconds=_env_float(
            "RECOGNITION_INTERVAL_SECONDS",
            3.0,
        ),
        max_recognition_image_width=_env_int(
            "MAX_RECOGNITION_IMAGE_WIDTH",
            320,
        ),
    )
    frames_processed = 0

    try:
        recognition_process.start()
        camera.open()
        while True:
            frame = camera.read_frame()
            frame = cv2.resize(frame, (frame_width, frame_height))
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
        recognition_process.shutdown()
        camera.release()
        if show_camera_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
