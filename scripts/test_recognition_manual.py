import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

__test__ = False

from app.camera import Camera, CameraError
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


def run_manual_recognition():
    camera = Camera(index=_env_int("CAMERA_INDEX", 0))
    recognition_service = FaceRecognitionService()
    event_logger = EventLogger(
        logs_dir=os.getenv("LOGS_DIR", "data/logs"),
        evidence_dir=os.getenv("UNKNOWN_FACES_DIR", "data/unknown_faces"),
    )

    try:
        camera.open()
        frame = camera.read_frame()
        recognition = recognition_service.recognize(frame)
        identity = recognition.get("identity") or recognition.get("name", "unknown")
        if recognition.get("status") == "NO_FACE_DETECTED":
            result = {
                **recognition,
                "identity": identity,
                "name": recognition.get("name", "no_face_detected"),
                "status": "NO_FACE_DETECTED",
                "attention_level": "NONE",
                "evidence_path": None,
            }
            print(json.dumps(result, indent=2, sort_keys=True, default=str))
            return result

        risk = analyze_risk(is_recognized=recognition["recognized"])

        evidence_path = None
        if risk["status"] == "UNRECOGNIZED" and _env_bool(
            "SAVE_UNKNOWN_EVIDENCE",
            True,
        ):
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

        result = {
            **recognition,
            "identity": identity,
            "name": identity,
            "status": risk["status"],
            "attention_level": risk["attention_level"],
            "evidence_path": evidence_path,
        }
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return result
    finally:
        camera.release()


def main():
    try:
        return run_manual_recognition()
    except CameraError as exc:
        print(f"Face Security could not capture a webcam frame: {exc}")
        return None


if __name__ == "__main__":
    main()
