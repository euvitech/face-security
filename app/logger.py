import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

try:
    import cv2
except ImportError:
    class _UnavailableCv2:
        @staticmethod
        def imwrite(path, frame):
            raise RuntimeError("OpenCV is not available to save local evidence image.")

    cv2 = _UnavailableCv2()


class EventLogger:
    def __init__(self, logs_dir="data/logs", evidence_dir="data/unknown_faces"):
        self.logs_dir = Path(logs_dir)
        self.evidence_dir = Path(evidence_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.logs_dir / "events.jsonl"

    def register_event(self, name, status, attention_level, evidence_path=None):
        event = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "name": name,
            "status": status,
            "attention_level": attention_level,
            "evidence_path": evidence_path,
        }

        with self.log_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event) + "\n")

        return self.log_file

    def save_evidence(self, frame, status, name="unknown"):
        if status != "UNRECOGNIZED":
            return None

        safe_name = Path(str(name)).stem or "unknown"
        evidence_path = self.evidence_dir / f"{safe_name}_{uuid4().hex}.jpg"
        saved = cv2.imwrite(str(evidence_path), frame)

        if not saved:
            raise RuntimeError("Could not save local evidence image.")

        return str(evidence_path)
