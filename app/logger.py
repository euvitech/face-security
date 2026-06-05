import json
import os
import time
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


def _env_float(name, default):
    value = os.getenv(name)
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class EventLogger:
    def __init__(
        self,
        logs_dir=None,
        evidence_dir=None,
        unknown_evidence_cooldown_seconds=None,
        clock=None,
    ):
        self.logs_dir = Path(logs_dir or os.getenv("LOGS_DIR", "data/logs"))
        self.evidence_dir = Path(
            evidence_dir or os.getenv("UNKNOWN_FACES_DIR", "data/unknown_faces")
        )
        self.unknown_evidence_cooldown_seconds = (
            float(unknown_evidence_cooldown_seconds)
            if unknown_evidence_cooldown_seconds is not None
            else _env_float("UNKNOWN_EVIDENCE_COOLDOWN_SECONDS", 5.0)
        )
        self.clock = clock or time.monotonic
        self.last_unknown_evidence_time = None
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

    def save_unknown_evidence(self, frame, name="unknown", now=None):
        now = self.clock() if now is None else float(now)
        if not self._should_save_unknown_evidence(now):
            return None

        evidence_path = self.save_evidence(
            frame=frame,
            status="UNRECOGNIZED",
            name=name,
        )
        self.last_unknown_evidence_time = now
        return evidence_path

    def _should_save_unknown_evidence(self, now):
        if self.last_unknown_evidence_time is None:
            return True

        return (
            now - self.last_unknown_evidence_time
            >= self.unknown_evidence_cooldown_seconds
        )

    def save_evidence(self, frame, status, name="unknown"):
        if status != "UNRECOGNIZED":
            return None

        safe_name = Path(str(name)).stem or "unknown"
        evidence_path = self.evidence_dir / f"{safe_name}_{uuid4().hex}.jpg"
        saved = cv2.imwrite(str(evidence_path), frame)

        if not saved:
            raise RuntimeError("Could not save local evidence image.")

        return str(evidence_path)
