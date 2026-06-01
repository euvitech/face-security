import json
from pathlib import Path

import numpy as np

from app.logger import EventLogger


def test_event_logger_creates_logs_folder(tmp_path):
    logs_dir = tmp_path / "logs"

    EventLogger(logs_dir=logs_dir, evidence_dir=tmp_path / "unknown_faces")

    assert logs_dir.exists()
    assert logs_dir.is_dir()


def test_event_logger_registers_event_with_minimum_fields(tmp_path):
    logger = EventLogger(
        logs_dir=tmp_path / "logs",
        evidence_dir=tmp_path / "unknown_faces",
    )

    log_path = logger.register_event(
        name="unknown",
        status="UNRECOGNIZED",
        attention_level="ATTENTION",
        evidence_path="data/unknown_faces/frame.jpg",
    )

    rows = Path(log_path).read_text(encoding="utf-8").strip().splitlines()
    event = json.loads(rows[-1])

    assert event["timestamp"]
    assert event["name"] == "unknown"
    assert event["status"] == "UNRECOGNIZED"
    assert event["attention_level"] == "ATTENTION"
    assert event["evidence_path"] == "data/unknown_faces/frame.jpg"


def test_event_logger_saves_evidence_image_for_unrecognized_person(tmp_path, mocker):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    evidence_dir = tmp_path / "unknown_faces"
    logger = EventLogger(logs_dir=tmp_path / "logs", evidence_dir=evidence_dir)

    def fake_imwrite(path, image):
        Path(path).write_bytes(b"image")
        return True

    imwrite = mocker.patch("app.logger.cv2.imwrite", side_effect=fake_imwrite)

    evidence_path = logger.save_evidence(
        frame=frame,
        status="UNRECOGNIZED",
        name="unknown",
    )

    assert evidence_path is not None
    assert Path(evidence_path).parent == evidence_dir
    assert Path(evidence_path).exists()
    imwrite.assert_called_once()
