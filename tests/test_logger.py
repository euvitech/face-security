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


def test_event_logger_does_not_save_unknown_evidence_for_authorized_person(
    tmp_path,
    mocker,
):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    logger = EventLogger(
        logs_dir=tmp_path / "logs",
        evidence_dir=tmp_path / "unknown_faces",
    )
    imwrite = mocker.patch("app.logger.cv2.imwrite")

    evidence_path = logger.save_evidence(
        frame=frame,
        status="AUTHORIZED",
        name="rayston",
    )

    assert evidence_path is None
    imwrite.assert_not_called()


def test_event_logger_logs_recognized_person_event(tmp_path):
    logger = EventLogger(
        logs_dir=tmp_path / "logs",
        evidence_dir=tmp_path / "unknown_faces",
    )

    log_path = logger.register_event(
        name="rayston",
        status="AUTHORIZED",
        attention_level="LOW",
        evidence_path=None,
    )

    event = json.loads(Path(log_path).read_text(encoding="utf-8").splitlines()[-1])

    assert event["name"] == "rayston"
    assert event["status"] == "AUTHORIZED"
    assert event["attention_level"] == "LOW"
    assert event["evidence_path"] is None


def test_event_logger_logs_unrecognized_person_event(tmp_path):
    logger = EventLogger(
        logs_dir=tmp_path / "logs",
        evidence_dir=tmp_path / "unknown_faces",
    )

    log_path = logger.register_event(
        name="unknown",
        status="UNRECOGNIZED",
        attention_level="ATTENTION",
        evidence_path="data/unknown_faces/unknown.jpg",
    )

    event = json.loads(Path(log_path).read_text(encoding="utf-8").splitlines()[-1])

    assert event["name"] == "unknown"
    assert event["status"] == "UNRECOGNIZED"
    assert event["attention_level"] == "ATTENTION"
    assert event["evidence_path"] == "data/unknown_faces/unknown.jpg"


def test_event_logger_uses_log_and_unknown_dirs_from_environment(
    tmp_path,
    monkeypatch,
):
    logs_dir = tmp_path / "configured_logs"
    unknown_faces_dir = tmp_path / "configured_unknown_faces"
    monkeypatch.setenv("LOGS_DIR", str(logs_dir))
    monkeypatch.setenv("UNKNOWN_FACES_DIR", str(unknown_faces_dir))

    logger = EventLogger()

    assert logger.logs_dir == logs_dir
    assert logger.evidence_dir == unknown_faces_dir
