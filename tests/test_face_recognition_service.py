from pathlib import Path

import numpy as np
import pandas as pd

from app.face_recognition_service import FaceRecognitionService


def test_recognizes_registered_person(tmp_path, mocker):
    known_face = tmp_path / "known_faces" / "maria.jpg"
    known_face.parent.mkdir()
    known_face.write_bytes(b"image")
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    mocker.patch(
        "app.face_recognition_service.DeepFace.find",
        return_value=[pd.DataFrame({"identity": [str(known_face)]})],
    )

    service = FaceRecognitionService(known_faces_dir=known_face.parent)

    result = service.recognize(frame)

    assert result["recognized"] is True
    assert result["name"] == "maria"


def test_returns_unrecognized_when_there_is_no_match(tmp_path, mocker):
    known_faces_dir = tmp_path / "known_faces"
    known_faces_dir.mkdir()
    (known_faces_dir / "maria.jpg").write_bytes(b"image")
    mocker.patch(
        "app.face_recognition_service.DeepFace.find",
        return_value=[pd.DataFrame()],
    )

    service = FaceRecognitionService(known_faces_dir=known_faces_dir)

    result = service.recognize(np.zeros((10, 10, 3), dtype=np.uint8))

    assert result["recognized"] is False
    assert result["name"] == "unknown"


def test_uses_image_file_name_as_registered_person_identification(tmp_path, mocker):
    known_face = tmp_path / "known_faces" / "integrante_01.jpeg"
    known_face.parent.mkdir()
    known_face.write_bytes(b"image")
    mocker.patch(
        "app.face_recognition_service.DeepFace.find",
        return_value=[pd.DataFrame({"identity": [str(known_face)]})],
    )

    service = FaceRecognitionService(known_faces_dir=known_face.parent)

    result = service.recognize(np.zeros((10, 10, 3), dtype=np.uint8))

    assert result["name"] == "integrante_01"


def test_handles_empty_known_faces_folder(tmp_path, mocker):
    known_faces_dir = tmp_path / "known_faces"
    known_faces_dir.mkdir()
    deepface_find = mocker.patch("app.face_recognition_service.DeepFace.find")

    service = FaceRecognitionService(known_faces_dir=known_faces_dir)

    result = service.recognize(np.zeros((10, 10, 3), dtype=np.uint8))

    assert result["recognized"] is False
    assert result["name"] == "unknown"
    deepface_find.assert_not_called()


def test_handles_deepface_exceptions_without_breaking_system(tmp_path, mocker):
    known_faces_dir = tmp_path / "known_faces"
    known_faces_dir.mkdir()
    (known_faces_dir / "maria.jpg").write_bytes(b"image")
    mocker.patch(
        "app.face_recognition_service.DeepFace.find",
        side_effect=Exception("deepface failure"),
    )

    service = FaceRecognitionService(known_faces_dir=known_faces_dir)

    result = service.recognize(np.zeros((10, 10, 3), dtype=np.uint8))

    assert result["recognized"] is False
    assert result["name"] == "unknown"
