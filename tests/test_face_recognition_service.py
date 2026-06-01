from pathlib import Path

import numpy as np

from app.face_recognition_service import FaceRecognitionService


def _image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")
    return path


def _detected_face(frame):
    return [(0, 0, 10, 10)]


def test_should_list_all_known_face_images_from_known_faces_recursively(tmp_path):
    known_faces_dir = tmp_path / "known_faces"
    direct = _image(known_faces_dir / "rayston_01.jpg")
    nested = _image(known_faces_dir / "ana" / "photo_01.png")
    second_nested = _image(known_faces_dir / "member" / "photo_01.jpeg")

    service = FaceRecognitionService(
        known_faces_dir=known_faces_dir,
        face_detector=_detected_face,
    )

    assert service.list_known_face_images() == sorted(
        [direct, nested, second_nested],
        key=lambda path: str(path),
    )


def test_should_support_images_directly_inside_known_faces(tmp_path):
    known_faces_dir = tmp_path / "known_faces"
    direct = _image(known_faces_dir / "rayston_01.jpg")

    service = FaceRecognitionService(
        known_faces_dir=known_faces_dir,
        face_detector=_detected_face,
    )

    known_images = service.list_known_face_images()

    assert direct in known_images


def test_should_support_subfolders_per_person(tmp_path):
    known_faces_dir = tmp_path / "known_faces"
    nested = _image(known_faces_dir / "rayston" / "photo_01.jpg")

    service = FaceRecognitionService(
        known_faces_dir=known_faces_dir,
        face_detector=_detected_face,
    )

    known_images = service.list_known_face_images()

    assert nested in known_images



def test_should_ignore_non_image_files(tmp_path):
    known_faces_dir = tmp_path / "known_faces"
    valid_image = _image(known_faces_dir / "rayston_01.jpg")
    (known_faces_dir / "notes.txt").write_text("not an image", encoding="utf-8")

    service = FaceRecognitionService(
        known_faces_dir=known_faces_dir,
        face_detector=_detected_face,
    )

    assert service.list_known_face_images() == [valid_image]


def test_should_ignore_representations_pkl_files(tmp_path):
    known_faces_dir = tmp_path / "known_faces"
    valid_image = _image(known_faces_dir / "rayston_01.jpg")
    (known_faces_dir / "representations_facenet.pkl").write_bytes(b"cache")

    service = FaceRecognitionService(
        known_faces_dir=known_faces_dir,
        face_detector=_detected_face,
    )

    assert service.list_known_face_images() == [valid_image]


def test_should_extract_identity_from_filename_for_direct_image(tmp_path):
    known_faces_dir = tmp_path / "known_faces"
    image = _image(known_faces_dir / "rayston_02.jpg")

    service = FaceRecognitionService(
        known_faces_dir=known_faces_dir,
        face_detector=_detected_face,
    )


    assert service.identity_for_image(image) == "rayston"


def test_should_extract_identity_from_parent_folder_for_nested_image(tmp_path):
    known_faces_dir = tmp_path / "known_faces"
    image = _image(known_faces_dir / "ana" / "photo_01.jpg")

    service = FaceRecognitionService(
        known_faces_dir=known_faces_dir,
        face_detector=_detected_face,
    )

    assert service.identity_for_image(image) == "ana"


def test_should_return_unrecognized_when_known_faces_is_empty(tmp_path, mocker):
    known_faces_dir = tmp_path / "known_faces"
    known_faces_dir.mkdir()
    deepface_verify = mocker.patch("app.face_recognition_service.DeepFace.verify")

    service = FaceRecognitionService(
        known_faces_dir=known_faces_dir,
        face_detector=_detected_face,
    )
    result = service.recognize(np.zeros((10, 10, 3), dtype=np.uint8))

    assert result == {
        "recognized": False,
        "identity": "unknown",
        "name": "unknown",
        "status": "UNRECOGNIZED",
        "confidence": None,
        "distance": None,
        "matched_image": None,
    }
    deepface_verify.assert_not_called()


def test_should_call_deepface_verify_with_configured_options(tmp_path, monkeypatch, mocker):
    known_faces_dir = tmp_path / "known_faces"
    known_face = _image(known_faces_dir / "rayston_01.jpg")
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    monkeypatch.setenv("RECOGNITION_MODEL", "ArcFace")
    monkeypatch.setenv("DISTANCE_METRIC", "euclidean_l2")
    monkeypatch.setenv("RECOGNITION_THRESHOLD", "0.42")
    monkeypatch.setenv("ENFORCE_DETECTION", "true")
    deepface_verify = mocker.patch(
        "app.face_recognition_service.DeepFace.verify",
        return_value={"verified": False, "distance": 0.9},
    )

    service = FaceRecognitionService(
        known_faces_dir=known_faces_dir,
        face_detector=_detected_face,
    )
    service.recognize(frame)

    deepface_verify.assert_called_once_with(
        img1_path=frame,
        img2_path=str(known_face),
        model_name="ArcFace",
        distance_metric="euclidean_l2",
        enforce_detection=True,
        threshold=0.42,
        silent=True,
    )


def test_should_return_authorized_when_deepface_returns_valid_match(tmp_path, mocker):
    known_faces_dir = tmp_path / "known_faces"
    known_face = _image(known_faces_dir / "rayston_01.jpg")
    mocker.patch(
        "app.face_recognition_service.DeepFace.verify",
        return_value={"verified": True, "distance": 0.31},
    )

    service = FaceRecognitionService(
        known_faces_dir=known_faces_dir,
        face_detector=_detected_face,
    )
    result = service.recognize(np.zeros((10, 10, 3), dtype=np.uint8))

    assert result["recognized"] is True
    assert result["status"] == "AUTHORIZED"
    assert result["identity"] == "rayston"
    assert result["name"] == "rayston"
    assert result["distance"] == 0.31
    assert result["matched_image"] == str(known_face)


def test_should_return_matched_person_identity_from_nested_folder(tmp_path, mocker):
    known_faces_dir = tmp_path / "known_faces"
    known_face = _image(known_faces_dir / "rayston" / "photo_01.jpg")
    mocker.patch(
        "app.face_recognition_service.DeepFace.verify",
        return_value={"verified": True, "distance": 0.33},
    )

    service = FaceRecognitionService(
        known_faces_dir=known_faces_dir,
        face_detector=_detected_face,
    )
    result = service.recognize(np.zeros((10, 10, 3), dtype=np.uint8))

    assert result["identity"] == "rayston"
    assert result["matched_image"] == str(known_face)


def test_should_return_unrecognized_when_deepface_returns_no_matches(tmp_path, mocker):
    known_faces_dir = tmp_path / "known_faces"
    _image(known_faces_dir / "rayston_01.jpg")
    mocker.patch(
        "app.face_recognition_service.DeepFace.verify",
        return_value={"verified": False, "distance": 0.9},
    )

    service = FaceRecognitionService(
        known_faces_dir=known_faces_dir,
        face_detector=_detected_face,
    )
    result = service.recognize(np.zeros((10, 10, 3), dtype=np.uint8))

    assert result["recognized"] is False
    assert result["identity"] == "unknown"
    assert result["status"] == "UNRECOGNIZED"
    assert result["distance"] is None
    assert result["matched_image"] is None


def test_should_handle_deepface_exceptions_without_crashing(tmp_path, mocker):
    known_faces_dir = tmp_path / "known_faces"
    _image(known_faces_dir / "rayston_01.jpg")
    mocker.patch(
        "app.face_recognition_service.DeepFace.verify",
        side_effect=Exception("deepface failure"),
    )

    service = FaceRecognitionService(
        known_faces_dir=known_faces_dir,
        face_detector=_detected_face,
    )
    result = service.recognize(np.zeros((10, 10, 3), dtype=np.uint8))

    assert result["recognized"] is False
    assert result["identity"] == "unknown"
    assert result["status"] == "UNRECOGNIZED"


def test_should_rebuild_deepface_cache_when_configured(tmp_path, monkeypatch, mocker):
    known_faces_dir = tmp_path / "known_faces"
    _image(known_faces_dir / "rayston_01.jpg")
    cache_file = known_faces_dir / "representations_facenet.pkl"
    cache_file.write_bytes(b"cache")
    monkeypatch.setenv("REBUILD_FACE_CACHE", "true")
    mocker.patch(
        "app.face_recognition_service.DeepFace.verify",
        return_value={"verified": False, "distance": 0.9},
    )

    service = FaceRecognitionService(known_faces_dir=known_faces_dir)
    service.recognize(np.zeros((10, 10, 3), dtype=np.uint8))

    assert not cache_file.exists()


def test_should_not_identify_anyone_when_no_face_is_detected(tmp_path, mocker):
    known_faces_dir = tmp_path / "known_faces"
    _image(known_faces_dir / "Vinicius.jpeg")
    deepface_verify = mocker.patch(
        "app.face_recognition_service.DeepFace.verify",
        return_value={"verified": True, "distance": 0.24},
    )
    service = FaceRecognitionService(
        known_faces_dir=known_faces_dir,
        face_detector=lambda frame: [],
    )

    result = service.recognize(np.zeros((10, 10, 3), dtype=np.uint8))

    assert result == {
        "recognized": False,
        "identity": "none",
        "name": "no_face_detected",
        "status": "NO_FACE_DETECTED",
        "confidence": None,
        "distance": None,
        "matched_image": None,
    }
    deepface_verify.assert_not_called()


def test_should_use_known_faces_dir_from_environment(tmp_path, monkeypatch):
    known_faces_dir = tmp_path / "known_faces"
    monkeypatch.setenv("KNOWN_FACES_DIR", str(known_faces_dir))

    service = FaceRecognitionService()

    assert service.known_faces_dir == known_faces_dir
