import runpy
from pathlib import Path

import numpy as np


def test_manual_webcam_script_delegates_to_main(mocker):
    project_root = Path(__file__).resolve().parents[1]
    main_function = mocker.patch("main.main")

    runpy.run_path(
        str(project_root / "scripts" / "test_webcam_manual.py"),
        run_name="__main__",
    )

    main_function.assert_called_once()


def test_manual_recognition_script_captures_one_frame_without_gui(mocker):
    project_root = Path(__file__).resolve().parents[1]
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    camera = mocker.Mock()
    camera.read_frame.return_value = frame
    camera_class = mocker.patch("app.camera.Camera", return_value=camera)
    recognition_service = mocker.Mock()
    recognition_service.recognize.return_value = {
        "recognized": False,
        "identity": "unknown",
        "name": "unknown",
        "status": "UNRECOGNIZED",
        "confidence": None,
        "distance": None,
        "matched_image": None,
    }
    recognition_service_class = mocker.patch(
        "app.face_recognition_service.FaceRecognitionService",
        return_value=recognition_service,
    )
    event_logger = mocker.Mock()
    event_logger.save_evidence.return_value = "data/unknown_faces/unknown.jpg"
    event_logger_class = mocker.patch(
        "app.logger.EventLogger",
        return_value=event_logger,
    )
    imshow = mocker.patch("cv2.imshow", create=True)
    mocker.patch("builtins.print")

    runpy.run_path(
        str(project_root / "scripts" / "test_recognition_manual.py"),
        run_name="__main__",
    )

    camera_class.assert_called_once()
    recognition_service_class.assert_called_once()
    event_logger_class.assert_called_once()
    camera.open.assert_called_once()
    recognition_service.recognize.assert_called_once_with(frame)
    event_logger.save_evidence.assert_called_once_with(
        frame=frame,
        status="UNRECOGNIZED",
        name="unknown",
    )
    event_logger.register_event.assert_called_once_with(
        name="unknown",
        status="UNRECOGNIZED",
        attention_level="ATTENTION",
        evidence_path="data/unknown_faces/unknown.jpg",
    )
    camera.release.assert_called_once()
    imshow.assert_not_called()


def test_manual_recognition_script_does_not_save_or_log_when_no_face(mocker):
    project_root = Path(__file__).resolve().parents[1]
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    camera = mocker.Mock()
    camera.read_frame.return_value = frame
    mocker.patch("app.camera.Camera", return_value=camera)
    recognition_service = mocker.Mock()
    recognition_service.recognize.return_value = {
        "recognized": False,
        "identity": "none",
        "name": "no_face_detected",
        "status": "NO_FACE_DETECTED",
        "confidence": None,
        "distance": None,
        "matched_image": None,
    }
    mocker.patch(
        "app.face_recognition_service.FaceRecognitionService",
        return_value=recognition_service,
    )
    event_logger = mocker.Mock()
    mocker.patch("app.logger.EventLogger", return_value=event_logger)
    mocker.patch("builtins.print")

    runpy.run_path(
        str(project_root / "scripts" / "test_recognition_manual.py"),
        run_name="__main__",
    )

    recognition_service.recognize.assert_called_once_with(frame)
    event_logger.save_evidence.assert_not_called()
    event_logger.register_event.assert_not_called()
    camera.release.assert_called_once()
