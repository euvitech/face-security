import numpy as np

from main import main, process_frame


def test_main_flow_for_unrecognized_person_saves_evidence_and_logs_event(mocker):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    recognition_service = mocker.Mock()
    recognition_service.recognize.return_value = {
        "recognized": False,
        "name": "unknown",
    }
    event_logger = mocker.Mock()
    event_logger.save_evidence.return_value = "data/unknown_faces/unknown.jpg"
    overlay = mocker.patch("main.apply_alert_overlay", return_value=frame)

    result = process_frame(
        frame=frame,
        recognition_service=recognition_service,
        event_logger=event_logger,
    )

    assert result["status"] == "UNRECOGNIZED"
    assert result["attention_level"] == "ATTENTION"
    assert result["evidence_path"] == "data/unknown_faces/unknown.jpg"
    assert result["frame"] is frame
    overlay.assert_called_once()
    assert "UNRECOGNIZED" in overlay.call_args.args[1]
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


def test_main_starts_camera_loop_with_mocks(mocker):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    camera = mocker.Mock()
    camera.read_frame.return_value = frame
    mocker.patch("main.Camera", return_value=camera)
    mocker.patch("main.FaceRecognitionService")
    mocker.patch("main.EventLogger")
    process_frame_mock = mocker.patch(
        "main.process_frame",
        return_value={"frame": frame},
    )
    imshow = mocker.patch("main.cv2.imshow")
    mocker.patch("main.cv2.waitKey", return_value=ord("q"))
    destroy_windows = mocker.patch("main.cv2.destroyAllWindows")

    main()

    camera.open.assert_called_once()
    process_frame_mock.assert_called_once()
    imshow.assert_called_once_with("Face Security", frame)
    camera.release.assert_called_once()
    destroy_windows.assert_called_once()
