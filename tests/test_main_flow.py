import numpy as np

from main import RealTimeRecognitionFlow, main, process_frame


def _recognized_result(name="rayston"):
    return {
        "recognized": True,
        "identity": name,
        "name": name,
        "status": "AUTHORIZED",
        "confidence": 0.7,
        "distance": 0.3,
        "matched_image": f"data/known_faces/{name}_01.jpg",
    }


def _unrecognized_result():
    return {
        "recognized": False,
        "identity": "unknown",
        "name": "unknown",
        "status": "UNRECOGNIZED",
        "confidence": None,
        "distance": None,
        "matched_image": None,
    }


def test_process_frame_returns_authorized_low_for_known_match_without_evidence(mocker):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    recognition_service = mocker.Mock()
    recognition_service.recognize.return_value = _recognized_result("rayston")
    event_logger = mocker.Mock()
    overlay = mocker.patch("main.apply_alert_overlay", return_value=frame)

    result = process_frame(
        frame=frame,
        recognition_service=recognition_service,
        event_logger=event_logger,
    )

    assert result["status"] == "AUTHORIZED"
    assert result["attention_level"] == "LOW"
    assert result["identity"] == "rayston"
    assert result["evidence_path"] is None
    overlay.assert_called_once()
    event_logger.save_evidence.assert_not_called()
    event_logger.register_event.assert_called_once_with(
        name="rayston",
        status="AUTHORIZED",
        attention_level="LOW",
        evidence_path=None,
    )


def test_process_frame_returns_unrecognized_attention_and_saves_evidence(mocker):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    recognition_service = mocker.Mock()
    recognition_service.recognize.return_value = _unrecognized_result()
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
    overlay.assert_called_once()
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


def test_process_frame_shows_waiting_state_when_no_face_is_detected(mocker):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
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
    event_logger = mocker.Mock()
    overlay = mocker.patch("main.apply_alert_overlay", return_value=frame)

    result = process_frame(
        frame=frame,
        recognition_service=recognition_service,
        event_logger=event_logger,
    )

    assert result["status"] == "NO_FACE_DETECTED"
    assert result["attention_level"] == "NONE"
    assert result["evidence_path"] is None
    assert "NO_FACE_DETECTED" in overlay.call_args.args[1]
    event_logger.save_evidence.assert_not_called()
    event_logger.register_event.assert_not_called()


def test_process_frame_does_not_save_unknown_evidence_when_disabled(mocker):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    recognition_service = mocker.Mock()
    recognition_service.recognize.return_value = _unrecognized_result()
    event_logger = mocker.Mock()
    mocker.patch("main.apply_alert_overlay", return_value=frame)

    result = process_frame(
        frame=frame,
        recognition_service=recognition_service,
        event_logger=event_logger,
        save_unknown_evidence=False,
    )

    assert result["status"] == "UNRECOGNIZED"
    assert result["evidence_path"] is None
    event_logger.save_evidence.assert_not_called()


def test_main_uses_mocks_without_real_webcam_or_deepface(monkeypatch, mocker):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    camera = mocker.Mock()
    camera.read_frame.return_value = frame
    camera_class = mocker.patch("main.Camera", return_value=camera)
    recognition_service_class = mocker.patch("main.FaceRecognitionService")
    event_logger_class = mocker.patch("main.EventLogger")
    face_detector_class = mocker.patch("main.FaceDetector")
    realtime_flow = mocker.Mock()
    realtime_flow.process_frame.return_value = {"frame": frame}
    realtime_flow_class = mocker.patch(
        "main.RealTimeRecognitionFlow",
        return_value=realtime_flow,
    )
    imshow = mocker.patch("main.cv2.imshow")
    mocker.patch("main.cv2.waitKey", return_value=ord("q"))
    destroy_windows = mocker.patch("main.cv2.destroyAllWindows")
    monkeypatch.setenv("CAMERA_INDEX", "2")
    monkeypatch.setenv("SHOW_CAMERA_WINDOW", "true")
    monkeypatch.setenv("RECOGNITION_INTERVAL_FRAMES", "12")
    monkeypatch.setenv("RECOGNITION_INTERVAL_SECONDS", "2.5")
    monkeypatch.setenv("UNKNOWN_EVIDENCE_COOLDOWN_SECONDS", "7")

    main()

    camera_class.assert_called_once_with(index=2)
    recognition_service_class.assert_called_once_with(require_face_detection=False)
    event_logger_class.assert_called_once()
    face_detector_class.assert_called_once()
    realtime_flow_class.assert_called_once_with(
        face_detector=face_detector_class.return_value,
        recognition_service=recognition_service_class.return_value,
        event_logger=event_logger_class.return_value,
        recognition_interval_frames=12,
        recognition_interval_seconds=2.5,
        unknown_evidence_cooldown_seconds=7.0,
        save_unknown_evidence=True,
    )
    camera.open.assert_called_once()
    realtime_flow.process_frame.assert_called_once_with(frame)
    imshow.assert_called_once_with("Face Security", frame)
    camera.release.assert_called_once()
    destroy_windows.assert_called_once()


def test_main_reads_frames_continuously_until_max_frames(monkeypatch, mocker):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    camera = mocker.Mock()
    camera.read_frame.return_value = frame
    mocker.patch("main.Camera", return_value=camera)
    mocker.patch("main.FaceRecognitionService")
    mocker.patch("main.EventLogger")
    mocker.patch("main.FaceDetector")
    realtime_flow = mocker.Mock()
    realtime_flow.process_frame.return_value = {
        "frame": frame,
        "status": "AUTHORIZED",
        "name": "rayston",
        "attention_level": "LOW",
    }
    mocker.patch("main.RealTimeRecognitionFlow", return_value=realtime_flow)
    monkeypatch.setenv("SHOW_CAMERA_WINDOW", "false")

    main(max_frames=3)

    assert camera.read_frame.call_count == 3
    assert realtime_flow.process_frame.call_count == 3


def test_main_can_run_without_camera_window(monkeypatch, mocker):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    camera = mocker.Mock()
    camera.read_frame.return_value = frame
    mocker.patch("main.Camera", return_value=camera)
    mocker.patch("main.FaceRecognitionService")
    mocker.patch("main.EventLogger")
    mocker.patch("main.FaceDetector")
    realtime_flow = mocker.Mock()
    realtime_flow.process_frame.return_value = {
        "frame": frame,
        "status": "AUTHORIZED",
        "name": "rayston",
        "attention_level": "LOW",
    }
    mocker.patch("main.RealTimeRecognitionFlow", return_value=realtime_flow)
    imshow = mocker.patch("main.cv2.imshow")
    wait_key = mocker.patch("main.cv2.waitKey")
    destroy_windows = mocker.patch("main.cv2.destroyAllWindows")
    monkeypatch.setenv("SHOW_CAMERA_WINDOW", "false")

    main(max_frames=1)

    imshow.assert_not_called()
    wait_key.assert_not_called()
    destroy_windows.assert_not_called()
    camera.release.assert_called_once()


def test_main_shows_camera_window_by_default(monkeypatch, mocker):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    camera = mocker.Mock()
    camera.read_frame.return_value = frame
    mocker.patch("main.Camera", return_value=camera)
    mocker.patch("main.FaceRecognitionService")
    mocker.patch("main.EventLogger")
    mocker.patch("main.FaceDetector")
    realtime_flow = mocker.Mock()
    realtime_flow.process_frame.return_value = {"frame": frame}
    mocker.patch("main.RealTimeRecognitionFlow", return_value=realtime_flow)
    imshow = mocker.patch("main.cv2.imshow")
    mocker.patch("main.cv2.waitKey", return_value=ord("q"))
    destroy_windows = mocker.patch("main.cv2.destroyAllWindows")
    monkeypatch.delenv("SHOW_CAMERA_WINDOW", raising=False)

    main()

    imshow.assert_called_once_with("Face Security", frame)
    destroy_windows.assert_called_once()


def test_realtime_flow_detects_faces_every_frame_but_does_not_recognize_every_frame(mocker):
    frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(3)]
    face_detector = mocker.Mock()
    face_detector.detect.return_value = [{"x": 1, "y": 2, "w": 3, "h": 4}]
    recognition_service = mocker.Mock()
    recognition_service.recognize.return_value = _recognized_result("rayston")
    draw = mocker.patch("main.draw_face_overlays", side_effect=lambda frame, **kwargs: frame)

    flow = RealTimeRecognitionFlow(
        face_detector=face_detector,
        recognition_service=recognition_service,
        event_logger=mocker.Mock(),
        recognition_interval_frames=30,
        recognition_interval_seconds=999,
        clock=mocker.Mock(side_effect=[0, 0.1, 0.2]),
    )

    for frame in frames:
        flow.process_frame(frame)

    assert face_detector.detect.call_count == 3
    assert recognition_service.recognize.call_count == 1
    assert draw.call_count == 3


def test_realtime_flow_calls_recognition_by_frame_interval(mocker):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    face_detector = mocker.Mock()
    face_detector.detect.return_value = [{"x": 1, "y": 2, "w": 3, "h": 4}]
    recognition_service = mocker.Mock()
    recognition_service.recognize.return_value = _recognized_result("rayston")
    mocker.patch("main.draw_face_overlays", side_effect=lambda frame, **kwargs: frame)

    flow = RealTimeRecognitionFlow(
        face_detector=face_detector,
        recognition_service=recognition_service,
        event_logger=mocker.Mock(),
        recognition_interval_frames=2,
        recognition_interval_seconds=999,
        clock=mocker.Mock(side_effect=[0, 0.1, 0.2]),
    )

    flow.process_frame(frame)
    flow.process_frame(frame)
    flow.process_frame(frame)

    assert recognition_service.recognize.call_count == 2


def test_realtime_flow_calls_recognition_by_seconds_interval(mocker):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    face_detector = mocker.Mock()
    face_detector.detect.return_value = [{"x": 1, "y": 2, "w": 3, "h": 4}]
    recognition_service = mocker.Mock()
    recognition_service.recognize.return_value = _recognized_result("rayston")
    mocker.patch("main.draw_face_overlays", side_effect=lambda frame, **kwargs: frame)

    flow = RealTimeRecognitionFlow(
        face_detector=face_detector,
        recognition_service=recognition_service,
        event_logger=mocker.Mock(),
        recognition_interval_frames=999,
        recognition_interval_seconds=1.5,
        clock=mocker.Mock(side_effect=[0, 1.4, 1.6]),
    )

    flow.process_frame(frame)
    flow.process_frame(frame)
    flow.process_frame(frame)

    assert recognition_service.recognize.call_count == 2


def test_realtime_flow_keeps_last_result_between_recognition_calls(mocker):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    face_detector = mocker.Mock()
    face_detector.detect.return_value = [{"x": 1, "y": 2, "w": 3, "h": 4}]
    recognition_service = mocker.Mock()
    recognition_service.recognize.return_value = _recognized_result("rayston")
    draw = mocker.patch("main.draw_face_overlays", side_effect=lambda frame, **kwargs: frame)

    flow = RealTimeRecognitionFlow(
        face_detector=face_detector,
        recognition_service=recognition_service,
        event_logger=mocker.Mock(),
        recognition_interval_frames=30,
        recognition_interval_seconds=999,
        clock=mocker.Mock(side_effect=[0, 0.1]),
    )

    first = flow.process_frame(frame)
    second = flow.process_frame(frame)

    assert first["status"] == "AUTHORIZED"
    assert second["status"] == "AUTHORIZED"
    assert recognition_service.recognize.call_count == 1
    assert draw.call_args.kwargs["recognition"]["name"] == "rayston"


def test_realtime_flow_saves_unknown_evidence_only_with_cooldown(mocker):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    face_detector = mocker.Mock()
    face_detector.detect.return_value = [{"x": 1, "y": 2, "w": 3, "h": 4}]
    recognition_service = mocker.Mock()
    recognition_service.recognize.return_value = _unrecognized_result()
    event_logger = mocker.Mock()
    event_logger.save_evidence.return_value = "data/unknown_faces/unknown.jpg"
    mocker.patch("main.draw_face_overlays", side_effect=lambda frame, **kwargs: frame)

    flow = RealTimeRecognitionFlow(
        face_detector=face_detector,
        recognition_service=recognition_service,
        event_logger=event_logger,
        recognition_interval_frames=1,
        recognition_interval_seconds=999,
        unknown_evidence_cooldown_seconds=5,
        clock=mocker.Mock(side_effect=[0, 1, 6]),
    )

    flow.process_frame(frame)
    flow.process_frame(frame)
    flow.process_frame(frame)

    assert recognition_service.recognize.call_count == 3
    assert event_logger.save_evidence.call_count == 2
    assert event_logger.register_event.call_count == 3
