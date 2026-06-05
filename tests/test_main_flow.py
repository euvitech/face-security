from pathlib import Path

import numpy as np

import main as main_module
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


def test_main_module_does_not_import_deepface_or_recognition_service():
    source = Path(main_module.__file__).read_text(encoding="utf-8")

    assert "deepface" not in source.lower()
    assert "FaceRecognitionService" not in source


def test_main_uses_process_wrapper_without_real_webcam_or_deepface(monkeypatch, mocker):
    raw_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    resized_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    camera = mocker.Mock()
    camera.read_frame.return_value = raw_frame
    camera_class = mocker.patch("main.Camera", return_value=camera)
    recognition_process_class = mocker.patch("main.RecognitionProcess")
    face_detector_class = mocker.patch("main.FaceDetector")
    realtime_flow = mocker.Mock()
    realtime_flow.process_frame.return_value = {"frame": resized_frame}
    realtime_flow_class = mocker.patch(
        "main.RealTimeRecognitionFlow",
        return_value=realtime_flow,
    )
    resize = mocker.patch("main.cv2.resize", return_value=resized_frame)
    imshow = mocker.patch("main.cv2.imshow")
    mocker.patch("main.cv2.waitKey", return_value=ord("q"))
    destroy_windows = mocker.patch("main.cv2.destroyAllWindows")
    monkeypatch.setenv("CAMERA_INDEX", "2")
    monkeypatch.setenv("FRAME_WIDTH", "640")
    monkeypatch.setenv("FRAME_HEIGHT", "480")
    monkeypatch.setenv("SHOW_CAMERA_WINDOW", "true")
    monkeypatch.setenv("RECOGNITION_INTERVAL_SECONDS", "2.5")
    monkeypatch.setenv("UNKNOWN_EVIDENCE_COOLDOWN_SECONDS", "7")
    monkeypatch.setenv("MAX_RECOGNITION_IMAGE_WIDTH", "320")
    monkeypatch.setenv("USE_RECOGNITION_PROCESS", "true")

    main()

    camera_class.assert_called_once_with(index=2)
    recognition_process_class.assert_called_once_with(
        unknown_evidence_cooldown_seconds=7.0,
        save_unknown_evidence=True,
    )
    recognition_process_class.return_value.start.assert_called_once()
    face_detector_class.assert_called_once()
    realtime_flow_class.assert_called_once_with(
        face_detector=face_detector_class.return_value,
        recognition_process=recognition_process_class.return_value,
        recognition_interval_seconds=2.5,
        max_recognition_image_width=320,
    )
    camera.open.assert_called_once()
    resize.assert_called_once_with(raw_frame, (640, 480))
    realtime_flow.process_frame.assert_called_once_with(resized_frame)
    imshow.assert_called_once_with("Face Security", resized_frame)
    recognition_process_class.return_value.shutdown.assert_called_once()
    camera.release.assert_called_once()
    destroy_windows.assert_called_once()


def test_main_reads_frames_continuously_until_max_frames(monkeypatch, mocker):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    camera = mocker.Mock()
    camera.read_frame.return_value = frame
    mocker.patch("main.Camera", return_value=camera)
    mocker.patch("main.RecognitionProcess")
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
    mocker.patch("main.RecognitionProcess")
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
    mocker.patch("main.RecognitionProcess")
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


def test_realtime_flow_detects_faces_every_frame_and_draws_boxes_while_processing(mocker):
    frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(3)]
    face_detector = mocker.Mock()
    face_detector.detect.return_value = [{"x": 1, "y": 2, "w": 3, "h": 4}]
    recognition_process = mocker.Mock()
    recognition_process.is_processing = False
    recognition_process.get_display_result.return_value = {
        "status": "PROCESSING",
        "name": "unknown",
        "identity": "unknown",
        "recognized": False,
    }

    def submit_processing(*args, **kwargs):
        recognition_process.is_processing = True
        return True

    recognition_process.submit.side_effect = submit_processing
    draw = mocker.patch("main.draw_face_overlays", side_effect=lambda frame, **kwargs: frame)

    flow = RealTimeRecognitionFlow(
        face_detector=face_detector,
        recognition_process=recognition_process,
        recognition_interval_seconds=999,
        clock=mocker.Mock(side_effect=[0, 0.1, 0.2]),
    )

    for frame in frames:
        flow.process_frame(frame)

    assert face_detector.detect.call_count == 3
    assert recognition_process.submit.call_count == 1
    assert draw.call_count == 3


def test_realtime_flow_resizes_recognition_image_to_max_width(mocker):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    box = {"x": 0, "y": 0, "w": 80, "h": 40}
    face_detector = mocker.Mock()
    face_detector.detect.return_value = [box]
    recognition_process = mocker.Mock()
    recognition_process.is_processing = False
    recognition_process.submit.return_value = True
    recognition_process.get_display_result.return_value = _unrecognized_result()
    resized_face = np.zeros((20, 40, 3), dtype=np.uint8)
    resize = mocker.patch("main.cv2.resize", return_value=resized_face)
    mocker.patch("main.draw_face_overlays", side_effect=lambda frame, **kwargs: frame)

    flow = RealTimeRecognitionFlow(
        face_detector=face_detector,
        recognition_process=recognition_process,
        recognition_interval_seconds=2.0,
        max_recognition_image_width=40,
        clock=lambda: 0,
    )

    flow.process_frame(frame)

    resize.assert_called_once()
    actual_frame, actual_size = resize.call_args.args
    assert np.array_equal(actual_frame, frame[0:40, 0:80])
    assert actual_size == (40, 20)
    assert recognition_process.submit.call_args.kwargs["frame"] is resized_face


def test_realtime_flow_sends_cropped_frame_copy_to_background_worker(mocker):
    frame = np.arange(10 * 10 * 3, dtype=np.uint8).reshape((10, 10, 3))
    box = {"x": 1, "y": 2, "w": 3, "h": 4}
    face_detector = mocker.Mock()
    face_detector.detect.return_value = [box]
    recognition_process = mocker.Mock()
    recognition_process.is_processing = False
    recognition_process.submit.return_value = True
    recognition_process.get_display_result.return_value = _unrecognized_result()
    mocker.patch("main.draw_face_overlays", side_effect=lambda frame, **kwargs: frame)

    flow = RealTimeRecognitionFlow(
        face_detector=face_detector,
        recognition_process=recognition_process,
        recognition_interval_seconds=2.0,
        clock=lambda: 0,
    )

    flow.process_frame(frame)

    submitted_frame = recognition_process.submit.call_args.kwargs["frame"]
    assert submitted_frame.shape == (4, 3, 3)
    assert np.array_equal(submitted_frame, frame[2:6, 1:4])
    assert not np.shares_memory(submitted_frame, frame)


def test_realtime_flow_starts_recognition_only_when_interval_passed(mocker):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    face_detector = mocker.Mock()
    face_detector.detect.return_value = [{"x": 1, "y": 2, "w": 3, "h": 4}]
    recognition_process = mocker.Mock()
    recognition_process.is_processing = False
    recognition_process.submit.return_value = True
    recognition_process.get_display_result.return_value = _unrecognized_result()
    mocker.patch("main.draw_face_overlays", side_effect=lambda frame, **kwargs: frame)

    flow = RealTimeRecognitionFlow(
        face_detector=face_detector,
        recognition_process=recognition_process,
        recognition_interval_seconds=2.0,
        clock=mocker.Mock(side_effect=[0, 1.0, 2.0, 2.1]),
    )

    for _ in range(4):
        flow.process_frame(frame)

    assert recognition_process.submit.call_count == 2


def test_realtime_flow_skips_recognition_job_if_process_is_busy(mocker):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    face_detector = mocker.Mock()
    face_detector.detect.return_value = [{"x": 1, "y": 2, "w": 3, "h": 4}]
    recognition_process = mocker.Mock()
    recognition_process.is_processing = True
    recognition_process.get_display_result.return_value = {
        "status": "PROCESSING",
        "name": "unknown",
        "identity": "unknown",
    }
    draw = mocker.patch("main.draw_face_overlays", side_effect=lambda frame, **kwargs: frame)

    flow = RealTimeRecognitionFlow(
        face_detector=face_detector,
        recognition_process=recognition_process,
        recognition_interval_seconds=0,
        clock=lambda: 10,
    )

    flow.process_frame(frame)

    recognition_process.submit.assert_not_called()
    draw.assert_called_once()


def test_realtime_flow_does_not_start_recognition_when_no_face_exists(mocker):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    face_detector = mocker.Mock()
    face_detector.detect.return_value = []
    recognition_process = mocker.Mock()
    recognition_process.is_processing = False
    recognition_process.get_display_result.return_value = _recognized_result("rayston")
    mocker.patch("main.draw_face_overlays", side_effect=lambda frame, **kwargs: frame)

    flow = RealTimeRecognitionFlow(
        face_detector=face_detector,
        recognition_process=recognition_process,
        recognition_interval_seconds=2.0,
        clock=lambda: 0,
    )

    result = flow.process_frame(frame)

    recognition_process.submit.assert_not_called()
    assert result["status"] == "NO_FACE_DETECTED"


def test_realtime_flow_keeps_last_result_visible_between_recognition_calls(mocker):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    face_detector = mocker.Mock()
    face_detector.detect.return_value = [{"x": 1, "y": 2, "w": 3, "h": 4}]
    last_result = _recognized_result("rayston")
    recognition_process = mocker.Mock()
    recognition_process.is_processing = False
    recognition_process.submit.return_value = True
    recognition_process.get_display_result.return_value = last_result
    draw = mocker.patch("main.draw_face_overlays", side_effect=lambda frame, **kwargs: frame)

    flow = RealTimeRecognitionFlow(
        face_detector=face_detector,
        recognition_process=recognition_process,
        recognition_interval_seconds=2.0,
        clock=mocker.Mock(side_effect=[0, 0.5]),
    )

    first = flow.process_frame(frame)
    second = flow.process_frame(frame)

    assert recognition_process.submit.call_count == 1
    assert first["status"] == "AUTHORIZED"
    assert second["status"] == "AUTHORIZED"
    assert second["name"] == "rayston"
    assert draw.call_args.kwargs["recognition"] == last_result
