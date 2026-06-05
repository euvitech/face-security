import numpy as np

from app.face_detector import FaceDetector


def test_face_detector_returns_bounding_boxes_with_coordinates(mocker):
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    cascade = mocker.Mock()
    cascade.detectMultiScale.return_value = [(10, 12, 30, 32)]
    cvt_color = mocker.patch(
        "app.face_detector.cv2.cvtColor",
        return_value=np.zeros((80, 80), dtype=np.uint8),
    )

    detector = FaceDetector(cascade=cascade)
    boxes = detector.detect(frame)

    assert boxes == [{"x": 10, "y": 12, "w": 30, "h": 32}]
    cvt_color.assert_called_once_with(frame, detector.grayscale_mode)
    cascade.detectMultiScale.assert_called_once()


def test_face_detector_returns_empty_list_when_no_face_is_found(mocker):
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    cascade = mocker.Mock()
    cascade.detectMultiScale.return_value = []
    mocker.patch(
        "app.face_detector.cv2.cvtColor",
        return_value=np.zeros((80, 80), dtype=np.uint8),
    )

    detector = FaceDetector(cascade=cascade)

    assert detector.detect(frame) == []


def test_face_detector_uses_mocked_opencv_without_webcam(mocker):
    cascade = mocker.Mock()
    cascade.empty.return_value = False
    cascade.detectMultiScale.return_value = []
    cascade_class = mocker.patch(
        "app.face_detector.cv2.CascadeClassifier",
        return_value=cascade,
    )
    mocker.patch("app.face_detector.cv2.data.haarcascades", "/tmp/")
    mocker.patch("app.face_detector.Path.exists", return_value=True)
    mocker.patch(
        "app.face_detector.cv2.cvtColor",
        return_value=np.zeros((80, 80), dtype=np.uint8),
    )

    detector = FaceDetector()
    detector.detect(np.zeros((80, 80, 3), dtype=np.uint8))

    cascade_class.assert_called_once()


def test_face_detector_reuses_initialized_cascade_across_frames(mocker):
    cascade = mocker.Mock()
    cascade.empty.return_value = False
    cascade.detectMultiScale.return_value = []
    cascade_class = mocker.patch(
        "app.face_detector.cv2.CascadeClassifier",
        return_value=cascade,
    )
    mocker.patch("app.face_detector.cv2.data.haarcascades", "/tmp/")
    mocker.patch("app.face_detector.Path.exists", return_value=True)
    mocker.patch(
        "app.face_detector.cv2.cvtColor",
        return_value=np.zeros((80, 80), dtype=np.uint8),
    )
    detector = FaceDetector()

    detector.detect(np.zeros((80, 80, 3), dtype=np.uint8))
    detector.detect(np.zeros((80, 80, 3), dtype=np.uint8))

    cascade_class.assert_called_once()
    assert cascade.detectMultiScale.call_count == 2
