import numpy as np
import pytest

from app.camera import Camera, CameraError


def test_camera_opens_webcam_index_zero(mocker):
    capture = mocker.Mock()
    capture.isOpened.return_value = True
    video_capture = mocker.patch("app.camera.cv2.VideoCapture", return_value=capture)

    camera = Camera()
    camera.open()

    video_capture.assert_called_once_with(0)


def test_camera_captures_frame_correctly(mocker):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    capture = mocker.Mock()
    capture.isOpened.return_value = True
    capture.read.return_value = (True, frame)
    mocker.patch("app.camera.cv2.VideoCapture", return_value=capture)

    camera = Camera()
    camera.open()

    assert camera.read_frame() is frame


def test_camera_raises_friendly_error_when_webcam_cannot_be_opened(mocker):
    capture = mocker.Mock()
    capture.isOpened.return_value = False
    mocker.patch("app.camera.cv2.VideoCapture", return_value=capture)

    camera = Camera()

    with pytest.raises(CameraError, match="webcam"):
        camera.open()


def test_camera_releases_resource_when_finished(mocker):
    capture = mocker.Mock()
    capture.isOpened.return_value = True
    mocker.patch("app.camera.cv2.VideoCapture", return_value=capture)

    camera = Camera()
    camera.open()
    camera.release()

    capture.release.assert_called_once()
