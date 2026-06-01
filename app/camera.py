try:
    import cv2
except ImportError:
    class _UnavailableCv2:
        @staticmethod
        def VideoCapture(index):
            raise RuntimeError("OpenCV is not available to open the webcam.")

    cv2 = _UnavailableCv2()


class CameraError(RuntimeError):
    pass


class Camera:
    def __init__(self, index=0):
        self.index = index
        self.capture = None

    def open(self):
        try:
            self.capture = cv2.VideoCapture(self.index)
        except RuntimeError as exc:
            raise CameraError(str(exc)) from exc

        if not self.capture.isOpened():
            raise CameraError("Could not open webcam. Check the camera connection.")

        return self.capture

    def read_frame(self):
        if self.capture is None:
            raise CameraError("Webcam is not open.")

        success, frame = self.capture.read()
        if not success:
            raise CameraError("Could not capture a frame from the webcam.")

        return frame

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
