from pathlib import Path


try:
    import cv2
except ImportError:
    class _UnavailableCv2:
        COLOR_BGR2GRAY = 0

        class data:
            haarcascades = ""

        @staticmethod
        def cvtColor(*args, **kwargs):
            raise RuntimeError("OpenCV is not available for face detection.")

        @staticmethod
        def CascadeClassifier(*args, **kwargs):
            raise RuntimeError("OpenCV is not available for face detection.")

    cv2 = _UnavailableCv2()


class FaceDetector:
    def __init__(self, cascade=None):
        self.grayscale_mode = cv2.COLOR_BGR2GRAY
        self.cascade = cascade

    def detect(self, frame):
        cascade = self._get_cascade()
        if cascade is None:
            return []

        try:
            gray_frame = cv2.cvtColor(frame, self.grayscale_mode)
            detections = cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
            )
        except Exception:
            return []

        return [
            {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
            for x, y, w, h in detections
        ]

    def _get_cascade(self):
        if self.cascade is not None:
            return self.cascade

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not cascade_path.exists():
            return None

        cascade = cv2.CascadeClassifier(str(cascade_path))
        if hasattr(cascade, "empty") and cascade.empty():
            return None

        self.cascade = cascade
        return self.cascade
