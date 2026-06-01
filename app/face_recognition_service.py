import os
import re
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


try:
    from deepface import DeepFace
except Exception:
    class _UnavailableDeepFace:
        @staticmethod
        def find(*args, **kwargs):
            raise RuntimeError("DeepFace is not available for facial recognition.")

        @staticmethod
        def verify(*args, **kwargs):
            raise RuntimeError("DeepFace is not available for facial recognition.")

    DeepFace = _UnavailableDeepFace()


def _env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name, default):
    value = os.getenv(name)
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class FaceRecognitionService:
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(
        self,
        known_faces_dir=None,
        model_name=None,
        distance_metric=None,
        threshold=None,
        enforce_detection=None,
        rebuild_face_cache=None,
        require_face_detection=None,
        face_detector=None,
    ):
        self.known_faces_dir = Path(
            known_faces_dir or os.getenv("KNOWN_FACES_DIR", "data/known_faces")
        )
        self.model_name = model_name or os.getenv("RECOGNITION_MODEL", "Facenet")
        self.distance_metric = distance_metric or os.getenv("DISTANCE_METRIC", "cosine")
        self.threshold = self._configured_threshold(threshold)
        self.enforce_detection = self._configured_bool(
            enforce_detection,
            "ENFORCE_DETECTION",
            False,
        )
        self.rebuild_face_cache = self._configured_bool(
            rebuild_face_cache,
            "REBUILD_FACE_CACHE",
            True,
        )
        self.require_face_detection = self._configured_bool(
            require_face_detection,
            "REQUIRE_FACE_DETECTION",
            True,
        )
        self.face_detector = face_detector
        self._face_cascade = None
        self._cache_rebuilt = False

    def list_known_face_images(self):
        if not self.known_faces_dir.exists():
            return []

        images = [
            path
            for path in self.known_faces_dir.rglob("*")
            if self._is_supported_image(path)
        ]
        return sorted(images, key=lambda path: str(path))

    def identity_for_image(self, image_path):
        path = Path(image_path)

        try:
            relative_path = path.relative_to(self.known_faces_dir)
        except ValueError:
            relative_path = path

        if len(relative_path.parts) > 1:
            return relative_path.parts[0]

        return re.sub(r"_[0-9]+$", "", path.stem) or path.stem

    def recognize(self, frame):
        self._rebuild_cache_once_if_configured()
        known_images = self.list_known_face_images()

        if not known_images:
            return self._unrecognized_result()

        if self.require_face_detection and not self.has_visible_face(frame):
            return self._no_face_detected_result()

        best_match = None
        for known_image in known_images:
            try:
                verification = DeepFace.verify(
                    img1_path=frame,
                    img2_path=str(known_image),
                    model_name=self.model_name,
                    distance_metric=self.distance_metric,
                    enforce_detection=self.enforce_detection,
                    threshold=self.threshold,
                    silent=True,
                )
            except Exception:
                continue

            if not self._is_valid_match(verification):
                continue

            distance = self._distance_from(verification)
            if best_match is None or self._is_better_match(distance, best_match[1]):
                best_match = (known_image, distance)

        if best_match is None:
            return self._unrecognized_result()

        matched_image, distance = best_match
        return self._recognized_result(matched_image, distance)

    def rebuild_cache(self):
        if not self.known_faces_dir.exists():
            return

        for cache_path in self.known_faces_dir.rglob("representations_*.pkl"):
            if cache_path.is_file():
                cache_path.unlink()

    def has_visible_face(self, frame):
        try:
            if self.face_detector is not None:
                detections = self.face_detector(frame) or []
                return len(detections) > 0

            cascade = self._get_face_cascade()
            if cascade is None:
                return False

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
            )
            return len(detections) > 0
        except Exception:
            return False

    def _configured_threshold(self, threshold):
        if threshold is not None:
            return float(threshold)

        return _env_float("RECOGNITION_THRESHOLD", 0.60)

    def _configured_bool(self, configured_value, env_name, default):
        if configured_value is not None:
            return bool(configured_value)

        return _env_bool(env_name, default)

    def _is_supported_image(self, path):
        if not path.is_file():
            return False
        if path.name.startswith("representations_") and path.suffix.lower() == ".pkl":
            return False

        return path.suffix.lower() in self.IMAGE_EXTENSIONS

    def _get_face_cascade(self):
        if self._face_cascade is not None:
            return self._face_cascade

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not cascade_path.exists():
            return None

        cascade = cv2.CascadeClassifier(str(cascade_path))
        if hasattr(cascade, "empty") and cascade.empty():
            return None

        self._face_cascade = cascade
        return self._face_cascade

    def _rebuild_cache_once_if_configured(self):
        if self.rebuild_face_cache and not self._cache_rebuilt:
            self.rebuild_cache()
            self._cache_rebuilt = True

    def _is_valid_match(self, verification):
        if (
            not isinstance(verification, dict)
            or verification.get("verified") is not True
        ):
            return False

        distance = self._distance_from(verification)
        if distance is not None and distance > self.threshold:
            return False

        return True

    def _distance_from(self, verification):
        if not isinstance(verification, dict):
            return None

        distance = verification.get("distance")
        if distance is None:
            return None

        try:
            return float(distance)
        except (TypeError, ValueError):
            return None

    def _is_better_match(self, candidate_distance, current_best_distance):
        if candidate_distance is None:
            return current_best_distance is None
        if current_best_distance is None:
            return True

        return candidate_distance < current_best_distance

    def _recognized_result(self, matched_image, distance):
        identity = self.identity_for_image(matched_image)
        return {
            "recognized": True,
            "identity": identity,
            "name": identity,
            "status": "AUTHORIZED",
            "confidence": self._confidence_from_distance(distance),
            "distance": distance,
            "matched_image": str(matched_image),
        }

    def _unrecognized_result(self):
        return {
            "recognized": False,
            "identity": "unknown",
            "name": "unknown",
            "status": "UNRECOGNIZED",
            "confidence": None,
            "distance": None,
            "matched_image": None,
        }

    def _no_face_detected_result(self):
        return {
            "recognized": False,
            "identity": "none",
            "name": "no_face_detected",
            "status": "NO_FACE_DETECTED",
            "confidence": None,
            "distance": None,
            "matched_image": None,
        }

    def _confidence_from_distance(self, distance):
        if distance is None:
            return None

        return max(0.0, min(1.0, 1.0 - distance))
