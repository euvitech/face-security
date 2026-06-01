from pathlib import Path


try:
    from deepface import DeepFace
except Exception:
    class _UnavailableDeepFace:
        @staticmethod
        def find(*args, **kwargs):
            raise RuntimeError("DeepFace is not available for facial recognition.")

    DeepFace = _UnavailableDeepFace()


class FaceRecognitionService:
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(self, known_faces_dir="data/known_faces"):
        self.known_faces_dir = Path(known_faces_dir)

    def recognize(self, frame):
        if not self._has_known_faces():
            return self._unrecognized_result()

        try:
            matches = DeepFace.find(
                img_path=frame,
                db_path=str(self.known_faces_dir),
                enforce_detection=False,
                silent=True,
            )
        except Exception:
            return self._unrecognized_result()

        identity = self._first_identity(matches)
        if identity is None:
            return self._unrecognized_result()

        return {"recognized": True, "name": Path(identity).stem}

    def _has_known_faces(self):
        if not self.known_faces_dir.exists():
            return False

        return any(
            path.is_file() and path.suffix.lower() in self.IMAGE_EXTENSIONS
            for path in self.known_faces_dir.iterdir()
        )

    def _first_identity(self, matches):
        if not matches:
            return None

        first_result = matches[0] if isinstance(matches, list) else matches
        if first_result is None or first_result.empty or "identity" not in first_result:
            return None

        return first_result.iloc[0]["identity"]

    def _unrecognized_result(self):
        return {"recognized": False, "name": "unknown"}
