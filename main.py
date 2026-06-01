try:
    import cv2
except ImportError:
    class _UnavailableCv2:
        @staticmethod
        def imshow(*args, **kwargs):
            raise RuntimeError("OpenCV is not available to display the webcam.")

        @staticmethod
        def waitKey(delay):
            return ord("q")

        @staticmethod
        def destroyAllWindows():
            return None

    cv2 = _UnavailableCv2()

from app.alert_service import apply_alert_overlay, build_alert_text
from app.camera import Camera, CameraError
from app.face_recognition_service import FaceRecognitionService
from app.logger import EventLogger
from app.risk_analyzer import analyze_risk


def process_frame(frame, recognition_service, event_logger, sensitive_area=False):
    recognition = recognition_service.recognize(frame)
    risk = analyze_risk(
        is_recognized=recognition["recognized"],
        sensitive_area=sensitive_area,
    )
    alert_text = build_alert_text(
        name=recognition["name"],
        status=risk["status"],
        attention_level=risk["attention_level"],
    )
    frame_with_alert = apply_alert_overlay(frame, alert_text)
    evidence_path = event_logger.save_evidence(
        frame=frame,
        status=risk["status"],
        name=recognition["name"],
    )
    event_logger.register_event(
        name=recognition["name"],
        status=risk["status"],
        attention_level=risk["attention_level"],
        evidence_path=evidence_path,
    )

    return {
        "frame": frame_with_alert,
        "name": recognition["name"],
        "status": risk["status"],
        "attention_level": risk["attention_level"],
        "evidence_path": evidence_path,
    }


def main():
    camera = Camera(index=0)
    recognition_service = FaceRecognitionService()
    event_logger = EventLogger()

    try:
        camera.open()
        while True:
            frame = camera.read_frame()
            result = process_frame(frame, recognition_service, event_logger)
            cv2.imshow("Face Security", result["frame"])

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except CameraError as exc:
        print(f"Face Security could not start the webcam: {exc}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
