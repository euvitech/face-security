try:
    import cv2
except ImportError:
    class _UnavailableCv2:
        FONT_HERSHEY_SIMPLEX = 0

        @staticmethod
        def putText(*args, **kwargs):
            raise RuntimeError("OpenCV is not available to draw visual alerts.")

    cv2 = _UnavailableCv2()


def build_alert_text(name, status, attention_level):
    display_name = name if status == "AUTHORIZED" else "unrecognized person"
    return f"{status}: {display_name} | Attention: {attention_level}"


def apply_alert_overlay(frame, text):
    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    return frame
