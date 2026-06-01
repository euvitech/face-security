try:
    import cv2
except ImportError:
    class _UnavailableCv2:
        FONT_HERSHEY_SIMPLEX = 0

        @staticmethod
        def rectangle(*args, **kwargs):
            raise RuntimeError("OpenCV is not available to draw face boxes.")

        @staticmethod
        def putText(*args, **kwargs):
            raise RuntimeError("OpenCV is not available to draw visual alerts.")

    cv2 = _UnavailableCv2()


def build_alert_text(name, status, attention_level):
    if status == "NO_FACE_DETECTED":
        return f"{status}: waiting for face | Attention: {attention_level}"

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


def draw_face_overlays(frame, boxes, recognition=None):
    recognition = recognition or {}
    status = recognition.get("status", "NO_FACE_DETECTED")
    name = recognition.get("name") or recognition.get("identity") or "unknown"
    label = _label_for(status, name)
    color = _color_for(status)

    for box in boxes:
        x = int(box["x"])
        y = int(box["y"])
        w = int(box["w"])
        h = int(box["h"])
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame,
            label,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    if not boxes:
        apply_alert_overlay(
            frame,
            build_alert_text("no_face_detected", "NO_FACE_DETECTED", "NONE"),
        )

    return frame


def _label_for(status, name):
    if status == "AUTHORIZED":
        return f"AUTHORIZED: {name}"
    if status == "UNRECOGNIZED":
        return "UNRECOGNIZED"

    return "NO_FACE_DETECTED"


def _color_for(status):
    if status == "AUTHORIZED":
        return (0, 255, 0)
    if status == "UNRECOGNIZED":
        return (0, 255, 255)

    return (180, 180, 180)
