import numpy as np

from app.alert_service import (
    apply_alert_overlay,
    build_alert_text,
    draw_face_overlays,
)


FORBIDDEN_TERMS = [
    "cr" + "iminal",
    "cr" + "ime",
    "su" + "spect " + "cr" + "iminal",
    "dan" + "gerous person",
    "gui" + "lty",
    "wan" + "ted person",
]


def test_builds_visual_text_for_authorized_person():
    text = build_alert_text(
        name="integrante_01",
        status="AUTHORIZED",
        attention_level="LOW",
    )

    assert "AUTHORIZED" in text
    assert "integrante_01" in text
    assert "LOW" in text


def test_builds_visual_text_for_unrecognized_person():
    text = build_alert_text(
        name="unknown",
        status="UNRECOGNIZED",
        attention_level="ATTENTION",
    )

    assert "UNRECOGNIZED" in text
    assert "ATTENTION" in text


def test_builds_visual_text_when_waiting_for_face():
    text = build_alert_text(
        name="no_face_detected",
        status="NO_FACE_DETECTED",
        attention_level="NONE",
    )

    assert "NO_FACE_DETECTED" in text
    assert "waiting for face" in text.lower()
    assert "NONE" in text


def test_apply_alert_overlay_writes_text_on_frame(mocker):
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    put_text = mocker.patch("app.alert_service.cv2.putText", return_value=frame)

    result = apply_alert_overlay(
        frame=frame,
        text="UNRECOGNIZED | Attention: ATTENTION",
    )

    assert result is frame
    put_text.assert_called_once()


def test_draw_face_overlays_draws_rectangle_and_label(mocker):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    rectangle = mocker.patch("app.alert_service.cv2.rectangle", return_value=frame)
    put_text = mocker.patch("app.alert_service.cv2.putText", return_value=frame)

    result = draw_face_overlays(
        frame=frame,
        boxes=[{"x": 10, "y": 12, "w": 30, "h": 32}],
        recognition={"status": "AUTHORIZED", "name": "rayston"},
    )

    assert result is frame
    rectangle.assert_called_once_with(frame, (10, 12), (40, 44), (0, 255, 0), 2)
    put_text.assert_called_once()
    assert "AUTHORIZED" in put_text.call_args.args[1]


def test_draw_face_overlays_shows_unrecognized_label(mocker):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mocker.patch("app.alert_service.cv2.rectangle", return_value=frame)
    put_text = mocker.patch("app.alert_service.cv2.putText", return_value=frame)

    draw_face_overlays(
        frame=frame,
        boxes=[{"x": 10, "y": 12, "w": 30, "h": 32}],
        recognition={"status": "UNRECOGNIZED", "name": "unknown"},
    )

    assert "UNRECOGNIZED" in put_text.call_args.args[1]


def test_alert_text_uses_safe_non_accusatory_language():
    texts = [
        build_alert_text("integrante_01", "AUTHORIZED", "LOW"),
        build_alert_text("unknown", "UNRECOGNIZED", "ATTENTION"),
        build_alert_text("unknown", "UNRECOGNIZED", "ALERT"),
        build_alert_text("no_face_detected", "NO_FACE_DETECTED", "NONE"),
    ]
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_face_overlays(
        frame=frame,
        boxes=[{"x": 10, "y": 12, "w": 30, "h": 32}],
        recognition={"status": "UNRECOGNIZED", "name": "unknown"},
    )

    rendered = " ".join(text.lower() for text in texts)

    assert all(term not in rendered for term in FORBIDDEN_TERMS)
