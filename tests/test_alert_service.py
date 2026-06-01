import numpy as np

from app.alert_service import apply_alert_overlay, build_alert_text


FORBIDDEN_TERMS = [
    "criminal",
    "crime",
    "suspect criminal",
    "dangerous person",
    "guilty",
    "wanted person",
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


def test_apply_alert_overlay_writes_text_on_frame(mocker):
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    put_text = mocker.patch("app.alert_service.cv2.putText", return_value=frame)

    result = apply_alert_overlay(
        frame=frame,
        text="UNRECOGNIZED | Attention: ATTENTION",
    )

    assert result is frame
    put_text.assert_called_once()


def test_alert_text_uses_safe_non_accusatory_language():
    texts = [
        build_alert_text("integrante_01", "AUTHORIZED", "LOW"),
        build_alert_text("unknown", "UNRECOGNIZED", "ATTENTION"),
        build_alert_text("unknown", "UNRECOGNIZED", "ALERT"),
    ]

    rendered = " ".join(text.lower() for text in texts)

    assert all(term not in rendered for term in FORBIDDEN_TERMS)
