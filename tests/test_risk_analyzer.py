from app.risk_analyzer import analyze_risk


FORBIDDEN_TERMS = [
    "cr" + "iminal",
    "cr" + "ime",
    "ban" + "dit",
    "record",
    "gui" + "lty",
    "su" + "spect",
    "dan" + "gerous",
    "wan" + "ted",
]


def test_recognized_person_returns_authorized_low_attention():
    result = analyze_risk(is_recognized=True)

    assert result["status"] == "AUTHORIZED"
    assert result["attention_level"] == "LOW"


def test_unrecognized_person_returns_attention():
    result = analyze_risk(is_recognized=False)

    assert result["status"] == "UNRECOGNIZED"
    assert result["attention_level"] == "ATTENTION"


def test_unrecognized_person_in_sensitive_area_returns_alert():
    result = analyze_risk(is_recognized=False, sensitive_area=True)

    assert result["status"] == "UNRECOGNIZED"
    assert result["attention_level"] == "ALERT"


def test_risk_result_never_uses_accusatory_terms():
    results = [
        analyze_risk(is_recognized=True),
        analyze_risk(is_recognized=False),
        analyze_risk(is_recognized=False, sensitive_area=True),
    ]

    for result in results:
        rendered = " ".join(str(value).lower() for value in result.values())
        assert all(term not in rendered for term in FORBIDDEN_TERMS)
