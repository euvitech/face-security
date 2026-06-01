def analyze_risk(is_recognized, sensitive_area=False):
    if is_recognized:
        return {"status": "AUTHORIZED", "attention_level": "LOW"}

    attention_level = "ALERT" if sensitive_area else "ATTENTION"
    return {"status": "UNRECOGNIZED", "attention_level": attention_level}
