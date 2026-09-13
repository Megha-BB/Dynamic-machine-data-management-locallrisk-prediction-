"""Small local risk classifier trained on synthetic machine readings."""
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def _training_data():
    rows = []
    for temperature, pressure, vibration, risk in [
        (55, 90, "Low", "Low Risk"), (62, 98, "Low", "Low Risk"),
        (68, 105, "Medium", "Medium Risk"), (75, 112, "Medium", "Medium Risk"),
        (82, 118, "High", "High Risk"), (92, 130, "High", "High Risk"),
        (48, 85, "Low", "Low Risk"), (72, 108, "High", "Medium Risk"),
        (88, 125, "Medium", "High Risk"), (60, 110, "Low", "Medium Risk"),
    ]:
        rows.append({"temperature": temperature, "pressure": pressure, "vibration": vibration, "risk": risk})
    return rows


_model = None


def _get_model():
    global _model
    if _model is None:
        training = _training_data()
        features = [{k: value for k, value in row.items() if k != "risk"} for row in training]
        labels = [row["risk"] for row in training]
        _model = Pipeline([
            ("encode", DictVectorizer(sparse=False)),
            ("classify", LogisticRegression(max_iter=1000, random_state=7)),
        ])
        _model.fit(features, labels)
    return _model


def predict_risk(values: dict) -> dict:
    """Predict from the original three model features; future fields are ignored."""
    features = {
        "temperature": float(values.get("Temperature", 0)),
        "pressure": float(values.get("Pressure", 0)),
        "vibration": str(values.get("Vibration", "Low")),
    }
    model = _get_model()
    risk = model.predict([features])[0]
    probabilities = model.predict_proba([features])[0]
    confidence = round(float(max(probabilities)) * 100)
    return {"risk_level": risk, "confidence": confidence, "used_fields": ["Temperature", "Pressure", "Vibration"]}
