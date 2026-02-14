import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import shap

class TriageModel:
    def __init__(self):
        self.model = RandomForestClassifier()
        self.le = LabelEncoder()
        self.encoders = {}

    def train(self):
        df = pd.read_csv("triage_data.csv")

        for col in ["Condition","Symptoms","Risk_Level"]:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.encoders[col] = le

        X = df[["Age","Blood_Pressure","Heart_Rate","Temperature","Condition","Symptoms"]]
        y = df["Risk_Level"]

        self.model.fit(X, y)
        self.explainer = shap.TreeExplainer(self.model)

    def predict(self, input_data):
        df = pd.DataFrame([input_data])
        for col in ["Condition","Symptoms"]:
            df[col] = self.encoders[col].transform(df[col])

        prediction = self.model.predict(df)[0]
        confidence = max(self.model.predict_proba(df)[0]) * 100

        shap_values = self.explainer.shap_values(df)

        return {
            "risk": self.encoders["Risk_Level"].inverse_transform([prediction])[0],
            "confidence": round(confidence,2)
        }