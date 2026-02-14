import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/symptoms_departments.csv")

X = df["symptom_text"]

# Model 1 → Department
y_dept = df["department"]

dept_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=200))
])

dept_model.fit(X, y_dept)

joblib.dump(dept_model, "department_model.pkl")


# Model 2 → Risk Level
y_risk = df["risk_level"]

risk_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=200))
])

risk_model.fit(X, y_risk)

joblib.dump(risk_model, "risk_model.pkl")

print("✅ Department and Risk models trained & saved.")


  