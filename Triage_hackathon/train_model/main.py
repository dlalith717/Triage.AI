from flask import Flask, request, render_template_string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

app = Flask(__name__)

dept_model = joblib.load("department_model.pkl")
risk_model = joblib.load("risk_model.pkl")

HTML = """
<h2>AI Triage System</h2>
<form method="post">
  <input type="text" name="symptom" placeholder="Enter symptoms" required>
  <input type="submit" value="Check">
</form>

{% if dept %}
<h3>Department: {{ dept }}</h3>
<h3>Risk Level: {{ risk }}</h3>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def home():
    dept = None
    risk = None

    if request.method == "POST":
        symptom = request.form["symptom"]

        dept = dept_model.predict([symptom])[0]
        risk = risk_model.predict([symptom])[0]

        if any(word in symptom.lower() for word in 
               ["not breathing", "heavy bleeding", "unconscious", "cannot breathe", "collapsed"]):
            dept = "Emergency"
            risk = "High"

    return render_template_string(HTML, dept=dept, risk=risk)

if __name__ == "__main__":
    app.run(debug=True)

("tfidf", TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1,2)
))
symptom = request.form["symptom"].lower()

# Emergency override
if any(word in symptom for word in 
       ["not breathing", "heavy bleeding", "unconscious", "collapsed"]):
    dept = "Emergency"
    risk = "High"
    dept_conf = 100
    risk_conf = 100
else:
    dept_probs = dept_model.predict_proba([symptom])[0]
    risk_probs = risk_model.predict_proba([symptom])[0]

    dept_index = dept_probs.argmax()
    risk_index = risk_probs.argmax()

    dept = dept_model.classes_[dept_index]
    risk = risk_model.classes_[risk_index]

    dept_conf = round(dept_probs[dept_index] * 100, 2)
    risk_conf = round(risk_probs[risk_index] * 100, 2)

