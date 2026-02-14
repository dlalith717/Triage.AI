import pandas as pd
import numpy as np
import random

def generate_data(n=1000):
    data = []
    for i in range(n):
        age = random.randint(1, 90)
        bp = random.randint(90, 180)
        hr = random.randint(60, 140)
        temp = round(random.uniform(97, 104), 1)

        conditions = random.choice(["None", "Diabetes", "Hypertension", "Heart Disease"])
        symptoms = random.choice(["Chest Pain", "Fever", "Headache", "Breathing Issue"])

        risk = "Low"

        if bp > 160 or hr > 120 or temp > 102:
            risk = "High"
        elif bp > 140 or temp > 100:
            risk = "Medium"

        dept = "General Medicine"
        if symptoms == "Chest Pain":
            dept = "Cardiology"
        if symptoms == "Breathing Issue":
            dept = "Emergency"
        if symptoms == "Headache":
            dept = "Neurology"

        data.append([age, bp, hr, temp, conditions, symptoms, risk, dept])

    df = pd.DataFrame(data, columns=[
        "Age","Blood_Pressure","Heart_Rate","Temperature",
        "Condition","Symptoms","Risk_Level","Department"
    ])

    df.to_csv("triage_data.csv", index=False)
    return df