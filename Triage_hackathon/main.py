from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader
import sqlite3
import shutil
import dataset
from model import TriageModel
from chatbot import chatbot_response

app = FastAPI()

dataset.generate_data()
model = TriageModel()
model.train()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Environment(loader=FileSystemLoader("templates"))

@app.get("/", response_class=HTMLResponse)
def login():
    template = templates.get_template("login.html")
    return template.render()

@app.post("/predict")
def predict(age:int=Form(...), bp:int=Form(...), hr:int=Form(...),
            temp:float=Form(...), condition:str=Form(...), symptoms:str=Form(...)):

    result = model.predict({
        "Age":age,
        "Blood_Pressure":bp,
        "Heart_Rate":hr,
        "Temperature":temp,
        "Condition":condition,
        "Symptoms":symptoms
    })

    return result

@app.post("/chat")
def chat(message:str=Form(...), lang:str=Form(...)):
    return {"reply": chatbot_response(message, lang)}

@app.post("/upload")
def upload(file: UploadFile = File(...)):
    with open(f"uploads/{file.filename}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"status":"Uploaded Successfully"}