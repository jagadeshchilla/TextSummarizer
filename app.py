from fastapi import FastAPI, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.requests import Request
import uvicorn
import os
import sys
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from src.text_summarizer.pipeline.prediction_pipeline import PredictionPipeline

text:str= "What is the Text Summarizer?"

app=FastAPI()

# Setup templates
templates = Jinja2Templates(directory="templates")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/",tags=["authentication"])
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/train")
async def train_route():
    try:
        os.system("python main.py")
        return Response("Training successful !!")
    except Exception as e:
        raise Exception(f"Error: {e}")
    
@app.post("/predict")
async def predict_route(text: str = Form(...)):
    try:
        pipeline=PredictionPipeline()
        summary=pipeline.predict(text)
        return summary
    except Exception as e:
        raise Exception(f"Error: {e}")
    
if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000)