import sys
import os

import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv("MONGODB_URL_KEY")
print(mongo_db_url)

import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File,UploadFile,Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

from networksecurity.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from networksecurity.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME
from networksecurity.utils.ml_utils.monitoring.monitor import initialize_monitoring,log_prediction_event

from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import PlainTextResponse
import time


database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
initialize_monitoring()
# Prometheus Metrics
# ----------------------------

PREDICTION_COUNTER = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)

FAILED_PREDICTION_COUNTER = Counter(
    "prediction_failures_total",
    "Total number of failed predictions"
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time taken for prediction"
)

PRED_CLASS_0_COUNTER = Counter(
    "predicted_class_0_total",
    "Total number of predictions for class 0"
)

PRED_CLASS_1_COUNTER = Counter(
    "predicted_class_1_total",
    "Total number of predictions for class 1"
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline=TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
@app.post("/predict")
async def predict_route(request:Request, file:UploadFile=File(...)):
    start_time = time.time()
    PREDICTION_COUNTER.inc()
    try:
        df=pd.read_csv(file.file)
        preprocessor=load_object("final_model/preprocessor.pkl")
        final_model=load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor,model=final_model)
        print(df.iloc[0])
        y_pred = network_model.predict(df)
        print(y_pred)
        for pred in list(y_pred):
            if int(pred) == 0:
                PRED_CLASS_0_COUNTER.inc()
            elif int(pred) == 1:
                PRED_CLASS_1_COUNTER.inc()
                
        duration = time.time() - start_time
        PREDICTION_LATENCY.observe(duration)

        df['predicted_column'] = y_pred
        print(df['predicted_column'])

        model_name = type(final_model).__name__
        for _ in range(len(y_pred)):
            log_prediction_event(model_name=model_name, status="success")

        df.to_csv("prediction_output/output.csv")
        table_html=df.to_html(classes='table-striped')
        return templates.TemplateResponse("table.html", {"request": request, "table":table_html})
    
    except Exception as e:
        # Log failure
        log_prediction_event(model_name="Unknown", status="failure")
        FAILED_PREDICTION_COUNTER.inc()
        raise NetworkSecurityException(e,sys)
    
@app.get("/monitor")
async def monitoring_dashboard():
    try:
        df = pd.read_csv("monitoring_logs.csv")

        total_predictions = len(df)
        successful = len(df[df["status"] == "success"])
        failed = len(df[df["status"] == "failure"])

        error_rate = (failed / total_predictions) * 100 if total_predictions > 0 else 0

        return {
            "total_predictions": total_predictions,
            "successful_predictions": successful,
            "failed_predictions": failed,
            "error_rate_percent": round(error_rate, 2)
        }

    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest())


    
if __name__=="__main__":
    app_run(app,host="localhost", port=8000)