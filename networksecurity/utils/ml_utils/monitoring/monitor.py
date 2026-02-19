import os
import csv
from datetime import datetime

MONITORING_FILE = "monitoring_logs.csv"

def initialize_monitoring():
    if not os.path.exists(MONITORING_FILE):
        with open(MONITORING_FILE, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "timestamp",
                "event_type",
                "model_name",
                "status"
            ])

def log_prediction_event(model_name, status):
    with open(MONITORING_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now(),
            "prediction",
            model_name,
            status
        ])
