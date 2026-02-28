# Network Security – Phishing Detection System

## Project Overview
This project is an end-to-end Machine Learning system designed to detect phishing websites using supervised classification models.

The system automates:
- Data ingestion from MongoDB
- Data validation & schema checking
- Data drift detection
- Data transformation using preprocessing pipelines
- Model training with hyperparameter tuning
- Experiment tracking with MLflow
- REST API deployment using FastAPI
- Real-time prediction via file upload
- System & model monitoring using Prometheus and Grafana

This project simulates a production-style ML pipeline following MLOps best practices.

## Business Problem
Phishing attacks are one of the most common cybersecurity threats. Detecting malicious websites early helps prevent:
- Financial fraud
- Data breaches
- Credential theft

The goal is to build a machine learning model that classifies a website as:
- 0 → Legitimate
- 1 → Phishing

The system is designed to be scalable, modular, and reliable.

## System Architecture

![](images\project_flow.jpg)
<img src="images\project_flow.jpg" width="850" alt="Project banner">

Pipeline Flow:
Data Ingestion → Data Validation → Data Transformation → 
Model Training → Model Saving → API Deployment → Monitoring

Each stage produces artifacts that are consumed by the next stage.

## Project Workflow

1. Data Ingestion
- Data is fetched from MongoDB
- Converted into Pandas DataFrame
- Stored in feature store
- Split into training and testing datasets

![](images\data_ingestion.jpg)
<img src="images\data_ingestion.jpg" width="600" alt="App screenshot">

2. Data Validation
- Schema validation using YAML
- Column count verification
- Missing value checks
- Data drift detection using Kolmogorov–Smirnov test
- Drift report generated as YAML

![](images\data_validation.jpg)
<img src="images\data_validation.jpg" width="600" alt="App screenshot">

Why this matters:
Ensures data quality before training and prevents garbage-in-garbage-out.

3. Data Transformation
- Target column separation
- Missing values handled using KNN Imputer
- Preprocessing pipeline created using Pipeline
- Transformed NumPy arrays saved
- Preprocessing object saved (preprocessor.pkl)

![](images\data_transformation.jpg)
<img src="images\data_transformation.jpg" width="600" alt="App screenshot">

Design Choice:
Using Pipeline prevents data leakage between training and testing.

4. Model Training
- Models trained:
    - Random Forest
    - Decision Tree
    - Gradient Boosting
    - Logistic Regression
    - AdaBoost
- Hyperparameter tuning using GridSearchCV
- Best model selected
- Metrics computed:
    - F1 Score
    - Precision
    - Recall
- Experiment tracking with MLflow
- Model saved as artifact

![](images\model_trainer.jpg)
<img src="images\model_trainer.jpg" width="600" alt="App screenshot">

Why F1 Score?
Phishing datasets are often imbalanced. F1 balances precision and recall.

5. Experiment Tracking (MLflow)
For each experiment:
- Model name logged
- Hyperparameters logged
- Train & Test metrics logged
- Confusion matrix saved
- Model artifact saved

This enables reproducibility and comparison between models.

6. API Deployment (FastAPI)
- /train -> triggers full training pipeline
- /predict → uploads CSV and returns predictions

Swagger UI:
http://localhost:8000/docs

7. Monitoring & Observability (Prometheus + Grafana)

The system exposes metrics to monitor both system health and model behavior.

System Metrics
- Total prediction requests
- Failed prediction requests
- Error Rate %
- Prediction latency (P95)

Model Metrics
- Count of predicted class 0 (Legitimate)
- Count of predicted class 1 (Phishing)
- Predicted class 1 share %
- Percentage distribution of predictions

Grafana dashboard visualizes these metrics in real time.
This helps detect:
- Service instability
- Increased error rate
- Prediction drift

## Model Evaluation

Metrics Used:
- F1 Score
- Precision
- Recall

Evaluation results stored in ClassificationMetricArtifact for:
- Training data
- Testing data

## Technical Highlights

- Modular pipeline architecture
- Artifact-based communication
- Custom exception handling
- Centralized logging
- Config-driven design
- MLflow experiment tracking
- Prometheus & Grafana monitoring
- FastAPI-based inference service
- Model + Preprocessor separation

## Project Structure

    networksecurity/
    ├── components/
    ├── constant/
    ├── entity/
    ├── exception/
    ├── logging/
    ├── pipeline/
    └── utils/
    app.py
    main.py
    requirements.txt
    monitoring_logs.csv
    README.md
    setup.py
    test_mongodb.py

Artifacts are saved in:
   
    Artifacts/
    ├──data_ingestion/
    ├──data_validation/
    ├──data_transformation/
    ├──model_trainer/

## API Endpoints
    1. Train Model
        GET /train

    Triggers full pipeline execution.

    2. Predict
        POST /predict

    Upload a CSV file → returns predictions in tabular format.

Swagger Documentation available at:
http://localhost:8000/docs

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- FastAPI
- MLflow
- DagsHub
- MongoDB
- Prometheus
- Grafana
- YAML
- Pickle
- Uvicorn

## How to Run Locally

1. Clone repository

        git clone <https://github.com/akashdeepsingh-DS/networksecurity.git>
        cd networksecurity

2. Install dependencies

       pip install -r requirements.txt

3. Set Environment Variable

       Create .env file:
       MONGODB_URL_KEY=your_mongodb_connection_string

4. Run API

       python app.py

Visit:
http://localhost:8000/docs

## Production Design Principles Used

- Separation of concerns
- Config-driven architecture
- Custom error tracing
- Logging and monitoring
- Artifact-based communication
- Reproducibility with timestamped pipelines
- Experiment tracking
- Hyperparameter tuning

## Future Improvements

- Docker containerization
- CI/CD with GitHub Actions
- Cloud deployment (AWS / Azure / Render)
- Model versioning strategy
- Feature importance visualization
- Automated retraining pipeline

## 👨‍💻 Author

Akash Deep Singh
Post-Graduate Diploma in Big Data Analytics and Artificial Intelligence
Georgian College, Canada

Aspiring Data Scientist / ML Engineer specializing in AI.
