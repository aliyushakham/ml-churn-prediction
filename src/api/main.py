"""
FastAPI приложение для предсказаний оттока клиентов
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import yaml
from pathlib import Path
import sys
import uvicorn
from typing import Dict, Any, Optional
from src.api.templates.custom_docs import get_custom_docs_html

root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from src.data.preprocess import DataPreprocessor
from src.features.build_features import FeatureEngineer
from src.utils.logger import app_logger

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API для предсказания оттока клиентов (Churn Prediction)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("config/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

model = None
preprocessor = None
feature_engineer = None
feature_names = None

class CustomerData(BaseModel):
    """Модель входных данных для предсказания"""
    gender: str = Field(..., description="Пол (Male/Female)")
    SeniorCitizen: int = Field(..., description="Пенсионер (0/1)")
    Partner: str = Field(..., description="Наличие партнера (Yes/No)")
    Dependents: str = Field(..., description="Наличие иждивенцев (Yes/No)")
    tenure: int = Field(..., description="Длительность обслуживания (месяцы)")
    PhoneService: str = Field(..., description="Услуга телефонии (Yes/No)")
    MultipleLines: str = Field(..., description="Несколько линий (Yes/No/No phone service)")
    InternetService: str = Field(..., description="Тип интернета (DSL/Fiber optic/No)")
    OnlineSecurity: str = Field(..., description="Онлайн безопасность (Yes/No/No internet service)")
    OnlineBackup: str = Field(..., description="Онлайн бэкап (Yes/No/No internet service)")
    DeviceProtection: str = Field(..., description="Защита устройств (Yes/No/No internet service)")
    TechSupport: str = Field(..., description="Техподдержка (Yes/No/No internet service)")
    StreamingTV: str = Field(..., description="Стриминг ТВ (Yes/No/No internet service)")
    StreamingMovies: str = Field(..., description="Стриминг кино (Yes/No/No internet service)")
    Contract: str = Field(..., description="Тип контракта (Month-to-month/One year/Two year)")
    PaperlessBilling: str = Field(..., description="Электронный счет (Yes/No)")
    PaymentMethod: str = Field(..., description="Способ оплаты")
    MonthlyCharges: float = Field(..., description="Ежемесячные платежи")
    TotalCharges: float = Field(..., description="Всего заплачено")

    class Config:
        schema_extra = {
            "example": {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.5,
                "TotalCharges": 845.0
            }
        }

class PredictionResponse(BaseModel):
    """Модель ответа"""
    churn_probability: float = Field(..., description="Вероятность оттока (0-1)")
    prediction: str = Field(..., description="Предсказание (Yes/No)")
    confidence: str = Field(..., description="Уверенность предсказания")
    risk_factors: Optional[Dict[str, float]] = Field(None, description="Факторы риска")

@app.on_event("startup")
async def load_models():
    """Загрузка модели и препроцессора при старте"""
    global model, preprocessor, feature_engineer, feature_names
    
    app_logger.info("Загрузка модели и препроцессора...")
    
    try:
        model_path = config['api']['model_path']
        if not Path(model_path).exists():
            app_logger.error(f"Модель не найдена по пути {model_path}")
            return
        
        model = joblib.load(model_path)
        app_logger.info("Модель загружена успешно")
        
        preprocessor = DataPreprocessor()
        preprocessor.load_preprocessor("artifacts/models/preprocessor.joblib")
        app_logger.info("Препроцессор загружен успешно")
        
        feature_engineer = FeatureEngineer()
        
        metadata_path = "artifacts/models/model_metadata.json"
        if Path(metadata_path).exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                feature_names = metadata.get('feature_names')
                app_logger.info(f"Метаданные загружены, признаков: {len(feature_names) if feature_names else 0}")
        
    except Exception as e:
        app_logger.error(f"Ошибка загрузки модели: {str(e)}")

@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Customer Churn Prediction API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerData):
    """
    Предсказание оттока для одного клиента
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        df = pd.DataFrame([customer.dict()])
        
        X_processed, _ = preprocessor.prepare_features(df)
        
        X_features = feature_engineer.create_features(X_processed)
        
        if feature_names:
            X_features = X_features[feature_names]
        
        churn_probability = float(model.predict_proba(X_features)[0, 1])
        
        if churn_probability > 0.8 or churn_probability < 0.2:
            confidence = "High"
        elif churn_probability > 0.6 or churn_probability < 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        risk_factors = {
            "tenure": float(customer.tenure),
            "contract_type": customer.Contract,
            "monthly_charges": float(customer.MonthlyCharges)
        }
        
        return PredictionResponse(
            churn_probability=round(churn_probability, 4),
            prediction="Yes" if churn_probability > 0.5 else "No",
            confidence=confidence,
            risk_factors=risk_factors
        )
        
    except Exception as e:
        app_logger.error(f"Ошибка предсказания: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(customers: list[CustomerData]):
    """
    Пакетное предсказание для нескольких клиентов
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        for customer in customers:
            df = pd.DataFrame([customer.dict()])
            X_processed, _ = preprocessor.prepare_features(df)
            X_features = feature_engineer.create_features(X_processed)
            
            if feature_names:
                X_features = X_features[feature_names]
            
            proba = float(model.predict_proba(X_features)[0, 1])
            results.append({
                "input": customer.dict(),
                "churn_probability": round(proba, 4),
                "prediction": "Yes" if proba > 0.5 else "No"
            })
        
        return {"predictions": results}
        
    except Exception as e:
        app_logger.error(f"Ошибка пакетного предсказания: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/custom-docs", response_class=HTMLResponse)
async def custom_docs():
    """Кастомная документация API"""
    return get_custom_docs_html()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)