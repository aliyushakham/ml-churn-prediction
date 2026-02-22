# Customer Churn Prediction ML Pipeline 🚀

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)

## 📋 О проекте

ML сервис для предсказания оттока клиентов. 
Модель анализирует данные и возвращает вероятность того, что клиент уйдет из компании.

## 🏗️ Архитектура

**Data Layer** — загрузка и очистка данных  
**Feature Layer** — кодирование и создание признаков  
**Model Layer** — XGBoost обучение  
**API Layer** — FastAPI эндпоинты  

## 📊 Результаты

| Метрика | Значение |
|---------|----------|
| ROC-AUC | 0.85 |
| Accuracy | 0.82 |
| F1-Score | 0.75 |

## 🚀 Быстрый старт

```bash
# Клонируем
git clone https://github.com/aliyushakham/ml-churn-prediction.git
cd ml-churn-prediction

# Виртуальное окружение
python3 -m venv venv
source venv/bin/activate  # для Mac

# Зависимости
pip install -r requirements.txt

# Обучаем модель
python -m src.pipeline.training_pipeline

# Запускаем API
uvicorn src.api.main:app --reload --port 8000