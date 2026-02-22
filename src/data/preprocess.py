"""
Модуль для предобработки данных
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import yaml
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any
from src.utils.logger import app_logger

class DataPreprocessor:
    """Класс для предобработки данных перед обучением"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Инициализация препроцессора
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = app_logger
        self.preprocessor = None
        self.label_encoders = {}
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Очистка данных от выбросов и ошибок
        
        Args:
            df: Исходный DataFrame
            
        Returns:
            pd.DataFrame: Очищенный DataFrame
        """
        self.logger.info("Начало очистки данных")
        df_clean = df.copy()
        
        # Удаляем дубликаты
        initial_len = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        self.logger.info(f"Удалено дубликатов: {initial_len - len(df_clean)}")
        
        # Удаляем колонки, которые не будем использовать
        columns_to_drop = self.config['features'].get('features_to_drop', [])
        existing_cols = [col for col in columns_to_drop if col in df_clean.columns]
        if existing_cols:
            df_clean = df_clean.drop(columns=existing_cols)
            self.logger.info(f"Удалены колонки: {existing_cols}")
        
        # Обработка TotalCharges (может быть строкой)
        if 'TotalCharges' in df_clean.columns:
            df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        
        self.logger.info("Очистка данных завершена")
        return df_clean
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Обработка пропущенных значений
        
        Args:
            df: DataFrame с пропусками
            
        Returns:
            pd.DataFrame: DataFrame без пропусков
        """
        self.logger.info("Обработка пропущенных значений")
        df_filled = df.copy()
        
        # Проверяем пропуски
        missing = df_filled.isnull().sum()
        if missing.sum() > 0:
            self.logger.warning(f"Найдены пропуски:\n{missing[missing > 0]}")
            
            # Для числовых колонок заполняем медианой
            numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_filled[col].isnull().any():
                    median_val = df_filled[col].median()
                    df_filled[col].fillna(median_val, inplace=True)
                    self.logger.info(f"Колонка {col}: пропуски заполнены медианой {median_val:.2f}")
            
            # Для категориальных колонок заполняем модой
            categorical_cols = df_filled.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df_filled[col].isnull().any():
                    mode_val = df_filled[col].mode()[0]
                    df_filled[col].fillna(mode_val, inplace=True)
                    self.logger.info(f"Колонка {col}: пропуски заполнены модой {mode_val}")
        
        return df_filled
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Кодирование категориальных признаков
        
        Args:
            df: DataFrame с категориальными признаками
            
        Returns:
            pd.DataFrame: DataFrame с закодированными признаками
        """
        self.logger.info("Кодирование категориальных признаков")
        df_encoded = df.copy()
        
        categorical_cols = self.config['features']['categorical_features']
        existing_cat_cols = [col for col in categorical_cols if col in df_encoded.columns]
        
        for col in existing_cat_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            
            # Преобразуем в строки и кодируем
            df_encoded[col] = df_encoded[col].astype(str)
            df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col])
            
            self.logger.info(f"Колонка {col} закодирована")
        
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Масштабирование числовых признаков
        
        Args:
            df: DataFrame с числовыми признаками
            
        Returns:
            pd.DataFrame: DataFrame с масштабированными признаками
        """
        self.logger.info("Масштабирование числовых признаков")
        df_scaled = df.copy()
        
        numeric_cols = self.config['features']['numerical_features']
        existing_num_cols = [col for col in numeric_cols if col in df_scaled.columns]
        
        if existing_num_cols:
            for col in existing_num_cols:
                if df_scaled[col].isna().any():
                    self.logger.warning(f"Колонка {col} содержит NaN, заполняем медианой")
                    df_scaled[col] = df_scaled[col].fillna(df_scaled[col].median())
            
            scaler = StandardScaler()
            df_scaled[existing_num_cols] = scaler.fit_transform(df_scaled[existing_num_cols])
            
            self.scaler = scaler
            self.logger.info(f"Масштабированы колонки: {existing_num_cols}")
        
        return df_scaled
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Полный пайплайн подготовки признаков
        
        Args:
            df: Исходный DataFrame
            
        Returns:
            pd.DataFrame: Подготовленный DataFrame
        """
        self.logger.info("=" * 50)
        self.logger.info("НАЧАЛО ПОДГОТОВКИ ПРИЗНАКОВ")
        
        # Очистка
        df = self.clean_data(df)
        
        # Обработка пропусков
        df = self.handle_missing_values(df)
        
        # Разделяем фичи и таргет
        target_col = self.config['data']['target_column']
        y = None
        if target_col in df.columns:
            y = df[target_col].copy()
            # Кодируем таргет (Yes/No -> 1/0)
            if y.dtype == 'object':
                y = (y == 'Yes').astype(int)
            df = df.drop(columns=[target_col])
        
        # Кодирование категориальных
        df = self.encode_categorical(df)
        
        # Масштабирование числовых
        df = self.scale_features(df)
        
        # Добавляем проверку типов для XGBoost !!! ЭТО НОВАЯ СТРОКА !!!
        df = self.ensure_numeric_types(df)
        
        self.logger.info(f"Итоговое количество признаков: {df.shape[1]}")
        self.logger.info(f"Итоговое количество записей: {df.shape[0]}")
        self.logger.info("ПОДГОТОВКА ПРИЗНАКОВ ЗАВЕРШЕНА")
        self.logger.info("=" * 50)
        
        if y is not None:
            return df, y
        return df
    
    def save_preprocessor(self, path: str = "artifacts/models/preprocessor.joblib"):
        """
        Сохранение препроцессора
        
        Args:
            path: Путь для сохранения
        """
        preprocessor_artifacts = {
            'label_encoders': self.label_encoders,
            'scaler': getattr(self, 'scaler', None),
            'config': self.config
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor_artifacts, path)
        self.logger.info(f"Препроцессор сохранен в {path}")
    
    def load_preprocessor(self, path: str = "artifacts/models/preprocessor.joblib"):
        """
        Загрузка препроцессора
        
        Args:
            path: Путь к файлу препроцессора
        """
        artifacts = joblib.load(path)
        self.label_encoders = artifacts['label_encoders']
        self.scaler = artifacts['scaler']
        self.logger.info(f"Препроцессор загружен из {path}")

    def ensure_numeric_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Убеждаемся, что все данные числовые для XGBoost
        
        Args:
            df: DataFrame
            
        Returns:
            pd.DataFrame: DataFrame с числовыми типами
        """
        self.logger.info("Проверка типов данных для XGBoost")
        
        df_numeric = df.copy()
        
        for col in df_numeric.columns:
            if df_numeric[col].dtype == 'object':
                self.logger.warning(f"Колонка {col} имеет тип object, преобразуем в category и затем в int")
                df_numeric[col] = pd.Categorical(df_numeric[col]).codes
            elif df_numeric[col].dtype == 'category':
                self.logger.info(f"Колонка {col} преобразуем из category в int")
                df_numeric[col] = df_numeric[col].astype(int)
            elif df_numeric[col].dtype == 'bool':
                self.logger.info(f"Колонка {col} преобразуем из bool в int")
                df_numeric[col] = df_numeric[col].astype(int)
        
        self.logger.info("Все данные приведены к числовым типам")
        return df_numeric