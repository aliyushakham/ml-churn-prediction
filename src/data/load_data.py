"""
Модуль для загрузки данных
"""
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from typing import Optional, Tuple
from src.utils.logger import app_logger

class DataLoader:
    """Класс для загрузки и первичной обработки данных"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Инициализация загрузчика данных
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        self.config = self._load_config(config_path)
        self.logger = app_logger
        
    def _load_config(self, config_path: str) -> dict:
        """Загрузка конфигурации"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_raw_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Загрузка сырых данных
        
        Args:
            filepath: Путь к файлу с данными
            
        Returns:
            pd.DataFrame: Загруженные данные
        """
        if filepath is None:
            filepath = self.config['data']['raw_path']
        
        self.logger.info(f"Загрузка данных из {filepath}")
        
        # Создаем директорию, если её нет
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Проверяем существует ли файл
        if not Path(filepath).exists():
            self.logger.warning(f"Файл {filepath} не найден. Создаем тестовые данные.")
            return self._create_sample_data()
        
        # Загружаем данные
        df = pd.read_csv(filepath)
        self.logger.info(f"Загружено {len(df)} строк, {len(df.columns)} колонок")
        
        return df
    
    def _create_sample_data(self) -> pd.DataFrame:
        """
        Создание тестовых данных для демонстрации
        
        Returns:
            pd.DataFrame: Тестовые данные
        """
        self.logger.info("Создание тестовых данных")
        
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'customerID': [f'CUST_{i:05d}' for i in range(n_samples)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'Partner': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
            'tenure': np.random.randint(0, 72, n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
            'PaymentMethod': np.random.choice(
                ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], 
                n_samples
            ),
            'MonthlyCharges': np.random.uniform(20, 120, n_samples).round(2),
            'TotalCharges': np.random.uniform(100, 8000, n_samples).round(2),
            'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73])
        }
        
        df = pd.DataFrame(data)
        
        # Сохраняем тестовые данные
        save_path = self.config['data']['raw_path']
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        self.logger.info(f"Тестовые данные сохранены в {save_path}")
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Валидация данных
        
        Args:
            df: DataFrame для проверки
            
        Returns:
            Tuple[bool, str]: (прошла ли валидация, сообщение)
        """
        self.logger.info("Проверка качества данных")
        
        # Проверка на пустые данные
        if df.empty:
            return False, "DataFrame пуст"
        
        # Проверка на пропуски
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            self.logger.warning(f"Найдены пропуски в колонках: {missing_cols}")
        
        # Проверка целевой колонки
        target = self.config['data']['target_column']
        if target not in df.columns:
            return False, f"Целевая колонка {target} не найдена"
        
        # Проверка типов данных
        numeric_cols = self.config['features']['numerical_features']
        for col in numeric_cols:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    self.logger.warning(f"Колонка {col} должна быть числовой")
        
        self.logger.info("Валидация данных пройдена успешно")
        return True, "OK"