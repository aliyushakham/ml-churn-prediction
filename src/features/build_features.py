"""
Модуль для создания новых признаков (Feature Engineering)
"""
import pandas as pd
import numpy as np
from src.utils.logger import app_logger
import yaml

class FeatureEngineer:
    """Класс для создания новых признаков"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Инициализация инженера признаков
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = app_logger
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание новых признаков
        
        Args:
            df: Исходный DataFrame
            
        Returns:
            pd.DataFrame: DataFrame с новыми признаками
        """
        self.logger.info("=" * 50)
        self.logger.info("НАЧАЛО СОЗДАНИЯ ПРИЗНАКОВ")
        
        df_features = df.copy()
        
        # Проверяем какие фичи нужно создать
        create_features = self.config['features'].get('create_features', {})
        
        if create_features.get('avg_charge_per_month') and all(col in df.columns for col in ['TotalCharges', 'tenure']):
            df_features['AvgMonthlyCharges'] = df_features['TotalCharges'] / (df_features['tenure'] + 1)
            # Заполняем возможные NaN (если tenure = -1 или что-то такое)
            df_features['AvgMonthlyCharges'] = df_features['AvgMonthlyCharges'].fillna(0)
            self.logger.info("Создан признак: AvgMonthlyCharges")
        
        # 2. Группы по длительности обслуживания
        if create_features.get('tenure_group') and 'tenure' in df.columns:
            # Сначала создаем категории, потом заполняем NaN и конвертируем в int
            df_features['TenureGroup'] = pd.cut(
                df_features['tenure'],
                bins=[-1, 12, 24, 48, 72, float('inf')],  # Добавили -1 для захвата всех значений
                labels=[0, 1, 2, 3, 4]
            )
            # Заполняем NaN (если есть) значением 0
            df_features['TenureGroup'] = df_features['TenureGroup'].fillna(0)
            # Конвертируем в int
            df_features['TenureGroup'] = df_features['TenureGroup'].astype(int)
            self.logger.info("Создан признак: TenureGroup (числовой)")
        
        # 3. Количество подключенных услуг
        if create_features.get('service_count'):
            service_columns = [
                'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies'
            ]
            
            existing_services = [col for col in service_columns if col in df.columns]
            
            if existing_services:
                service_count = 0
                for col in existing_services:
                    if col == 'InternetService':
                        # InternetService: 0,1,2 (0 = No, 1 = DSL, 2 = Fiber optic)
                        service_count += (df_features[col] > 0).astype(int)
                    else:
                        # Другие сервисы: 0,1,2 (0 = No, 1 = Yes, 2 = No internet service)
                        service_count += (df_features[col] == 1).astype(int)  # 1 = Yes
                
                df_features['ServiceCount'] = service_count
                self.logger.info("Создан признак: ServiceCount")
        
        # 4. Отношение MonthlyCharges к среднему по тарифу
        if 'MonthlyCharges' in df.columns and 'Contract' in df.columns:
            avg_by_contract = df_features.groupby('Contract')['MonthlyCharges'].transform('mean')
            df_features['MonthlyChargesRatio'] = df_features['MonthlyCharges'] / avg_by_contract
            # Заполняем возможные NaN
            df_features['MonthlyChargesRatio'] = df_features['MonthlyChargesRatio'].fillna(1.0)
            self.logger.info("Создан признак: MonthlyChargesRatio")
        
        # 5. Семейный статус (Partner + Dependents)
        if 'Partner' in df.columns and 'Dependents' in df.columns:
            df_features['HasFamily'] = ((df_features['Partner'] == 1) |  # 1 = 'Yes'
                                        (df_features['Dependents'] == 1)).astype(int)
            self.logger.info("Создан признак: HasFamily")
        
        # 6. Лояльность клиента
        if 'tenure' in df.columns:
            df_features['LoyaltyLevel'] = pd.cut(
                df_features['tenure'],
                bins=[-1, 12, 36, 72],  # Добавили -1 для захвата всех значений
                labels=[0, 1, 2]
            )
            # Заполняем NaN (если есть) значением 0
            df_features['LoyaltyLevel'] = df_features['LoyaltyLevel'].fillna(0)
            # Конвертируем в int
            df_features['LoyaltyLevel'] = df_features['LoyaltyLevel'].astype(int)
            self.logger.info("Создан признак: LoyaltyLevel (числовой)")
        
        # Дополнительная проверка: убеждаемся что нет NaN
        nan_columns = df_features.columns[df_features.isna().any()].tolist()
        if nan_columns:
            self.logger.warning(f"Найдены NaN в колонках: {nan_columns}. Заполняем нулями.")
            for col in nan_columns:
                if df_features[col].dtype in ['int64', 'float64']:
                    df_features[col] = df_features[col].fillna(0)
                else:
                    df_features[col] = df_features[col].fillna(0).astype(int)
        
        # Убеждаемся, что все колонки имеют правильные типы для XGBoost
        for col in df_features.columns:
            if df_features[col].dtype == 'category':
                df_features[col] = df_features[col].astype(int)
                self.logger.info(f"Колонка {col} преобразована из category в int")
            elif df_features[col].dtype == 'object':
                self.logger.warning(f"Колонка {col} все еще object, преобразуем в int")
                df_features[col] = pd.Categorical(df_features[col]).codes
        
        self.logger.info(f"Добавлено новых признаков: {len(df_features.columns) - len(df.columns)}")
        self.logger.info("СОЗДАНИЕ ПРИЗНАКОВ ЗАВЕРШЕНО")
        self.logger.info("=" * 50)
        
        return df_features
    
    def get_feature_importance_description(self) -> dict:
        """
        Возвращает описание созданных признаков
        
        Returns:
            dict: Словарь с описанием признаков
        """
        return {
            'AvgMonthlyCharges': 'Среднемесячный платеж (TotalCharges / tenure)',
            'TenureGroup': 'Группа по длительности обслуживания (0-4)',
            'ServiceCount': 'Количество подключенных услуг',
            'MonthlyChargesRatio': 'Отношение платежа к среднему по типу контракта',
            'HasFamily': 'Наличие семьи (1 - есть, 0 - нет)',
            'LoyaltyLevel': 'Уровень лояльности (0-2)'
        }