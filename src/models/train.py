"""
Модуль для обучения модели XGBoost
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                           recall_score, f1_score, confusion_matrix, classification_report)
import xgboost as xgb
import yaml
import joblib
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.logger import app_logger

class ModelTrainer:
    """Класс для обучения модели"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Инициализация тренера модели
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = app_logger
        self.model = None
        self.feature_names = None
        
    def split_data(self, X: pd.DataFrame, y: pd.Series):
        """
        Разделение данных на train/test
        
        Args:
            X: Признаки
            y: Целевая переменная
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        self.logger.info("Разделение данных на train/test")
        
        test_size = self.config['data']['test_size']
        random_state = self.config['data']['random_state']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y 
        )
        
        self.logger.info(f"Размер обучающей выборки: {len(X_train)}")
        self.logger.info(f"Размер тестовой выборки: {len(X_test)}")
        self.logger.info(f"Соотношение классов в train: {y_train.mean():.3f}")
        self.logger.info(f"Соотношение классов в test: {y_test.mean():.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series, X_val=None, y_val=None):
        """
        Обучение XGBoost модели
        
        Args:
            X_train: Обучающие признаки
            y_train: Обучающие цели
            X_val: Валидационные признаки (опционально)
            y_val: Валидационные цели (опционально)
            
        Returns:
            Обученная модель
        """
        self.logger.info("=" * 50)
        self.logger.info("НАЧАЛО ОБУЧЕНИЯ XGBOOST")
        
        self.feature_names = X_train.columns.tolist()
        
        params = self.config['model'].get('params', {})
        
        xgb_params = {
            'n_estimators': params.get('n_estimators', 200),
            'learning_rate': params.get('learning_rate', 0.05),
            'max_depth': params.get('max_depth', 7),
            'subsample': params.get('subsample', 0.8),
            'colsample_bytree': params.get('colsample_bytree', 0.8),
            'random_state': params.get('random_state', 42),
            'n_jobs': params.get('n_jobs', -1),
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'objective': 'binary:logistic',
            'scale_pos_weight': self._calculate_scale_pos_weight(y_train)  # Балансировка классов
        }
        
        self.logger.info(f"Параметры модели: {xgb_params}")
        
        self.model = xgb.XGBClassifier(**xgb_params)
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train, 
                y_train,
                eval_set=eval_set,
                verbose=50
            )
            
            results = self.model.evals_result()
            train_loss = results['validation_0']['logloss'][-1]
            val_loss = results['validation_1']['logloss'][-1]
            self.logger.info(f"Финальная потеря на train: {train_loss:.4f}")
            self.logger.info(f"Финальная потеря на val: {val_loss:.4f}")
        else:
            self.model.fit(X_train, y_train)
        
        self.logger.info("ОБУЧЕНИЕ ЗАВЕРШЕНО")
        self.logger.info("=" * 50)
        
        return self.model
    
    def _calculate_scale_pos_weight(self, y):
        """
        Расчет веса для балансировки классов
        
        Args:
            y: Целевая переменная
            
        Returns:
            float: Вес для положительного класса
        """
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        
        if pos_count == 0:
            return 1.0
        
        scale_pos_weight = neg_count / pos_count
        self.logger.info(f"Scale pos weight: {scale_pos_weight:.2f} (neg: {neg_count}, pos: {pos_count})")
        
        return scale_pos_weight
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Оценка модели на тестовых данных
        
        Args:
            model: Обученная модель
            X_test: Тестовые признаки
            y_test: Тестовые цели
            
        Returns:
            dict: Словарь с метриками
        """
        self.logger.info("Оценка модели на тестовых данных")
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        self.logger.info("=" * 50)
        self.logger.info("МЕТРИКИ МОДЕЛИ:")
        for metric_name, value in metrics.items():
            self.logger.info(f"{metric_name}: {value:.4f}")
        
        cm = confusion_matrix(y_test, y_pred)
        self.logger.info(f"\nConfusion Matrix:")
        self.logger.info(f"TN: {cm[0,0]}  FP: {cm[0,1]}")
        self.logger.info(f"FN: {cm[1,0]}  TP: {cm[1,1]}")
        
        self.logger.info(f"\nClassification Report:")
        self.logger.info(f"\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")
        self.logger.info("=" * 50)
        
        return metrics
    
    def plot_feature_importance(self, model, top_n: int = 20):
        """
        Построение графика важности признаков
        
        Args:
            model: Обученная модель
            top_n: Количество топ признаков для отображения
            
        Returns:
            pd.DataFrame: Важность признаков
        """
        self.logger.info("Построение графика важности признаков")
        
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        top_features = feature_importance.head(top_n)
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        plt.barh(range(len(top_features)), top_features['importance'], color=colors)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance (F-score)')
        plt.title(f'Top {top_n} Feature Importance (XGBoost)')
        plt.gca().invert_yaxis()
        
        for i, v in enumerate(top_features['importance']):
            plt.text(v, i, f' {v:.3f}', va='center')
        
        plt.tight_layout()
        
        save_path = Path("artifacts/metrics/feature_importance.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"График сохранен в {save_path}")
        
        csv_path = Path("artifacts/metrics/feature_importance.csv")
        feature_importance.to_csv(csv_path, index=False)
        self.logger.info(f"CSV сохранен в {csv_path}")
        
        self.logger.info("\nТоп-5 самых важных признаков:")
        for idx, row in top_features.head().iterrows():
            self.logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return feature_importance
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_folds: int = 5) -> dict:
        """
        Кросс-валидация модели
        
        Args:
            X: Признаки
            y: Целевая переменная
            n_folds: Количество фолдов
            
        Returns:
            dict: Результаты кросс-валидации
        """
        self.logger.info(f"Проведение {n_folds}-fold кросс-валидации")
        
        params = self.config['model'].get('params', {})
        
        xgb_params = {
            'n_estimators': params.get('n_estimators', 200),
            'learning_rate': params.get('learning_rate', 0.05),
            'max_depth': params.get('max_depth', 7),
            'subsample': params.get('subsample', 0.8),
            'colsample_bytree': params.get('colsample_bytree', 0.8),
            'random_state': params.get('random_state', 42),
            'n_jobs': -1,
            'use_label_encoder': False,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }
        
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        scores = {
            'roc_auc': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = xgb.XGBClassifier(**xgb_params)
            model.fit(
                X_train, 
                y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False
            )
            
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = model.predict(X_val)
            
            scores['roc_auc'].append(roc_auc_score(y_val, y_pred_proba))
            scores['accuracy'].append(accuracy_score(y_val, y_pred))
            scores['precision'].append(precision_score(y_val, y_pred))
            scores['recall'].append(recall_score(y_val, y_pred))
            scores['f1'].append(f1_score(y_val, y_pred))
            
            self.logger.info(f"Fold {fold} - ROC-AUC: {scores['roc_auc'][-1]:.4f}, F1: {scores['f1'][-1]:.4f}")
        
        cv_results = {}
        for metric in scores:
            cv_results[metric] = {
                'mean': float(np.mean(scores[metric])),
                'std': float(np.std(scores[metric]))
            }
            self.logger.info(f"{metric}: {cv_results[metric]['mean']:.4f} ± {cv_results[metric]['std']:.4f}")
        
        return cv_results
    
    def save_model(self, metrics: dict):
        """
        Сохранение модели и метрик
        
        Args:
            metrics: Словарь с метриками
        """
        Path("artifacts/models").mkdir(parents=True, exist_ok=True)
        Path("artifacts/metrics").mkdir(parents=True, exist_ok=True)
        
        model_path = "artifacts/models/latest_model.joblib"
        joblib.dump(self.model, model_path)
        self.logger.info(f"Модель сохранена в {model_path}")
        
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'xgboost',
            'params': self.model.get_params(),
            'metrics': metrics,
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'n_classes': self.model.n_classes_ if hasattr(self.model, 'n_classes_') else 2
        }
        
        metadata_path = "artifacts/models/model_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Метаданные сохранены в {metadata_path}")
        
        metrics_path = "artifacts/metrics/latest_metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Метрики сохранены в {metrics_path}")
    
    def load_model(self, model_path: str = "artifacts/models/latest_model.joblib"):
        """
        Загрузка сохраненной модели
        
        Args:
            model_path: Путь к модели
        """
        self.model = joblib.load(model_path)
        self.logger.info(f"Модель загружена из {model_path}")
        
        metadata_path = model_path.replace('.joblib', '_metadata.json')
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.feature_names = metadata.get('feature_names')
                self.logger.info(f"Метаданные загружены, признаков: {len(self.feature_names) if self.feature_names else 0}")
    
    def run_training(self, X: pd.DataFrame, y: pd.Series):
        """
        Полный цикл обучения
        
        Args:
            X: Признаки
            y: Целевая переменная
            
        Returns:
            tuple: (модель, метрики)
        """
        self.logger.info("=" * 60)
        self.logger.info("ЗАПУСК ПОЛНОГО ЦИКЛА ОБУЧЕНИЯ")
        self.logger.info("=" * 60)
        
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train, 
            test_size=0.2, 
            random_state=42,
            stratify=y_train
        )
        
        model = self.train_xgboost(X_train_final, y_train_final, X_val, y_val)
        
        metrics = self.evaluate_model(model, X_test, y_test)
        
        feature_importance = self.plot_feature_importance(model)
        
        cv_results = self.cross_validate(X_train, y_train)
        
        metrics['cv_mean_roc_auc'] = cv_results['roc_auc']['mean']
        metrics['cv_std_roc_auc'] = cv_results['roc_auc']['std']
        
        self.save_model(metrics)
        
        self.logger.info("=" * 60)
        self.logger.info("✅ ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
        self.logger.info("=" * 60)
        
        return model, metrics

if __name__ == "__main__":
    print("Модуль ModelTrainer с XGBoost готов к работе!")