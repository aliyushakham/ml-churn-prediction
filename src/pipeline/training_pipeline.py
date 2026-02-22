"""
Основной пайплайн обучения
Объединяет все модули в один поток
"""
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from src.data.load_data import DataLoader
from src.data.preprocess import DataPreprocessor
from src.features.build_features import FeatureEngineer
from src.models.train import ModelTrainer
from src.utils.logger import app_logger
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class TrainingPipeline:
    """Главный класс пайплайна обучения"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Инициализация пайплайна
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        self.config_path = config_path
        self.logger = app_logger
        
        self.data_loader = DataLoader(config_path)
        self.preprocessor = DataPreprocessor(config_path)
        self.feature_engineer = FeatureEngineer(config_path)
        self.model_trainer = ModelTrainer(config_path)
        
    def run(self):
        """
        Запуск полного пайплайна обучения
        """
        self.logger.info("=" * 60)
        self.logger.info("ЗАПУСК TRAINING PIPELINE")
        self.logger.info("=" * 60)
        
        try:
            self.logger.info("\n📥 Шаг 1: Загрузка данных")
            df = self.data_loader.load_raw_data()
            
            is_valid, message = self.data_loader.validate_data(df)
            if not is_valid:
                self.logger.error(f"Ошибка валидации данных: {message}")
                return False
            
            self.logger.info("\n🧹 Шаг 2: Предобработка данных")
            X, y = self.preprocessor.prepare_features(df)
            
            self.logger.info("\n⚙️ Шаг 3: Feature Engineering")
            X = self.feature_engineer.create_features(X)
            
            self.logger.info("\n🤖 Шаг 4: Обучение модели")
            model, metrics = self.model_trainer.run_training(X, y)
            
            self.logger.info("\n💾 Шаг 5: Сохранение артефактов")
            self.preprocessor.save_preprocessor()
            
            self.logger.info("\n📊 Шаг 6: Кросс-валидация")
            cv_results = self.model_trainer.cross_validate(X, y)
            
            self.logger.info("=" * 60)
            self.logger.info("✅ PIPELINE УСПЕШНО ЗАВЕРШЕН!")
            self.logger.info("=" * 60)
            
            self.logger.info("\n📈 ИТОГОВЫЕ МЕТРИКИ:")
            self.logger.info(f"ROC-AUC на тесте: {metrics['roc_auc']:.4f}")
            self.logger.info(f"F1-Score на тесте: {metrics['f1']:.4f}")
            self.logger.info(f"CV ROC-AUC: {cv_results['roc_auc']['mean']:.4f} ± {cv_results['roc_auc']['std']:.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка в пайплайне: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Точка входа для запуска пайплайна"""
    pipeline = TrainingPipeline()
    success = pipeline.run()
    
    if success:
        print("\n✨ Пайплайн выполнен успешно! Модель готова к использованию.")
    else:
        print("\n❌ Пайплайн завершился с ошибкой. Проверьте логи.")

if __name__ == "__main__":
    main()