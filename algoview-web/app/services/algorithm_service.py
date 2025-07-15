from flask import current_app
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

logger = logging.getLogger(__name__)

class AlgorithmService:
    def __init__(self):
        """Инициализация сервиса алгоритмов"""
        logger.info("Initializing AlgorithmService")
        
        # Словарь доступных алгоритмов
        self.algorithms = {
            'logistic_regression': LogisticRegression(random_state=42),
            'lda': LinearDiscriminantAnalysis(),
            'gaussian_nb': GaussianNB(),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        # Словарь для хранения обученных моделей
        self.trained_models = {}
        
        logger.info(f"Available algorithms: {list(self.algorithms.keys())}")
        
    def get_algorithm(self, algo_id):
        """Получает алгоритм по ID"""
        if algo_id not in self.algorithms:
            logger.warning(f"Algorithm {algo_id} not found")
            return None
            
        # Если модель еще не обучена, создаем новый экземпляр
        if algo_id not in self.trained_models:
            self.trained_models[algo_id] = self.algorithms[algo_id].__class__(**self.algorithms[algo_id].get_params())
            
        logger.debug(f"Returning algorithm: {algo_id}")
        return self.trained_models[algo_id]
        
    def calculate_metrics(self, y_true, y_pred):
        """Рассчитывает метрики качества модели"""
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred)
            }
            
            logger.debug(f"Calculated metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}", exc_info=True)
            return {
                'error': f'Ошибка при расчете метрик: {str(e)}'
            } 