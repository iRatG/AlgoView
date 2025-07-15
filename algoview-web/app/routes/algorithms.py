from flask import Blueprint, render_template, jsonify, request, current_app
from app.services.algorithm_service import AlgorithmService
from app.services.visualization_service import VisualizationService
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import logging
import sys
import json
from sklearn.preprocessing import StandardScaler

# Настраиваем логирование с учетом кодировки Windows
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('algoview.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

bp = Blueprint('algorithms', __name__, url_prefix='/api')
algo_service = AlgorithmService()
viz_service = VisualizationService()

# Глобальные переменные для хранения данных
real_data = None
demo_splits = None

def init_data():
    """Инициализация данных при первом запросе"""
    global real_data, demo_splits
    
    try:
        # 1. Загружаем реальные данные для обучения
        data_path = os.path.join(current_app.config['DATA_DIR'], 'UCI_Credit_Card.csv')
        
        logger.info(f"Loading real data from: {data_path}")
        
        # Словарь для хранения реальных и демонстрационных данных
        real_data = {}
        demo_data = {}
        
        if os.path.exists(data_path):
            logger.info("Real data file found, reading CSV")
            try:
                df = pd.read_csv(data_path)
                
                if df.empty:
                    logger.error("Loaded CSV file is empty")
                    real_data = None
                else:
                    # Проверяем наличие необходимых колонок
                    required_columns = ['default.payment.next.month', 'ID']
                    if not all(col in df.columns for col in required_columns):
                        logger.error(f"Missing required columns: {required_columns}")
                        real_data = None
                    else:
                        # Подготовка реальных данных
                        X_real = df.drop(['default.payment.next.month', 'ID'], axis=1)
                        y_real = df['default.payment.next.month']
                        
                        # Проверяем на пропущенные значения
                        if X_real.isnull().any().any() or y_real.isnull().any():
                            logger.warning("Found missing values in data, filling with mean/mode")
                            X_real = X_real.fillna(X_real.mean())
                            y_real = y_real.fillna(y_real.mode()[0])
                        
                        # Стандартизация данных
                        scaler = StandardScaler()
                        X_real_scaled = scaler.fit_transform(X_real)
                        X_real_scaled = pd.DataFrame(X_real_scaled, columns=X_real.columns)
                        
                        # Разделяем реальные данные на обучающую и тестовую выборки
                        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
                            X_real_scaled, y_real, test_size=0.2, random_state=42
                        )
                        
                        # Сохраняем реальные данные для всех алгоритмов
                        real_data = {'data': (X_train_real, X_test_real, y_train_real, y_test_real)}
                        logger.info(f"Real data loaded. Shape: {X_real.shape}")
                
            except Exception as e:
                logger.error(f"Error reading real data: {str(e)}", exc_info=True)
                real_data = None
        else:
            logger.warning(f"Real data file not found at: {data_path}")
            real_data = None
        
        # 2. Создаем демонстрационные данные для визуализации
        logger.info("Creating demo data for visualization")
        n_samples = 1000
        
        # Создаем разные наборы данных для разных типов алгоритмов
        datasets = {}
        np.random.seed(42)
        
        # 1. Линейно разделимые данные для логистической регрессии и LDA
        X1_linear = np.random.normal(loc=[-2, -2], scale=[1, 1], size=(n_samples // 2, 2))
        X2_linear = np.random.normal(loc=[2, 2], scale=[1, 1], size=(n_samples // 2, 2))
        X_linear = np.vstack([X1_linear, X2_linear])
        y_linear = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
        
        # Перемешиваем данные
        indices = np.random.permutation(n_samples)
        X_linear = X_linear[indices]
        y_linear = y_linear[indices]
        
        # 2. Нелинейно разделимые данные для деревьев и случайного леса
        radius = np.random.normal(loc=2, scale=0.5, size=n_samples // 2)
        angles = np.random.uniform(0, 2 * np.pi, n_samples // 2)
        X1_nonlinear = np.vstack([
            radius * np.cos(angles),
            radius * np.sin(angles)
        ]).T
        
        radius = np.random.normal(loc=4, scale=0.5, size=n_samples // 2)
        angles = np.random.uniform(0, 2 * np.pi, n_samples // 2)
        X2_nonlinear = np.vstack([
            radius * np.cos(angles),
            radius * np.sin(angles)
        ]).T
        
        X_nonlinear = np.vstack([X1_nonlinear, X2_nonlinear])
        y_nonlinear = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
        
        # Перемешиваем данные
        indices = np.random.permutation(n_samples)
        X_nonlinear = X_nonlinear[indices]
        y_nonlinear = y_nonlinear[indices]
        
        # 3. Данные с несколькими кластерами для наивного Байеса
        X1_gauss = np.random.multivariate_normal(
            mean=[-2, -2],
            cov=[[1, 0.5], [0.5, 1]],
            size=n_samples // 4
        )
        X2_gauss = np.random.multivariate_normal(
            mean=[2, 2],
            cov=[[1, -0.5], [-0.5, 1]],
            size=n_samples // 4
        )
        X3_gauss = np.random.multivariate_normal(
            mean=[-2, 2],
            cov=[[1, 0], [0, 1]],
            size=n_samples // 4
        )
        X4_gauss = np.random.multivariate_normal(
            mean=[2, -2],
            cov=[[1, 0], [0, 1]],
            size=n_samples // 4
        )
        
        X_gauss = np.vstack([X1_gauss, X2_gauss, X3_gauss, X4_gauss])
        y_gauss = np.hstack([
            np.zeros(n_samples // 4),
            np.ones(n_samples // 4),
            np.zeros(n_samples // 4),
            np.ones(n_samples // 4)
        ])
        
        # Перемешиваем данные
        indices = np.random.permutation(n_samples)
        X_gauss = X_gauss[indices]
        y_gauss = y_gauss[indices]
        
        # Создаем словарь наборов данных для разных алгоритмов
        datasets = {
            'logistic_regression': (
                pd.DataFrame(X_linear, columns=['Feature 1', 'Feature 2']),
                pd.Series(y_linear, name='Target')
            ),
            'lda': (
                pd.DataFrame(X_linear, columns=['Feature 1', 'Feature 2']),
                pd.Series(y_linear, name='Target')
            ),
            'gaussian_nb': (
                pd.DataFrame(X_gauss, columns=['Feature 1', 'Feature 2']),
                pd.Series(y_gauss, name='Target')
            ),
            'decision_tree': (
                pd.DataFrame(X_nonlinear, columns=['Feature 1', 'Feature 2']),
                pd.Series(y_nonlinear, name='Target')
            ),
            'random_forest': (
                pd.DataFrame(X_nonlinear, columns=['Feature 1', 'Feature 2']),
                pd.Series(y_nonlinear, name='Target')
            )
        }
        
        # Разделяем демо-данные
        demo_splits = {}
        for algo_id, (X, y) in datasets.items():
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            demo_splits[algo_id] = (X_train, X_test, y_train, y_test)
        
        logger.info("Demo data created for visualization")
        
    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}", exc_info=True)
        raise

@bp.before_app_first_request
def initialize():
    """Инициализация данных перед первым запросом"""
    init_data()

def get_algorithms():
    """Возвращает список доступных алгоритмов"""
    return [
        {
            'category': 'Линейные модели',
            'algorithms': [
                {
                    'id': 'logistic_regression',
                    'name': 'Логистическая регрессия',
                    'description': 'Классический алгоритм для бинарной классификации'
                },
                {
                    'id': 'lda',
                    'name': 'Линейный дискриминантный анализ',
                    'description': 'Метод снижения размерности и классификации'
                }
            ]
        },
        {
            'category': 'Вероятностные модели',
            'algorithms': [
                {
                    'id': 'gaussian_nb',
                    'name': 'Наивный Байес',
                    'description': 'Вероятностный классификатор на основе теоремы Байеса'
                }
            ]
        },
        {
            'category': 'Деревья решений',
            'algorithms': [
                {
                    'id': 'decision_tree',
                    'name': 'Дерево решений',
                    'description': 'Иерархический алгоритм принятия решений'
                },
                {
                    'id': 'random_forest',
                    'name': 'Случайный лес',
                    'description': 'Ансамбль деревьев решений'
                }
            ]
        }
    ]

@bp.route('/algorithms')
def list_algorithms():
    """API endpoint для получения списка алгоритмов"""
    return jsonify(get_algorithms())

@bp.route('/train/<algo_id>', methods=['POST'])
def train_algorithm(algo_id):
    """API для обучения алгоритма"""
    try:
        logger.info(f"Starting training algorithm: {algo_id}")
        
        # Проверяем существование алгоритма
        algorithm = algo_service.get_algorithm(algo_id)
        if algorithm is None:
            return jsonify({
                'error': f'Algorithm {algo_id} not found'
            }), 404
        
        metrics = {}
        visualization = None
        
        # 1. Обучаем на реальных данных и получаем метрики
        if real_data is not None:
            X_train_real, X_test_real, y_train_real, y_test_real = real_data['data']
            
            # Создаем новый экземпляр алгоритма для реальных данных
            real_model = algorithm.__class__(**algorithm.get_params())
            
            logger.info("Training on real data...")
            real_model.fit(X_train_real, y_train_real)
            y_pred_real = real_model.predict(X_test_real)
            
            metrics = algo_service.calculate_metrics(y_test_real, y_pred_real)
            metrics['data_type'] = 'real'
            metrics['data_size'] = len(X_train_real) + len(X_test_real)
            logger.info(f"Real data metrics calculated: {metrics}")
        
        # 2. Обучаем на демо-данных для визуализации
        if algo_id in demo_splits:
            X_train_demo, X_test_demo, y_train_demo, y_test_demo = demo_splits[algo_id]
            
            logger.info("Training on demo data for visualization...")
            algorithm.fit(X_train_demo, y_train_demo)
            
            visualization = viz_service.get_visualization(algorithm, X_train_demo, y_train_demo)
            if 'error' in visualization:
                logger.error(f"Visualization error: {visualization['error']}")
                return jsonify({
                    'error': visualization['error']
                }), 500
                
            logger.info("Demo visualization created")
        
        return jsonify({
            'metrics': metrics,
            'visualization': visualization,
            'demo_note': 'Визуализация создана на демонстрационных данных для наглядности'
        })
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e)
        }), 500

@bp.route('/visualize/<algo_id>', methods=['POST'])
def visualize_algorithm(algo_id):
    """API для получения визуализации"""
    try:
        algorithm = algo_service.get_algorithm(algo_id)
        if algorithm is None:
            return jsonify({
                'error': f'Algorithm {algo_id} not found'
            }), 404
        
        if algo_id not in demo_splits:
            return jsonify({
                'error': 'Demo data not found for algorithm'
            }), 500
        
        X_train_demo, _, y_train_demo, _ = demo_splits[algo_id]
        visualization = viz_service.get_visualization(algorithm, X_train_demo, y_train_demo)
        
        if 'error' in visualization:
            logger.error(f"Visualization error: {visualization['error']}")
            return jsonify({
                'error': visualization['error']
            }), 500
        
        return jsonify({
            'visualization': visualization,
            'demo_note': 'Визуализация создана на демонстрационных данных для наглядности'
        })
        
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e)
        }), 500 