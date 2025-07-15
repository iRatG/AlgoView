import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Базовая конфигурация"""
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
    
    # Flask
    FLASK_APP = 'run.py'
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    
    # Загрузка файлов
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB максимальный размер файла
    ALLOWED_EXTENSIONS = {'csv', 'json', 'xlsx'}
    
    # Celery
    CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
    
    # Алгоритмы
    AVAILABLE_ALGORITHMS = [
        'LogisticRegression',
        'DecisionTree',
        'RandomForest',
        'KNN',
        'GaussianNB',
        'LDA'
    ]
    
    # Визуализация
    MAX_PLOT_POINTS = 1000  # Ограничение точек для визуализации
    DEFAULT_PLOT_THEME = 'plotly'  # или 'd3'
    
    # Данные
    MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB 