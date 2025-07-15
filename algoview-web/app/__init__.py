import os
from flask import Flask
from flask_cors import CORS
import logging

logger = logging.getLogger(__name__)

def create_app():
    """Создание и настройка приложения Flask"""
    app = Flask(__name__)
    CORS(app)
    
    # Настройка путей
    app.config['ROOT_DIR'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app.config['DATA_DIR'] = os.path.join(app.config['ROOT_DIR'], 'data')
    
    # Проверяем наличие директории с данными
    if not os.path.exists(app.config['DATA_DIR']):
        os.makedirs(app.config['DATA_DIR'])
        logger.warning(f"Created data directory: {app.config['DATA_DIR']}")
    
    # Регистрируем blueprints
    try:
        from app.routes import web, algorithms, datasets
        app.register_blueprint(web.bp)
        app.register_blueprint(algorithms.bp)
        app.register_blueprint(datasets.bp)
    except Exception as e:
        logger.error(f"Error registering blueprints: {str(e)}", exc_info=True)
        raise
    
    return app 