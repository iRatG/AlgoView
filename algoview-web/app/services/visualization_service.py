import io
import base64
import numpy as np
import plotly.graph_objects as go
from matplotlib.figure import Figure
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
import plotly.express as px
import logging
from sklearn.preprocessing import StandardScaler
import pandas as pd

logger = logging.getLogger(__name__)

class VisualizationService:
    """Сервис для конвертации и управления визуализациями"""
    
    def __init__(self):
        logger.info("Initializing VisualizationService")
        self.default_layout = {
            'template': 'plotly_white',
            'font': {'family': 'Arial, sans-serif'},
            'showlegend': True,
            'hovermode': 'closest',
            'width': 800,
            'height': 600
        }
        self.scaler = StandardScaler()
    
    def matplotlib_to_plotly(self, fig: Figure) -> dict:
        """Конвертирует matplotlib Figure в Plotly Figure"""
        # Получаем данные из matplotlib
        traces = []
        
        for ax in fig.axes:
            for line in ax.lines:
                trace = go.Scatter(
                    x=line.get_xdata(),
                    y=line.get_ydata(),
                    mode='lines+markers' if line.get_marker() else 'lines',
                    name=line.get_label() or 'Trace'
                )
                traces.append(trace)
            
            for collection in ax.collections:
                if hasattr(collection, 'get_offsets'):
                    points = collection.get_offsets()
                    if len(points):
                        trace = go.Scatter(
                            x=points[:, 0],
                            y=points[:, 1],
                            mode='markers',
                            marker={'color': collection.get_facecolor()[0]},
                            name=collection.get_label() or 'Points'
                        )
                        traces.append(trace)
        
        # Создаем layout
        layout = self.default_layout.copy()
        layout.update({
            'title': fig._suptitle.get_text() if fig._suptitle else '',
            'xaxis_title': ax.get_xlabel(),
            'yaxis_title': ax.get_ylabel()
        })
        
        return {'data': traces, 'layout': layout}
    
    def get_visualization(self, model, X, y):
        """Создает визуализацию для модели"""
        try:
            logger.info("Creating visualization")
            
            # Проверяем входные данные
            if X is None or y is None:
                logger.error("Input data is None")
                return {'error': 'Отсутствуют входные данные'}
                
            if not isinstance(X, pd.DataFrame):
                logger.error("X is not a pandas DataFrame")
                return {'error': 'Неверный формат входных данных: X должен быть DataFrame'}
                
            if len(X) != len(y):
                logger.error(f"Mismatched lengths: X={len(X)}, y={len(y)}")
                return {'error': 'Несоответствие размеров входных данных'}
            
            # Проверяем размерность данных
            if X.shape[1] != 2:
                logger.warning(f"Expected 2 features for visualization, got {X.shape[1]}")
                return {'error': 'Для визуализации требуется ровно 2 признака'}
            
            # Проверяем наличие пропущенных значений
            if X.isna().any().any() or pd.isna(y).any():
                logger.error("Found missing values in data")
                return {'error': 'Обнаружены пропущенные значения в данных'}
                
            try:
                # Пробуем преобразовать данные в числовой формат
                X = X.astype(float)
                y = pd.Series(y).astype(float)
            except Exception as e:
                logger.error(f"Error converting data to numeric: {str(e)}")
                return {'error': 'Ошибка преобразования данных в числовой формат'}
            
            # Проверяем, что модель обучена
            if not hasattr(model, 'predict') or not hasattr(model, 'fit'):
                logger.error("Model does not have required methods")
                return {'error': 'Модель не поддерживает необходимые методы'}
                
            try:
                # Пробуем сделать предсказание
                model.predict(X.iloc[:1])
            except Exception as e:
                logger.error(f"Model prediction failed: {str(e)}")
                return {'error': 'Ошибка при попытке использовать модель. Возможно, модель не обучена'}
            
            # Создаем сетку точек для визуализации
            x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
            y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 100),
                np.linspace(y_min, y_max, 100)
            )
            
            # Получаем предсказания модели
            mesh_X = pd.DataFrame(
                np.c_[xx.ravel(), yy.ravel()],
                columns=X.columns
            )
            
            if hasattr(model, 'predict_proba'):
                logger.debug("Using predict_proba for visualization")
                Z = model.predict_proba(mesh_X)[:, 1]
            else:
                logger.debug("Using predict for visualization")
                Z = model.predict(mesh_X)
            Z = Z.reshape(xx.shape)
            
            # Создаем данные для Plotly
            data = [
                # Контурный график вероятностей
                {
                    'type': 'contour',
                    'x': xx[0].tolist(),
                    'y': yy[:, 0].tolist(),
                    'z': Z.tolist(),
                    'colorscale': 'RdBu',
                    'opacity': 0.5,
                    'showscale': True,
                    'name': 'Границы решений'
                },
                # Точки обучающей выборки
                {
                    'type': 'scatter',
                    'x': X.iloc[:, 0].tolist(),
                    'y': X.iloc[:, 1].tolist(),
                    'mode': 'markers',
                    'marker': {
                        'color': y.tolist(),
                        'colorscale': 'RdBu',
                        'size': 8,
                        'opacity': 0.7,
                        'line': {
                            'color': 'white',
                            'width': 1
                        }
                    },
                    'name': 'Обучающие данные'
                }
            ]
            
            # Настройки макета
            layout = {
                'title': 'Визуализация решений модели',
                'xaxis': {
                    'title': X.columns[0],
                    'zeroline': False
                },
                'yaxis': {
                    'title': X.columns[1],
                    'zeroline': False
                },
                'hovermode': 'closest',
                'showlegend': True,
                'legend': {
                    'x': 1.05,
                    'y': 1
                },
                'plot_bgcolor': '#f8f9fa',
                'paper_bgcolor': '#ffffff'
            }
            
            logger.info("Visualization created successfully")
            return {
                'data': data,
                'layout': layout
            }
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}", exc_info=True)
            return {
                'error': f'Ошибка при создании визуализации: {str(e)}'
            }
    
    def get_demo_visualization(self, algorithm):
        """Создает демо-визуализацию для алгоритма"""
        # Генерируем демо-данные
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        # Обучаем алгоритм
        algorithm.fit(X, y)
        
        # Получаем визуализацию
        return self.get_visualization(algorithm)
    
    def get_training_visualization(self, algorithm, history):
        """Создает визуализацию процесса обучения"""
        fig = go.Figure()
        
        # График ошибки
        if 'loss' in history:
            fig.add_trace(go.Scatter(
                y=history['loss'],
                mode='lines',
                name='Ошибка'
            ))
        
        # График точности
        if 'accuracy' in history:
            fig.add_trace(go.Scatter(
                y=history['accuracy'],
                mode='lines',
                name='Точность'
            ))
        
        # Настройка layout
        layout = self.default_layout.copy()
        layout.update({
            'title': 'Процесс обучения',
            'xaxis_title': 'Эпоха',
            'yaxis_title': 'Значение'
        })
        fig.update_layout(layout)
        
        return fig 