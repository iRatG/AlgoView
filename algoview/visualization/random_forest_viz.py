import numpy as np
import matplotlib.pyplot as plt
from ..ensemble.bagging import RandomForest

class RandomForestVisualizer:
    def __init__(self):
        # Генерируем синтетические данные
        np.random.seed(42)
        n_samples = 100
        
        # Создаем спиральные данные
        def make_spiral(n_samples, noise=0.5):
            n = np.sqrt(np.random.rand(n_samples)) * 780 * (2 * np.pi) / 360
            d1x = -np.cos(n) * n + np.random.rand(n_samples) * noise
            d1y = np.sin(n) * n + np.random.rand(n_samples) * noise
            return np.vstack((d1x, d1y)).T

        X1 = make_spiral(n_samples)
        X2 = make_spiral(n_samples) + np.array([5, 5])
        self.X = np.vstack([X1, X2])
        self.y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
        
        # Создаем и обучаем модель
        self.forest = RandomForest(n_estimators=5, max_depth=5)
        self.forest.fit(self.X, self.y)

    def plot_decision_boundary(self, model, X, y, title):
        # Создаем сетку точек
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))

        # Получаем предсказания для всех точек сетки
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Отрисовываем границу решений
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        plt.title(title)
        plt.xlabel('Признак 1 (лимит по карте)')
        plt.ylabel('Признак 2 (сумма последнего платежа)')

    def visualize(self):
        n_cols = 3
        n_rows = 2
        plt.figure(figsize=(15, 10))
        
        # Визуализируем каждое дерево отдельно
        for i, tree in enumerate(self.forest.trees[:5]):
            plt.subplot(n_rows, n_cols, i + 1)
            self.plot_decision_boundary(tree, self.X, self.y, f'Дерево {i+1}')
        
        # Визуализируем итоговый ансамбль
        plt.subplot(n_rows, n_cols, 6)
        self.plot_decision_boundary(self.forest, self.X, self.y, 'Случайный лес (все деревья)')
        
        plt.tight_layout()
        plt.show()
        
        # Визуализируем важность признаков
        if hasattr(self.forest, 'feature_importances_'):
            plt.figure(figsize=(8, 4))
            importances = self.forest.feature_importances_
            plt.bar(range(len(importances)), importances)
            plt.title('Важность признаков')
            plt.xlabel('Индекс признака')
            plt.ylabel('Важность')
            plt.show()

if __name__ == "__main__":
    visualizer = RandomForestVisualizer()
    visualizer.visualize() 