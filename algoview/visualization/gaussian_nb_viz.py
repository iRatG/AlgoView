import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from ..base_algorithms.probabilistic.naive_bayes import GaussianNB

class GaussianNBVisualizer:
    def __init__(self):
        # Генерируем синтетические данные
        np.random.seed(42)
        n_samples = 100
        
        # Класс 0
        mean1 = [2, 2]
        cov1 = [[1, 0.5], [0.5, 1]]
        X1 = np.random.multivariate_normal(mean1, cov1, n_samples)
        
        # Класс 1
        mean2 = [4, 4]
        cov2 = [[1.5, -0.5], [-0.5, 1.5]]
        X2 = np.random.multivariate_normal(mean2, cov2, n_samples)
        
        self.X = np.vstack([X1, X2])
        self.y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
        
        # Создаем и обучаем модель
        self.nb = GaussianNB()
        self.nb.fit(self.X, self.y)

    def plot_gaussian_contours(self, mean, std, color, alpha=0.3):
        x, y = np.mgrid[0:7:.01, 0:7:.01]
        pos = np.dstack((x, y))
        rv = multivariate_normal(mean, np.diag(std**2))
        plt.contour(x, y, rv.pdf(pos), colors=color, alpha=alpha)

    def visualize(self):
        plt.figure(figsize=(15, 5))
        
        # 1. Исходные данные
        plt.subplot(131)
        plt.scatter(self.X[self.y == 0][:, 0], self.X[self.y == 0][:, 1], 
                   c='blue', label='Класс 0', alpha=0.5)
        plt.scatter(self.X[self.y == 1][:, 0], self.X[self.y == 1][:, 1], 
                   c='red', label='Класс 1', alpha=0.5)
        plt.title('Исходные данные')
        plt.legend()
        plt.grid(True)
        
        # 2. Гауссовы распределения признаков
        plt.subplot(132)
        for i in range(2):  # для каждого класса
            class_data = self.X[self.y == i]
            color = 'blue' if i == 0 else 'red'
            label = f'Класс {i}'
            
            # Рисуем точки
            plt.scatter(class_data[:, 0], class_data[:, 1], 
                       c=color, label=label, alpha=0.5)
            
            # Рисуем контуры распределений
            self.plot_gaussian_contours(
                self.nb.mean_[i], 
                np.sqrt(self.nb.var_[i]), 
                color
            )
        
        plt.title('Гауссовы распределения по классам')
        plt.legend()
        plt.grid(True)
        
        # 3. Границы принятия решений
        plt.subplot(133)
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        # Получаем предсказания для всех точек сетки
        Z = self.nb.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Отрисовываем границу решений
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(self.X[self.y == 0][:, 0], self.X[self.y == 0][:, 1], 
                   c='blue', label='Класс 0', alpha=0.5)
        plt.scatter(self.X[self.y == 1][:, 0], self.X[self.y == 1][:, 1], 
                   c='red', label='Класс 1', alpha=0.5)
        plt.title('Границы принятия решений')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    visualizer = GaussianNBVisualizer()
    visualizer.visualize() 