import numpy as np
import matplotlib.pyplot as plt
from ..base_algorithms.linear.logistic import LogisticRegression

class LogisticRegressionVisualizer:
    def __init__(self):
        # Генерируем синтетические данные
        np.random.seed(42)
        n_samples = 100
        
        # Генерируем признаки
        X1 = np.random.normal(2, 1, (n_samples // 2, 1))
        X2 = np.random.normal(4, 1, (n_samples // 2, 1))
        self.X = np.vstack([X1, X2])
        
        # Генерируем метки классов
        self.y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
        
        self.model = LogisticRegression(lr=0.01, n_iters=1000)
        self.history = []

    def plot_decision_boundary(self, weights, bias, step):
        plt.figure(figsize=(10, 6))
        
        # Создаем сетку точек
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        # Получаем предсказания для всех точек сетки
        Z = 1 / (1 + np.exp(-(np.dot(np.c_[xx.ravel(), yy.ravel()], weights) + bias)))
        Z = Z.reshape(xx.shape)
        
        # Отрисовываем границу решений
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(self.X[self.y == 0][:, 0], self.X[self.y == 0][:, 1], 
                   c='blue', label='Класс 0', alpha=0.5)
        plt.scatter(self.X[self.y == 1][:, 0], self.X[self.y == 1][:, 1], 
                   c='red', label='Класс 1', alpha=0.5)
        
        # Рисуем разделяющую линию
        plt.contour(xx, yy, Z, levels=[0.5], colors='k', linestyles='--')
        
        plt.title(f'Шаг {step}: Граница принятия решений')
        plt.xlabel('Признак 1 (лимит по карте)')
        plt.ylabel('Признак 2 (сумма последнего платежа)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize(self):
        # Обучаем модель и сохраняем историю весов
        n_samples, n_features = self.X.shape
        weights = np.zeros(n_features)
        bias = 0
        lr = 0.01
        n_iters = 1000
        
        self.history = [(weights.copy(), bias)]
        
        for i in range(n_iters):
            # Прямой проход
            linear_model = np.dot(self.X, weights) + bias
            y_predicted = 1 / (1 + np.exp(-linear_model))
            
            # Градиенты
            dw = (1 / n_samples) * np.dot(self.X.T, (y_predicted - self.y))
            db = (1 / n_samples) * np.sum(y_predicted - self.y)
            
            # Обновление параметров
            weights -= lr * dw
            bias -= lr * db
            
            if i % 200 == 0:
                self.history.append((weights.copy(), bias))
        
        # Выбираем несколько ключевых моментов обучения для визуализации
        steps_to_show = [0, 1, 3, len(self.history)-1]
        
        for step in steps_to_show:
            weights, bias = self.history[step]
            self.plot_decision_boundary(weights, bias, step)
            
        # Показываем график сходимости
        plt.figure(figsize=(10, 6))
        losses = []
        for w, b in self.history:
            y_pred = 1 / (1 + np.exp(-(np.dot(self.X, w) + b)))
            loss = -np.mean(self.y * np.log(y_pred + 1e-10) + 
                          (1 - self.y) * np.log(1 - y_pred + 1e-10))
            losses.append(loss)
        plt.plot(losses)
        plt.title('Сходимость функции потерь')
        plt.xlabel('Итерация')
        plt.ylabel('Функция потерь')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    visualizer = LogisticRegressionVisualizer()
    visualizer.visualize() 