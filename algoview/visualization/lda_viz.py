import numpy as np
import matplotlib.pyplot as plt
from ..base_algorithms.linear.lda import LDA

class LDAVisualizer:
    def __init__(self):
        self.lda = LDA()

    def plot_step(self, step_title, show_projection=False, show_means=False, show_direction=False):
        plt.figure(figsize=(10, 8))
        
        # Рисуем точки данных
        plt.scatter(self.X[self.y == 0][:, 0], self.X[self.y == 0][:, 1], 
                   c='blue', label='Класс 0', alpha=0.7)
        plt.scatter(self.X[self.y == 1][:, 0], self.X[self.y == 1][:, 1], 
                   c='red', label='Класс 1', alpha=0.7)
        
        if show_means:
            # Рисуем центры классов
            mean0 = np.mean(self.X1, axis=0)
            mean1 = np.mean(self.X2, axis=0)
            plt.scatter(mean0[0], mean0[1], c='blue', s=150, marker='*', label='Центр класса 0')
            plt.scatter(mean1[0], mean1[1], c='red', s=150, marker='*', label='Центр класса 1')
            
            # Рисуем общий центр
            mean_total = np.mean(self.X, axis=0)
            plt.scatter(mean_total[0], mean_total[1], c='green', s=150, marker='*', label='Общий центр')
        
        if show_direction and hasattr(self.lda, 'w'):
            # Рисуем направление проекции (собственный вектор)
            origin = np.mean(self.X, axis=0)
            direction = self.lda.w * 2  # Увеличиваем для наглядности
            plt.arrow(origin[0], origin[1], direction[0], direction[1], 
                     color='green', width=0.05, head_width=0.3, label='Направление проекции')
        
        if show_projection and hasattr(self.lda, 'w'):
            # Проецируем точки на направление LDA
            mean = np.mean(self.X, axis=0)
            for i, point in enumerate(self.X):
                # Вычисляем проекцию
                point_centered = point - mean
                proj = np.dot(point_centered, self.lda.w) * self.lda.w + mean
                plt.plot([point[0], proj[0]], [point[1], proj[1]], 'k--', alpha=0.3)
        
        # Если у нас есть направление проекции, рисуем разделяющую линию
        if hasattr(self.lda, 'w'):
            # Создаем сетку точек
            x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
            y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                np.linspace(y_min, y_max, 100))
            
            # Получаем предсказания для всех точек сетки
            Z = self.lda.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Отрисовываем границу решений
            plt.contour(xx, yy, Z, colors='k', linestyles='--', levels=[0.5])
        
        plt.title(step_title)
        plt.xlabel('Признак 1 (лимит по карте)')
        plt.ylabel('Признак 2 (сумма последнего платежа)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    def visualize(self):
        # 1. Показываем исходные данные
        self.plot_step("1. Исходные данные")
        
        # 2. Показываем центры классов
        self.plot_step("2. Центры классов", show_means=True)
        
        # 3. Показываем направление проекции
        self.plot_step("3. Направление проекции LDA", show_means=True, show_direction=True)
        
        # 4. Показываем проекции точек
        self.plot_step("4. Проекции точек на направление LDA", 
                      show_means=True, show_direction=True, show_projection=True)

if __name__ == "__main__":
    visualizer = LDAVisualizer()
    visualizer.visualize() 