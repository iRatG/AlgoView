import numpy as np
import matplotlib.pyplot as plt
from ..base_algorithms.neighbors.knn import KNN

'''
Объяснение:
На каждом шаге мы показываем тестовую точку и обучающие точки.

Вокруг тестовой точки выделяем ближайших соседей (сначала 1, потом 2, потом 3).

Линиями показываем связь тестовой точки с соседями.

В конце выводим итоговое голосование соседей и предсказанный класс.

Такой визуальный разбор позволяет понять, как алгоритм kNN выбирает ближайших соседей и как происходит голосование для классификации.
'''

class KNNVisualizer:
    def __init__(self):
        # Обучающие данные: 2 класса (0 и 1)
        self.X_train = np.array([
            [1, 2], [2, 3], [3, 1],   # Класс 0 (синие точки)
            [6, 5], [7, 7], [8, 6]    # Класс 1 (красные точки)
        ])
        self.y_train = np.array([0, 0, 0, 1, 1, 1])
        
        # Тестовая точка, которую хотим классифицировать
        self.X_test = np.array([[5, 5]])
        self.k = 3
        self.knn = KNN(k=self.k)
        self.knn.fit(self.X_train, self.y_train)

    def plot_knn_step(self, neighbors_idx, step_title):
        plt.figure(figsize=(6,6))
        # Рисуем обучающие точки
        plt.scatter(self.X_train[self.y_train==0][:,0], self.X_train[self.y_train==0][:,1], 
                   c='blue', label='Класс 0')
        plt.scatter(self.X_train[self.y_train==1][:,0], self.X_train[self.y_train==1][:,1], 
                   c='red', label='Класс 1')
        # Рисуем тестовую точку
        plt.scatter(self.X_test[:,0], self.X_test[:,1], c='green', s=150, 
                   label='Тестовая точка', marker='*')
        # Рисуем окружности вокруг тестовой точки, показывая соседей
        for i in neighbors_idx:
            plt.plot([self.X_test[0,0], self.X_train[i,0]], 
                    [self.X_test[0,1], self.X_train[i,1]], 'k--', lw=1)
            plt.scatter(self.X_train[i,0], self.X_train[i,1], 
                       edgecolor='black', s=200, facecolors='none', lw=2)
        plt.legend()
        plt.title(step_title)
        plt.xlim(0, 9)
        plt.ylim(0, 9)
        plt.grid(True)
        plt.show()

    def visualize(self):
        # 1. Начало: просто точки и тестовая точка
        self.plot_knn_step(neighbors_idx=[], step_title="Начальное состояние")

        # 2. Вычисляем расстояния и сортируем
        distances = np.sqrt(np.sum((self.X_train - self.X_test)**2, axis=1))
        sorted_idx = np.argsort(distances)

        # 3. Промежуточный шаг: первые 1 сосед
        self.plot_knn_step(neighbors_idx=sorted_idx[:1], step_title="1-й ближайший сосед")

        # 4. Промежуточный шаг: первые 2 соседа
        self.plot_knn_step(neighbors_idx=sorted_idx[:2], step_title="2 ближайших соседа")

        # 5. Итоговый шаг: первые k=3 соседа и предсказание
        self.plot_knn_step(neighbors_idx=sorted_idx[:self.k], 
                          step_title=f"{self.k} ближайших соседей и итоговое голосование")

        # Получаем предсказание от нашего KNN класса
        predicted_class = self.knn.predict(self.X_test)[0]
        print(f"Предсказанный класс для тестовой точки: {predicted_class}")

if __name__ == "__main__":
    visualizer = KNNVisualizer()
    visualizer.visualize() 