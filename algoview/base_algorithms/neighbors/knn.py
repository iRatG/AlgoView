import numpy as np

class KNN:
    """
    K-ближайших соседей.
    """
    def __init__(self, k=5, verbose=False):
        self.k = k
        self.verbose = verbose
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        predictions = []
        for i, x in enumerate(X):
            # Вычисление расстояний до всех точек обучающей выборки
            distances = [self._distance(x, x_train) for x_train in self.X_train]
            
            # Получение k ближайших соседей
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            
            # Голосование
            most_common = np.bincount(k_nearest_labels).argmax()
            predictions.append(most_common)
            
            if self.verbose and (i + 1) % 100 == 0:
                print(f'Processed {i + 1}/{len(X)} samples')
                
        return np.array(predictions)

    def predict_prob(self, X):
        probas = []
        classes = np.unique(self.y_train)
        
        for x in X:
            # Вычисление расстояний до всех точек обучающей выборки
            distances = [self._distance(x, x_train) for x_train in self.X_train]
            
            # Получение k ближайших соседей
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            
            # Подсчет частот классов среди соседей
            class_counts = np.bincount(k_nearest_labels, minlength=len(classes))
            probas.append(class_counts / self.k)
            
        return np.array(probas) 