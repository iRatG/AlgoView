import numpy as np

class LogisticRegression:
    """
    Логистическая регрессия с градиентным спуском.
    Методы:
    - fit(X, y): обучение модели
    - predict_prob(X): предсказание вероятностей
    - predict(X, threshold=0.5): предсказание классов
    """

    def __init__(self, lr=0.01, n_iters=1000, verbose=False):
        self.lr = lr
        self.n_iters = n_iters
        self.verbose = verbose
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        # Сигмоидальная функция
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Инициализация весов и смещения
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iters):
            # Линейная комбинация
            linear_model = np.dot(X, self.weights) + self.bias
            # Прогноз вероятностей
            y_predicted = self._sigmoid(linear_model)

            # Градиенты для весов и смещения
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Обновление весов и смещения
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if self.verbose and (i + 1) % 100 == 0:
                y_pred = self.predict(X)
                accuracy = np.mean(y_pred == y)
                print(f'Iteration {i + 1}/{self.n_iters}, Accuracy: {accuracy:.4f}')

    def predict_prob(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_prob(X)
        return (probabilities >= threshold).astype(int) 