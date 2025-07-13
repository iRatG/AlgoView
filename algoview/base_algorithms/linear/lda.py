import numpy as np

class LDA:
    """
    Линейный дискриминантный анализ.
    """

    def __init__(self):
        self.classes = None
        self.means = None
        self.priors = None
        self.cov = None
        self.w = None  # Направление проекции

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Вычисление средних значений для каждого класса
        self.means = []
        self.priors = []
        
        # Общая ковариационная матрица
        self.cov = np.zeros((n_features, n_features))
        
        for idx, cls in enumerate(self.classes):
            X_cls = X[y == cls]
            self.means.append(np.mean(X_cls, axis=0))
            self.priors.append(len(X_cls) / n_samples)
            
            # Вычисление ковариационной матрицы для класса
            centered_X = X_cls - self.means[idx]
            self.cov += np.dot(centered_X.T, centered_X)
            
        # Усреднение ковариационной матрицы
        self.cov /= n_samples
        
        # Преобразование в массивы numpy
        self.means = np.array(self.means)
        self.priors = np.array(self.priors)
        
        # Вычисление направления проекции (для двух классов)
        if len(self.classes) == 2:
            # w = Σ^(-1) * (μ₁ - μ₀)
            self.w = np.dot(np.linalg.inv(self.cov), (self.means[1] - self.means[0]))
            # Нормализация вектора
            self.w = self.w / np.linalg.norm(self.w)

    def _discriminant_score(self, x, cls):
        # Дискриминантная функция для класса
        mean = self.means[cls]
        prior = self.priors[cls]
        
        # Вычисление дискриминантной функции
        term1 = np.dot(np.dot(x, np.linalg.inv(self.cov)), mean)
        term2 = 0.5 * np.dot(np.dot(mean, np.linalg.inv(self.cov)), mean)
        term3 = np.log(prior)
        
        return term1 - term2 + term3

    def predict(self, X):
        predictions = []
        for x in X:
            # Вычисление дискриминантных функций для каждого класса
            scores = [self._discriminant_score(x, cls) for cls in range(len(self.classes))]
            # Выбор класса с максимальным значением дискриминантной функции
            pred = self.classes[np.argmax(scores)]
            predictions.append(pred)
        return np.array(predictions)

    def predict_prob(self, X):
        # Для простоты возвращаем нормализованные экспоненты дискриминантных функций
        probs = []
        for x in X:
            scores = [self._discriminant_score(x, cls) for cls in range(len(self.classes))]
            exp_scores = np.exp(scores - np.max(scores))  # Вычитаем максимум для численной стабильности
            probs.append(exp_scores / np.sum(exp_scores))
        return np.array(probs) 