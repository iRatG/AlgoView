import numpy as np

class GaussianNB:
    """
    Наивный байесовский классификатор с гауссовским распределением признаков.
    """

    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Инициализация параметров
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        # Вычисление параметров для каждого класса
        for idx, cls in enumerate(self.classes):
            X_cls = X[y == cls]
            self.mean[idx, :] = X_cls.mean(axis=0)
            self.var[idx, :] = X_cls.var(axis=0)
            self.priors[idx] = len(X_cls) / n_samples

    def _pdf(self, x, mean, var):
        # Вероятность по нормальному распределению
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * ((x - mean) ** 2) / var)

    def _predict_single(self, x):
        posteriors = []

        for idx, cls in enumerate(self.classes):
            # Априорная вероятность класса
            prior = np.log(self.priors[idx])
            
            # Правдоподобие признаков
            likelihood = np.sum(np.log(self._pdf(x, self.mean[idx, :], self.var[idx, :])))
            
            # Апостериорная вероятность (в логарифмическом масштабе)
            posterior = prior + likelihood
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def predict_prob(self, X):
        probas = []
        for x in X:
            posteriors = []
            for idx, cls in enumerate(self.classes):
                prior = np.log(self.priors[idx])
                likelihood = np.sum(np.log(self._pdf(x, self.mean[idx, :], self.var[idx, :])))
                posterior = prior + likelihood
                posteriors.append(posterior)
            
            # Преобразование логарифмических вероятностей в обычные и нормализация
            posteriors = np.array(posteriors)
            posteriors = np.exp(posteriors - np.max(posteriors))
            posteriors = posteriors / np.sum(posteriors)
            probas.append(posteriors)
            
        return np.array(probas) 