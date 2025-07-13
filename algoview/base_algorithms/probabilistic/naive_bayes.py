import numpy as np

class GaussianNB:
    """
    Наивный байесовский классификатор с гауссовским распределением признаков.
    """

    def __init__(self):
        self.classes_ = None
        self.mean_ = None
        self.var_ = None
        self.priors_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Инициализация параметров
        self.mean_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        self.priors_ = np.zeros(n_classes)

        # Вычисление параметров для каждого класса
        for idx, cls in enumerate(self.classes_):
            X_cls = X[y == cls]
            self.mean_[idx, :] = X_cls.mean(axis=0)
            self.var_[idx, :] = X_cls.var(axis=0)
            self.priors_[idx] = len(X_cls) / n_samples

    def _pdf(self, x, mean, var):
        # Вероятность по нормальному распределению
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * ((x - mean) ** 2) / var)

    def _predict_single(self, x):
        posteriors = []

        for idx, cls in enumerate(self.classes_):
            # Априорная вероятность класса
            prior = np.log(self.priors_[idx])
            
            # Правдоподобие признаков
            likelihood = np.sum(np.log(self._pdf(x, self.mean_[idx, :], self.var_[idx, :])))
            
            # Апостериорная вероятность (в логарифмическом масштабе)
            posterior = prior + likelihood
            posteriors.append(posterior)

        return self.classes_[np.argmax(posteriors)]

    def predict(self, X):
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def predict_prob(self, X):
        probas = []
        for x in X:
            posteriors = []
            for idx, cls in enumerate(self.classes_):
                prior = np.log(self.priors_[idx])
                likelihood = np.sum(np.log(self._pdf(x, self.mean_[idx, :], self.var_[idx, :])))
                posterior = prior + likelihood
                posteriors.append(posterior)
            
            # Преобразование логарифмических вероятностей в обычные и нормализация
            posteriors = np.array(posteriors)
            posteriors = np.exp(posteriors - np.max(posteriors))
            posteriors = posteriors / np.sum(posteriors)
            probas.append(posteriors)
            
        return np.array(probas) 