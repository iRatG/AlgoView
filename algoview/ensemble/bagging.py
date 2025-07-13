import numpy as np
import pickle
from ..base_algorithms.trees.decision_tree import DecisionTree

class RandomForest:
    """
    Случайный лес на основе деревьев решений.
    """
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2, max_features=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.trees = []

        for _ in range(self.n_estimators):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            # Бутстрэп-выборка
            X_sample, y_sample = self._bootstrap_sample(X, y)
            # Обучение дерева
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Голосование деревьев
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [np.bincount(tree_pred).argmax() for tree_pred in tree_preds]
        return np.array(y_pred)

    def predict_prob(self, X):
        # Усреднение вероятностей по всем деревьям
        probas = []
        for tree in self.trees:
            probas.append(tree.predict_prob(X))
        probas = np.array(probas)
        return np.mean(probas, axis=0)

class BaggingClassifierEnsemble:
    """
    Бэггинг классификатор с произвольными базовыми моделями.
    """
    def __init__(self, base_models_with_params, n_estimators=10, random_state=42):
        self.base_models_with_params = base_models_with_params
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.models = []

        # Создание n_estimators копий каждой базовой модели
        for model_class, params in self.base_models_with_params:
            for _ in range(self.n_estimators):
                # Создание новой модели с заданными параметрами
                model = model_class(**params)
                
                # Бутстрэп-выборка
                n_samples = X.shape[0]
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_sample = X[indices]
                y_sample = y[indices]
                
                # Обучение модели
                model.fit(X_sample, y_sample)
                self.models.append(model)

    def predict_prob(self, X):
        # Получение предсказаний от всех моделей
        all_probas = []
        for model in self.models:
            probas = model.predict_prob(X)
            all_probas.append(probas)
        
        # Усреднение вероятностей
        return np.mean(all_probas, axis=0)

    def predict(self, X, threshold=0.5):
        probas = self.predict_prob(X)
        return (probas[:, 1] >= threshold).astype(int)

    def evaluate(self, X_test, y_test):
        from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
        
        y_pred = self.predict(X_test)
        y_prob = self.predict_prob(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
        
        return metrics

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.models, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            models = pickle.load(f)
        
        # Создание нового экземпляра с пустыми параметрами
        ensemble = BaggingClassifierEnsemble([], n_estimators=1)
        ensemble.models = models
        
        return ensemble 