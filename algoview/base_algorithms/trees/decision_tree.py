import numpy as np

class Node:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

    @property
    def is_leaf(self):
        return self.left is None and self.right is None

class DecisionTree:
    """
    Дерево решений для задач классификации.
    """
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _gini(self, y):
        # Вычисление индекса Джини
        classes = np.unique(y)
        gini = 0
        for cls in classes:
            p = len(y[y == cls]) / len(y)
            gini += p * (1 - p)
        return gini

    def _best_split(self, X, y):
        # Поиск лучшего разбиения
        m = X.shape[0]
        if m <= self.min_samples_split:
            return None, None

        # Базовый Джини для текущего узла
        parent_gini = self._gini(y)
        best_gini = parent_gini
        best_idx = None
        best_thr = None

        # Перебор всех признаков
        n_features = X.shape[1]
        for idx in range(n_features):
            # Сортировка значений признака
            thresholds = np.unique(X[:, idx])
            
            # Перебор всех возможных порогов
            for thr in thresholds:
                # Разбиение на левую и правую части
                left_mask = X[:, idx] < thr
                right_mask = ~left_mask
                
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue

                # Вычисление Джини для разбиения
                gini_left = self._gini(y[left_mask])
                gini_right = self._gini(y[right_mask])
                gini = (len(y[left_mask]) * gini_left + len(y[right_mask]) * gini_right) / m

                # Обновление лучшего разбиения
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = thr

        return best_idx, best_thr

    def _build_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == c) for c in np.unique(y)]
        predicted_class = np.argmax(num_samples_per_class)
        
        node = Node(
            gini=self._gini(y),
            num_samples=len(y),
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        # Проверка условий остановки
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                
                node.feature_index = idx
                node.threshold = thr
                node.left = self._build_tree(X_left, y_left, depth + 1)
                node.right = self._build_tree(X_right, y_right, depth + 1)
        
        return node

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _predict_sample(self, x, node):
        if node.left is None:
            return node.predicted_class
        
        if x[node.feature_index] < node.threshold:
            return self._predict_sample(x, node.left)
        return self._predict_sample(x, node.right)

    def predict(self, X):
        predictions = [self._predict_sample(x, self.root) for x in X]
        return np.array(predictions)

    def predict_prob(self, X):
        # Для простоты возвращаем 1.0 для предсказанного класса, 0.0 для другого
        predictions = self.predict(X)
        probas = np.zeros((len(X), 2))
        probas[np.arange(len(X)), predictions] = 1.0
        return probas 