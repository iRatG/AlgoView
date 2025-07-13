import numpy as np

class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeRegressorScratch:
    """
    Дерево решений для задач регрессии.
    """
    def __init__(self, max_depth=3, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        
        # Базовый случай: достигнута максимальная глубина или мало образцов
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            leaf_value = np.mean(y)
            return Node(value=leaf_value)

        # Поиск лучшего разбиения
        best_feature, best_threshold = self._best_split(X, y, n_features)
        
        # Если разбиение не найдено
        if best_feature is None:
            leaf_value = np.mean(y)
            return Node(value=leaf_value)

        # Разбиение данных
        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = ~left_idxs
        
        # Рекурсивное построение поддеревьев
        left = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y, n_features):
        best_feature = None
        best_threshold = None
        best_mse = float('inf')
        
        # Перебор всех признаков
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            # Перебор всех возможных порогов
            for threshold in thresholds:
                left_idxs = X[:, feature] < threshold
                right_idxs = ~left_idxs
                
                if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
                    continue
                
                # Вычисление MSE для текущего разбиения
                mse = self._calculate_mse(y[left_idxs], y[right_idxs])
                
                if mse < best_mse:
                    best_feature = feature
                    best_threshold = threshold
                    best_mse = mse
                    
        return best_feature, best_threshold

    def _calculate_mse(self, left_y, right_y):
        # Вычисление MSE для разбиения
        left_mse = np.mean((left_y - np.mean(left_y)) ** 2) if len(left_y) > 0 else 0
        right_mse = np.mean((right_y - np.mean(right_y)) ** 2) if len(right_y) > 0 else 0
        return left_mse + right_mse

    def predict(self, X):
        predictions = [self._predict_sample(x, self.root) for x in X]
        return np.array(predictions)

    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature_idx] < node.threshold:
            return self._predict_sample(x, node.left)
        return self._predict_sample(x, node.right) 