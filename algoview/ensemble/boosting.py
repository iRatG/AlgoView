import numpy as np
from ..base_algorithms.trees.regression_tree import DecisionTreeRegressorScratch

class GBMClassifier:
    """
    Градиентный бустинг для задач классификации.
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=10,
                 early_stopping_rounds=None, verbose=False):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        
        self.trees = []
        self.best_iteration = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Инициализация предсказаний
        self.trees = []
        F = np.zeros(len(y_train))
        
        # Для раннего останова
        best_val_loss = float('inf')
        rounds_without_improve = 0
        
        for i in range(self.n_estimators):
            # Вычисление градиентов
            p = self._sigmoid(F)
            grad = p - y_train
            
            # Создание и обучение дерева на градиентах
            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_train, grad)
            
            # Обновление предсказаний
            update = tree.predict(X_train)
            F -= self.learning_rate * update
            
            self.trees.append(tree)
            
            # Проверка на валидационной выборке
            if X_val is not None and y_val is not None and self.early_stopping_rounds:
                val_pred = self.predict_prob(X_val)
                val_loss = -np.mean(y_val * np.log(val_pred) + (1 - y_val) * np.log(1 - val_pred))
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    rounds_without_improve = 0
                    self.best_iteration = i + 1
                else:
                    rounds_without_improve += 1
                    
                if rounds_without_improve >= self.early_stopping_rounds:
                    if self.verbose:
                        print(f'Early stopping at iteration {i + 1}')
                    break
                    
            if self.verbose and (i + 1) % 10 == 0:
                train_pred = self.predict_prob(X_train)
                train_loss = -np.mean(y_train * np.log(train_pred) + (1 - y_train) * np.log(1 - train_pred))
                print(f'Iteration {i + 1}, train loss: {train_loss:.4f}')

    def predict_prob(self, X):
        # Получение предсказаний от всех деревьев
        F = np.zeros(len(X))
        trees_to_use = len(self.trees) if self.best_iteration is None else self.best_iteration
        
        for tree in self.trees[:trees_to_use]:
            F -= self.learning_rate * tree.predict(X)
            
        return self._sigmoid(F)

    def predict(self, X, threshold=0.5):
        probas = self.predict_prob(X)
        return (probas >= threshold).astype(int) 