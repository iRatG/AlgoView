import numpy as np
import pickle

class StackingEnsemble:
    """
    Базовый стекинг ансамбль.
    """
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        self.fitted_base_models = None
        self.fitted_meta_model = None

    def fit(self, X_train, y_train, val_size=0.5, random_state=42):
        # Разделение данных на две части
        np.random.seed(random_state)
        n_samples = len(X_train)
        n_val = int(n_samples * val_size)
        indices = np.random.permutation(n_samples)
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        X_train_base = X_train[train_idx]
        y_train_base = y_train[train_idx]
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]

        # Обучение базовых моделей
        self.fitted_base_models = []
        for model in self.base_models:
            model.fit(X_train_base, y_train_base)
            self.fitted_base_models.append(model)

        # Получение предсказаний базовых моделей на валидационной выборке
        meta_features = np.column_stack([
            model.predict_prob(X_val)[:, 1] for model in self.fitted_base_models
        ])

        # Обучение мета-модели
        self.meta_model.fit(meta_features, y_val)
        self.fitted_meta_model = self.meta_model

    def predict(self, X):
        # Получение предсказаний базовых моделей
        meta_features = np.column_stack([
            model.predict_prob(X)[:, 1] for model in self.fitted_base_models
        ])
        # Предсказание мета-моделью
        return self.fitted_meta_model.predict(meta_features)

    def predict_prob(self, X):
        # Получение предсказаний базовых моделей
        meta_features = np.column_stack([
            model.predict_prob(X)[:, 1] for model in self.fitted_base_models
        ])
        # Предсказание вероятностей мета-моделью
        return self.fitted_meta_model.predict_prob(meta_features)

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
            pickle.dump({
                'base_models': self.fitted_base_models,
                'meta_model': self.fitted_meta_model
            }, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        ensemble = StackingEnsemble(
            base_models=data['base_models'],
            meta_model=data['meta_model']
        )
        ensemble.fitted_base_models = data['base_models']
        ensemble.fitted_meta_model = data['meta_model']
        
        return ensemble

class StackingEnsembleWithViz(StackingEnsemble):
    """
    Стекинг ансамбль с визуализацией.
    """
    def __init__(self, base_models, meta_model):
        super().__init__(base_models, meta_model)
        self.base_predictions = None
        self.meta_predictions = None

    def fit(self, X_train, y_train, val_size=0.5, random_state=42):
        # Разделение данных на две части
        np.random.seed(random_state)
        n_samples = len(X_train)
        n_val = int(n_samples * val_size)
        indices = np.random.permutation(n_samples)
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        X_train_base = X_train[train_idx]
        y_train_base = y_train[train_idx]
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]

        # Обучение базовых моделей и сохранение их предсказаний
        self.fitted_base_models = []
        base_val_predictions = []
        
        for i, model in enumerate(self.base_models):
            print(f"\nОбучение базовой модели {i + 1}/{len(self.base_models)}")
            model.fit(X_train_base, y_train_base)
            self.fitted_base_models.append(model)
            
            # Предсказания на валидационной выборке
            val_pred = model.predict_prob(X_val)[:, 1]
            base_val_predictions.append(val_pred)
            
            # Оценка качества базовой модели
            val_pred_class = (val_pred >= 0.5).astype(int)
            accuracy = np.mean(val_pred_class == y_val)
            print(f"Accuracy базовой модели на валидации: {accuracy:.4f}")

        # Сохранение предсказаний базовых моделей
        self.base_predictions = np.column_stack(base_val_predictions)

        # Обучение мета-модели
        print("\nОбучение мета-модели")
        self.meta_model.fit(self.base_predictions, y_val)
        self.fitted_meta_model = self.meta_model

        # Предсказания мета-модели
        self.meta_predictions = self.meta_model.predict_prob(self.base_predictions)[:, 1]
        meta_accuracy = np.mean((self.meta_predictions >= 0.5).astype(int) == y_val)
        print(f"Accuracy мета-модели на валидации: {meta_accuracy:.4f}")

    def predict(self, X):
        meta_features = np.column_stack([
            model.predict_prob(X)[:, 1] for model in self.fitted_base_models
        ])
        return self.fitted_meta_model.predict(meta_features)

    def predict_prob(self, X):
        meta_features = np.column_stack([
            model.predict_prob(X)[:, 1] for model in self.fitted_base_models
        ])
        return self.fitted_meta_model.predict_prob(meta_features)

    def evaluate(self, X_test, y_test):
        # Получение предсказаний всех моделей
        base_predictions = []
        base_metrics = []
        
        print("\nОценка базовых моделей:")
        for i, model in enumerate(self.fitted_base_models):
            pred_prob = model.predict_prob(X_test)[:, 1]
            pred = (pred_prob >= 0.5).astype(int)
            base_predictions.append(pred_prob)
            
            metrics = {
                'accuracy': np.mean(pred == y_test),
                'roc_auc': roc_auc_score(y_test, pred_prob)
            }
            base_metrics.append(metrics)
            print(f"Модель {i + 1}: Accuracy = {metrics['accuracy']:.4f}, AUC-ROC = {metrics['roc_auc']:.4f}")

        # Оценка мета-модели
        meta_features = np.column_stack(base_predictions)
        meta_pred_prob = self.fitted_meta_model.predict_prob(meta_features)[:, 1]
        meta_pred = (meta_pred_prob >= 0.5).astype(int)
        
        meta_metrics = {
            'accuracy': np.mean(meta_pred == y_test),
            'roc_auc': roc_auc_score(y_test, meta_pred_prob)
        }
        
        print(f"\nМета-модель: Accuracy = {meta_metrics['accuracy']:.4f}, AUC-ROC = {meta_metrics['roc_auc']:.4f}")
        
        return {
            'base_metrics': base_metrics,
            'meta_metrics': meta_metrics
        }

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({
                'base_models': self.fitted_base_models,
                'meta_model': self.fitted_meta_model,
                'base_predictions': self.base_predictions,
                'meta_predictions': self.meta_predictions
            }, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        ensemble = StackingEnsembleWithViz(
            base_models=data['base_models'],
            meta_model=data['meta_model']
        )
        ensemble.fitted_base_models = data['base_models']
        ensemble.fitted_meta_model = data['meta_model']
        ensemble.base_predictions = data['base_predictions']
        ensemble.meta_predictions = data['meta_predictions']
        
        return ensemble 