# coding: utf-8

 

# In[ ]:

 

import pickle

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

 

 

# In[ ]:

 

def analyze_dataset(df, target=None, categorical_threshold=10):

    """

    Универсальный инструмент для анализа набора данных.

    Выводит статистику, визуализации и анализ признаков.

    """

    sns.set(style="whitegrid")

    print("=== Общая информация о данных ===")

    print(f"Размер данных: {df.shape}")

    print("\nТипы данных и пропуски:")

    print(df.info())

   

    print("\nСтатистика по числовым признакам:")

    print(df.describe())

   

    print("\nКоличество пропусков по столбцам:")

    print(df.isnull().sum())

   

    categorical_cols = [col for col in df.columns if df[col].nunique() <= categorical_threshold and df[col].dtype != 'float64']

    numerical_cols = [col for col in df.columns if col not in categorical_cols and df[col].dtype in ['int64', 'float64']]

   

    if target is not None:

        if target in categorical_cols:

            categorical_cols.remove(target)

        if target in numerical_cols:

            numerical_cols.remove(target)

   

    print(f"\nКатегориальные признаки (<= {categorical_threshold} уникальных значений): {categorical_cols}")

    print(f"Числовые признаки: {numerical_cols}")

   

    for col in categorical_cols:

        plt.figure(figsize=(6,4))

        sns.countplot(x=col, data=df)

        plt.title(f'Распределение категориального признака: {col}')

        plt.xticks(rotation=45)

        plt.tight_layout()

        plt.show()

   

    for col in numerical_cols:

        plt.figure(figsize=(6,4))

        plt.hist(df[col].dropna(), bins=30, alpha=0.7, color='blue', edgecolor='black')

        plt.title(f'Распределение числового признака: {col}')

        plt.xlabel(col)

        plt.ylabel('Частота')

        plt.tight_layout()

        plt.show()

   

    if len(numerical_cols) > 1:

        plt.figure(figsize=(10,8))

        corr = df[numerical_cols].corr()

        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')

        plt.title('Корреляционная матрица числовых признаков')

        plt.tight_layout()

        plt.show()

   

    if target is not None:

        print(f"\nАнализ связи признаков с целевой переменной: {target}")

        if df[target].dtype in ['int64', 'float64'] and df[target].nunique() <= 10:

            for col in numerical_cols:

                plt.figure(figsize=(6,4))

                sns.boxplot(x=target, y=col, data=df)

                plt.title(f'{col} по классам {target}')

                plt.tight_layout()

                plt.show()

            for col in categorical_cols:

                plt.figure(figsize=(6,4))

                sns.countplot(x=col, hue=target, data=df)

                plt.title(f'{col} распределение по классам {target}')

                plt.xticks(rotation=45)

                plt.tight_layout()

                plt.show()

        else:

            for col in numerical_cols:

                plt.figure(figsize=(6,4))

                sns.scatterplot(x=col, y=target, data=df)

                plt.title(f'Зависимость {target} от {col}')

                plt.tight_layout()

                plt.show()

    print("\n=== Анализ завершен ===")

 

def feature_summary(df, target=None, categorical_threshold=10):

    """

    Формирует сводную таблицу метрик по признакам.

    """

    summary = []

    n = len(df)

 

    for col in df.columns:

        unique_vals = df[col].nunique(dropna=True)

        missing_ratio = df[col].isnull().mean()

        dtype = 'categorical' if (unique_vals <= categorical_threshold and df[col].dtype != 'float64') else 'numerical'

 

        corr = np.nan

        mean = np.nan

        var = np.nan

        freq_ratio = np.nan

 

        if dtype == 'numerical':

            mean = df[col].mean()

            var = df[col].var()

            if target is not None and col != target and pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[target]):

                corr = df[[col, target]].corr().iloc[0,1]

        else:

            freq_ratio = df[col].value_counts(normalize=True, dropna=True).max()

 

        summary.append({

            'feature': col,

            'type': dtype,

            'unique_values': unique_vals,

            'missing_ratio': missing_ratio,

            'correlation_with_target': corr,

            'mean': mean,

            'variance': var,

            'max_category_freq': freq_ratio

        })

 

    summary_df = pd.DataFrame(summary)

    return summary_df

 

def feature_selector(summary_df,

                     target_col,

                     max_missing_ratio=0.1,

                     min_variance=1e-5,

                     corr_threshold=0.1,

                     max_unique_cat=15,

                     exclude_features=None):

    if exclude_features is None:

        exclude_features = []

 

    df = summary_df[~summary_df['feature'].isin([target_col] + exclude_features)].copy()

 

    selected_features = []

 

    for _, row in df.iterrows():

        feat = row['feature']

        ftype = row['type']

        missing = row['missing_ratio']

        var = row['variance']

        corr = row['correlation_with_target']

        unique_vals = row['unique_values']

 

        if missing > max_missing_ratio:

            continue

 

        if ftype == 'numerical':

            if var < min_variance:

                continue

            if pd.isnull(corr) or abs(corr) < corr_threshold:

                continue

            selected_features.append(feat)

 

        elif ftype == 'categorical':

            if unique_vals > max_unique_cat:

                continue

            selected_features.append(feat)

 

    return selected_features

 

 

# Пример интеграции и запуска:

 

def full_analysis_pipeline(df, target, categorical_threshold=10,

                           max_missing_ratio=0.1, min_variance=1e-5,

                           corr_threshold=0.1, max_unique_cat=15,

                           exclude_features=None):

    """

    Полный пайплайн анализа и отбора признаков.

    """

    analyze_dataset(df, target=target, categorical_threshold=categorical_threshold)

    summary_df = feature_summary(df, target=target, categorical_threshold=categorical_threshold)

    selected_features = feature_selector(summary_df,

                                         target_col=target,

                                         max_missing_ratio=max_missing_ratio,

                                         min_variance=min_variance,

                                         corr_threshold=corr_threshold,

                                         max_unique_cat=max_unique_cat,

                                         exclude_features=exclude_features)

    print("\nОтобранные признаки для дальнейшей работы:")

    print(selected_features)

    return selected_features, summary_df

 

# Использование:

#selected_feats, summary = full_analysis_pipeline(dataframe, target='default.payment.next.month', exclude_features=['ID'])

 

 

# In[ ]:

 

import pandas as pd

 

# Загрузите датасет в переменную df

df = pd.read_csv('UCI_Credit_Card.csv')

 

# Предполагается, что функции analyze_dataset, feature_summary, feature_selector и full_analysis_pipeline уже определены

 

# Запускаем анализ и отбор признаков

selected_features, summary_df = full_analysis_pipeline(df, target='default.payment.next.month', exclude_features=['ID'])

 

print("Отобранные признаки:")

print(selected_features)

 

 

# # Этап 2

 

# In[ ]:

 

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

 

def simple_oversample(X, y, random_state=42):

    """

    Простое увеличение миноритарного класса путем повторения случайных примеров.

    """

    np.random.seed(random_state)

    unique, counts = np.unique(y, return_counts=True)

    class_counts = dict(zip(unique, counts))

 

    # Находим миноритарный и мажоритарный классы

    minority_class = unique[np.argmin(counts)]

    majority_class = unique[np.argmax(counts)]

 

    n_majority = class_counts[majority_class]

    n_minority = class_counts[minority_class]

 

    # Индексы миноритарного класса

    minority_indices = np.where(y == minority_class)[0]

 

    # Сколько нужно добавить примеров миноритарного класса

    n_to_add = n_majority - n_minority

 

    # Случайно выбираем с возвращением индексы для добавления

    add_indices = np.random.choice(minority_indices, size=n_to_add, replace=True)

 

    # Создаем новые X и y с добавленными примерами

    X_oversampled = np.vstack([X, X[add_indices]])

    y_oversampled = np.hstack([y, y[add_indices]])

 

    return X_oversampled, y_oversampled

 

def prepare_data_classic(df, selected_features, target_col):

    """

    Подготовка данных:

    - one-hot кодирование категориальных признаков

    - масштабирование числовых признаков

    - проверка баланса классов и классический oversampling при дисбалансе

    """

 

    categorical_feats = [col for col in selected_features if df[col].dtype == 'object' or (df[col].nunique() <= 15 and df[col].dtype != 'float64')]

    numerical_feats = [col for col in selected_features if col not in categorical_feats]

 

    # One-hot кодирование

    df_cat = pd.get_dummies(df[categorical_feats], drop_first=True) if categorical_feats else pd.DataFrame(index=df.index)

 

    # Масштабирование числовых

    scaler = StandardScaler()

    df_num = pd.DataFrame(scaler.fit_transform(df[numerical_feats]), columns=numerical_feats, index=df.index) if numerical_feats else pd.DataFrame(index=df.index)

 

    # Объединяем

    X = pd.concat([df_num, df_cat], axis=1).values

    y = df[target_col].values

 

    # Проверяем баланс классов

    unique, counts = np.unique(y, return_counts=True)

    class_distribution = dict(zip(unique, counts))

    print(f"Распределение классов: {class_distribution}")

 

    minority_ratio = min(counts) / max(counts)

    if minority_ratio < 0.4:

        print("Дисбаланс обнаружен — применяем классический oversampling...")

        X_res, y_res = simple_oversample(X, y)

        print(f"Новый размер выборки: {X_res.shape[0]}")

        return X_res, y_res

    else:

        print("Дисбаланс незначительный — oversampling не применяется.")

        return X, y

 

 

# In[ ]:

 

X_prepared, y_prepared = prepare_data_classic(df, selected_features, target_col='default.payment.next.month')

 

 

# # Следующий шаг

 

# In[ ]:

 

from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, precision_recall_curve, auc

import matplotlib.pyplot as plt

 

def evaluate_model(y_true, y_pred, y_prob=None, model_name='Model'):

    acc = accuracy_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred)

    recall = recall_score(y_true, y_pred)

    auc_roc = roc_auc_score(y_true, y_prob) if y_prob is not None else None

 

    print(f"{model_name} — Accuracy: {acc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, AUC-ROC: {auc_roc:.4f}" if auc_roc else

          f"{model_name} — Accuracy: {acc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}")

 

    if y_prob is not None:

        # ROC curve

        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y_true, y_prob)

        plt.figure(figsize=(12,5))

 

        plt.subplot(1,2,1)

        plt.plot(fpr, tpr, label=f'{model_name} (AUC={auc_roc:.3f})')

        plt.plot([0,1],[0,1],'k--')

        plt.xlabel('False Positive Rate')

        plt.ylabel('True Positive Rate')

        plt.title('ROC Curve')

        plt.legend()

 

        # Precision-Recall curve

        precision, recall_vals, _ = precision_recall_curve(y_true, y_prob)

        pr_auc = auc(recall_vals, precision)

 

        plt.subplot(1,2,2)

        plt.plot(recall_vals, precision, label=f'{model_name} (AUC={pr_auc:.3f})')

        plt.xlabel('Recall')

        plt.ylabel('Precision')

        plt.title('Precision-Recall Curve')

        plt.legend()

 

        plt.tight_layout()

        plt.show()

 

        # Возвращаем метрики в словаре

    return {

        'Accuracy': acc,

        'F1': f1,

        'Recall': recall,

        'AUC-ROC': auc_roc

    }

 

 

# # Данные.

 

# In[ ]:

 

# 1. Подготовка данных (пример с классическим oversampling)

X_prepared, y_prepared = prepare_data_classic(df, selected_features, target_col='default.payment.next.month')

 

# 2. Разбиение на train/test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    X_prepared, y_prepared, test_size=0.2, random_state=42, stratify=y_prepared

)

 

 

# # Логистическая регрессия

 

# In[ ]:

 

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

 

            # Обновление параметров

            self.weights -= self.lr * dw

            self.bias -= self.lr * db

 

            if self.verbose and i % (self.n_iters // 10) == 0:

                loss = -np.mean(y * np.log(y_predicted + 1e-15) + (1 - y) * np.log(1 - y_predicted + 1e-15))

                print(f"Итерация {i}, loss: {loss:.4f}")

 

    def predict_prob(self, X):

        linear_model = np.dot(X, self.weights) + self.bias

        return self._sigmoid(linear_model)

 

    def predict(self, X, threshold=0.5):

        probs = self.predict_prob(X)

        return (probs >= threshold).astype(int)

 

 

# In[ ]:

 

 

 

 

 

 

# # Линейный дискриминантный анализ (LDA)

 

# In[ ]:

 

import numpy as np

 

class LDA:

    """

    Линейный дискриминантный анализ для бинарной классификации.

    Предполагает нормальное распределение классов с общим ковариационным матрицей.

    """

 

    def __init__(self):

        self.means_ = None        # Средние по классам

        self.priors_ = None       # Приоритеты классов (априорные вероятности)

        self.cov_inv_ = None      # Обратная ковариационная матрица

        self.classes_ = None

 

    def fit(self, X, y):

        self.classes_ = np.unique(y)

        n_features = X.shape[1]

        self.means_ = {}

        self.priors_ = {}

        cov = np.zeros((n_features, n_features))

 

        for cls in self.classes_:

            X_c = X[y == cls]

            self.means_[cls] = np.mean(X_c, axis=0)

            self.priors_[cls] = X_c.shape[0] / X.shape[0]

            cov += (X_c - self.means_[cls]).T @ (X_c - self.means_[cls])

 

        cov /= (X.shape[0] - len(self.classes_))

        self.cov_inv_ = np.linalg.pinv(cov)  # псевдообратная для устойчивости

 

    def _discriminant_score(self, x, cls):

        mean = self.means_[cls]

        prior = self.priors_[cls]

        # Линейная дискриминантная функция

        return x @ self.cov_inv_ @ mean - 0.5 * mean.T @ self.cov_inv_ @ mean + np.log(prior)

 

    def predict(self, X):

        y_pred = []

        for x in X:

            scores = [self._discriminant_score(x, cls) for cls in self.classes_]

            y_pred.append(self.classes_[np.argmax(scores)])

        return np.array(y_pred)

 

    def predict_prob(self, X):

        # Для простоты возвращаем нормализованные экспоненты дискриминантных функций

        probs = []

        for x in X:

            scores = np.array([self._discriminant_score(x, cls) for cls in self.classes_])

            exp_scores = np.exp(scores - np.max(scores))  # стабилизация

            probs.append(exp_scores / exp_scores.sum())

        return np.array(probs)[:, 1]  # вероятность класса 1

 

 

# # Наивный Байес (GaussianNB)

 

# In[ ]:

 

import numpy as np

 

class GaussianNB:

    """

    Наивный Байес с предположением нормального распределения признаков.

    """

 

    def __init__(self):

        self.classes_ = None

        self.priors_ = None

        self.means_ = None

        self.vars_ = None

 

    def fit(self, X, y):

        self.classes_ = np.unique(y)

        n_features = X.shape[1]

        self.means_ = {}

        self.vars_ = {}

        self.priors_ = {}

 

        for cls in self.classes_:

            X_c = X[y == cls]

            self.means_[cls] = np.mean(X_c, axis=0)

            self.vars_[cls] = np.var(X_c, axis=0) + 1e-9  # для устойчивости

            self.priors_[cls] = X_c.shape[0] / X.shape[0]

 

    def _pdf(self, x, mean, var):

        # Вероятность по нормальному распределению

        coeff = 1.0 / np.sqrt(2.0 * np.pi * var)

        exponent = np.exp(- (x - mean) ** 2 / (2 * var))

        return coeff * exponent

 

    def _predict_single(self, x):

        posteriors = []

        for cls in self.classes_:

            prior = np.log(self.priors_[cls])

            conditional = np.sum(np.log(self._pdf(x, self.means_[cls], self.vars_[cls])))

            posterior = prior + conditional

            posteriors.append(posterior)

       return self.classes_[np.argmax(posteriors)]

 

    def predict(self, X):

        return np.array([self._predict_single(x) for x in X])

 

    def predict_prob(self, X):

        probs = []

        for x in X:

            posteriors = []

            for cls in self.classes_:

                prior = np.log(self.priors_[cls])

                conditional = np.sum(np.log(self._pdf(x, self.means_[cls], self.vars_[cls])))

                posteriors.append(prior + conditional)

            max_log = max(posteriors)

            exp_post = np.exp(np.array(posteriors) - max_log)

            prob = exp_post / exp_post.sum()

            probs.append(prob)

        probs = np.array(probs)

        return probs[:, 1]  # вероятность класса 1

 

 

# # Дерево решений (Decision Tree) — упрощённая реализация

 

# In[ ]:

 

import numpy as np

 

class DecisionTree:

    """

    Простейшее бинарное дерево решений с максимальной глубиной и минимальным количеством объектов в листе.

    Использует критерий Джини для разбиения.

    """

 

    class Node:

        def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):

            self.gini = gini

            self.num_samples = num_samples

            self.num_samples_per_class = num_samples_per_class

            self.predicted_class = predicted_class

            self.feature_index = None

            self.threshold = None

            self.left = None

            self.right = None

 

    def __init__(self, max_depth=5, min_samples_split=2):

        self.max_depth = max_depth

        self.min_samples_split = min_samples_split

        self.root = None

 

    def _gini(self, y):

        m = len(y)

        if m == 0:

            return 0

        counts = np.bincount(y)

        prob = counts / m

        return 1.0 - np.sum(prob ** 2)

 

    def _best_split(self, X, y):

        m, n = X.shape

        if m < self.min_samples_split:

            return None, None

 

        best_gini = 1.0

        best_idx, best_thr = None, None

 

        for idx in range(n):

            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))

            num_left = [0] * 2

            num_right = np.bincount(classes, minlength=2).tolist()

 

            for i in range(1, m):

                c = classes[i - 1]

                num_left[c] += 1

                num_right[c] -= 1

 

                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in [0, 1] if i > 0)

                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in [0, 1] if (m - i) > 0)

 

                gini = (i * gini_left + (m - i) * gini_right) / m

 

                if thresholds[i] == thresholds[i - 1]:

                    continue

 

                if gini < best_gini:

                    best_gini = gini

                    best_idx = idx

                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

 

        return best_idx, best_thr

 

    def _build_tree(self, X, y, depth=0):

        num_samples_per_class = [np.sum(y == i) for i in [0, 1]]

        predicted_class = np.argmax(num_samples_per_class)

        node = self.Node(

            gini=self._gini(y),

            num_samples=len(y),

            num_samples_per_class=num_samples_per_class,

            predicted_class=predicted_class,

        )

 

        if depth < self.max_depth:

           idx, thr = self._best_split(X, y)

            if idx is not None:

                indices_left = X[:, idx] < thr

                X_left, y_left = X[indices_left], y[indices_left]

                X_right, y_right = X[~indices_left], y[~indices_left]

                if len(y_left) > 0 and len(y_right) > 0:

                    node.feature_index = idx

                    node.threshold = thr

                    node.left = self._build_tree(X_left, y_left, depth + 1)

                    node.right = self._build_tree(X_right, y_right, depth + 1)

        return node

 

    def fit(self, X, y):

        self.root = self._build_tree(X, y)

 

    def _predict_sample(self, x, node):

        if node.left is None and node.right is None:

            return node.predicted_class

        if x[node.feature_index] < node.threshold:

            return self._predict_sample(x, node.left)

        else:

            return self._predict_sample(x, node.right)

 

    def predict(self, X):

        return np.array([self._predict_sample(x, self.root) for x in X])

 

    def predict_prob(self, X):

        # Для простоты возвращаем 1.0 для предсказанного класса, 0.0 для другого

        preds = self.predict(X)

        return preds  # можно расширить для вероятностей

 

 

# # k-ближайших соседей (k-NN)

 

# In[ ]:

 

import numpy as np

import time

 

class KNN:

    """

    Классический k-ближайших соседей с евклидовым расстоянием.

    Добавлен прогресс-лог и замер времени для удобства.

    """

 

    def __init__(self, k=5, verbose=False):

        self.k = k

        self.X_train = None

        self.y_train = None

        self.verbose = verbose

 

    def fit(self, X, y):

        self.X_train = X

        self.y_train = y

 

    def _distance(self, x1, x2):

        return np.sqrt(np.sum((x1 - x2) ** 2))

 

    def predict(self, X):

        y_pred = []

        n_samples = X.shape[0]

        start_time = time.time()

 

        for i, x in enumerate(X):

            distances = np.array([self._distance(x, x_train) for x_train in self.X_train])

            idx = np.argsort(distances)[:self.k]

            nearest_labels = self.y_train[idx]

            counts = np.bincount(nearest_labels)

            y_pred.append(np.argmax(counts))

 

            if self.verbose and (i + 1) % 100 == 0:

                elapsed = time.time() - start_time

                speed = (i + 1) / elapsed

                remaining = (n_samples - (i + 1)) / speed

                print(f"Обработано {i+1}/{n_samples} ({(i+1)/n_samples*100:.1f}%), "

                      f"скорость: {speed:.2f} объектов/сек, "

                      f"примерно осталось: {remaining:.1f} сек")

 

        return np.array(y_pred)

 

    def predict_prob(self, X):

        probs = []

        n_samples = X.shape[0]

        start_time = time.time()

 

        for i, x in enumerate(X):

            distances = np.array([self._distance(x, x_train) for x_train in self.X_train])

            idx = np.argsort(distances)[:self.k]

            nearest_labels = self.y_train[idx]

            prob = np.sum(nearest_labels == 1) / self.k

            probs.append(prob)

 

            if self.verbose and (i + 1) % 100 == 0:

                elapsed = time.time() - start_time

                speed = (i + 1) / elapsed

                remaining = (n_samples - (i + 1)) / speed

                print(f"Обработано {i+1}/{n_samples} ({(i+1)/n_samples*100:.1f}%), "

                      f"скорость: {speed:.2f} объектов/сек, "

                      f"примерно осталось: {remaining:.1f} сек")

 

        return np.array(probs)

 

 

# # Случайный лес (Random Forest) — упрощённый

 

# In[ ]:

 

import numpy as np

 

class RandomForest:

    """

    Упрощённый случайный лес из деревьев решений.

   """

 

    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2, max_features=None, random_state=42):

        self.n_estimators = n_estimators

        self.max_depth = max_depth

        self.min_samples_split = min_samples_split

        self.max_features = max_features

        self.trees = []

        self.random_state = random_state

        np.random.seed(self.random_state)

 

    def _bootstrap_sample(self, X, y):

        n_samples = X.shape[0]

        indices = np.random.choice(n_samples, size=n_samples, replace=True)

        return X[indices], y[indices]

 

    def fit(self, X, y):

        self.trees = []

        for _ in range(self.n_estimators):

            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)

            X_sample, y_sample = self._bootstrap_sample(X, y)

            if self.max_features is not None:

                feat_indices = np.random.choice(X.shape[1], self.max_features, replace=False)

                tree.feature_indices = feat_indices

                tree.fit(X_sample[:, feat_indices], y_sample)

            else:

                tree.fit(X_sample, y_sample)

                tree.feature_indices = np.arange(X.shape[1])

            self.trees.append(tree)

 

    def predict(self, X):

        # Голосование деревьев

        tree_preds = np.array([tree.predict(X[:, tree.feature_indices]) for tree in self.trees])

        y_pred = []

        for i in range(X.shape[0]):

            counts = np.bincount(tree_preds[:, i])

            y_pred.append(np.argmax(counts))

        return np.array(y_pred)

 

    def predict_prob(self, X):

        tree_preds = np.array([tree.predict(X[:, tree.feature_indices]) for tree in self.trees])

        probs = []

        for i in range(X.shape[0]):

            counts = np.bincount(tree_preds[:, i])

            prob = counts[1] / counts.sum() if len(counts) > 1 else 0.0

            probs.append(prob)

        return np.array(probs)

 

 

# In[ ]:

 

# Предполагается, что у вас есть X_train, y_train, X_test, y_test подготовленные numpy-массивы

 

 

model = LogisticRegression(lr=0.1, n_iters=3000, verbose=True)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_prob = model.predict_prob(X_test)

evaluate_model(y_test, y_pred, y_prob, model_name='Logistic Regression')

# Сохранение Logistic Regression

with open('logistic_regression_model.pkl', 'wb') as f:

    pickle.dump(model, f)

 

# Пример для LDA:

lda = LDA()

lda.fit(X_train, y_train)

y_pred = lda.predict(X_test)

y_prob = lda.predict_prob(X_test)

evaluate_model(y_test, y_pred, y_prob, model_name='LDA')

# Сохранение LDA

with open('lda_model.pkl', 'wb') as f:

    pickle.dump(lda, f)

   

# Аналогично для других моделей:

gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

y_prob = gnb.predict_prob(X_test)

evaluate_model(y_test, y_pred, y_prob, model_name='GaussianNB')

# Сохранение GaussianNB

with open('gaussian_nb_model.pkl', 'wb') as f:

    pickle.dump(gnb, f)

 

   

dt = DecisionTree(max_depth=7)

dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

evaluate_model(y_test, y_pred, y_pred, model_name='DecisionTree')  # у DecisionTree нет predict_prob, передаём y_pred

# Сохранение Decision Tree

with open('decision_tree_model.pkl', 'wb') as f:

    pickle.dump(dt, f)

 

 

 

rf = RandomForest(n_estimators=10, max_depth=7, max_features=int(np.sqrt(X_train.shape[1])))

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

y_prob = rf.predict_prob(X_test)

evaluate_model(y_test, y_pred, y_prob, model_name='RandomForest')

# Сохранение Random Forest

with open('random_forest_model.pkl', 'wb') as f:

    pickle.dump(rf, f)

 

   

    

knn = KNN(k=5, verbose=True)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

y_prob = knn.predict_prob(X_test)

evaluate_model(y_test, y_pred, y_prob, model_name='k-NN')

# Сохранение k-NN

with open('knn_model.pkl', 'wb') as f:

    pickle.dump(knn, f)   

 

 

# In[ ]:

 

 

 

 

# In[ ]:

 

 

 

 

# # 3 этап

 

# In[ ]:

 

## Stacking

 

 

# In[ ]:

 

import numpy as np

import pickle

 

class StackingEnsemble:

    def __init__(self, base_models, meta_model):

        """

        base_models: список обучаемых моделей (классы с fit/predict)

        meta_model: модель для обучения на мета-признаках

        """

        self.base_models = base_models

        self.meta_model = meta_model

        self.is_fitted = False

 

    def fit(self, X_train, y_train, val_size=0.5, random_state=42):

        from sklearn.model_selection import train_test_split

 

        print("Разбиваем данные на train и валидацию для стекинга...")

        X_train1, X_train2, y_train1, y_train2 = train_test_split(

            X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train

        )

 

        print("Обучаем базовые модели на первой части...")

        for model in self.base_models:

            print(f"Обучаем {model.__class__.__name__}...")

            model.fit(X_train1, y_train1)

 

        print("Формируем мета-признаки на второй части...")

        meta_features_train = np.column_stack([model.predict(X_train2) for model in self.base_models])

 

        print("Обучаем мета-модель...")

        self.meta_model.fit(meta_features_train, y_train2)

 

        self.is_fitted = True

        print("Обучение стекинга завершено.")

 

    def predict(self, X):

        if not self.is_fitted:

            raise Exception("Модель не обучена! Вызовите fit() перед predict().")

 

        meta_features = np.column_stack([model.predict(X) for model in self.base_models])

        return self.meta_model.predict(meta_features)

 

    def predict_prob(self, X):

        if not self.is_fitted:

            raise Exception("Модель не обучена! Вызовите fit() перед predict_prob().")

 

        meta_features = np.column_stack([model.predict(X) for model in self.base_models])

        if hasattr(self.meta_model, 'predict_prob'):

            return self.meta_model.predict_prob(meta_features)

        else:

            # Если predict_prob нет, возвращаем None

            return None

 

    def evaluate(self, X_test, y_test):

        y_pred = self.predict(X_test)

        y_prob = self.predict_prob(X_test)

        evaluate_model(y_test, y_pred, y_prob, model_name='Stacking Ensemble')

 

    def save(self, filename):

        with open(filename, 'wb') as f:

            pickle.dump(self, f)

        print(f"Ансамбль сохранён в '{filename}'")

 

    @staticmethod

    def load(filename):

        with open(filename, 'rb') as f:

            ensemble = pickle.load(f)

        print(f"Ансамбль загружен из '{filename}'")

        return ensemble

 

 

# In[ ]:

 

# Создаём базовые модели (без k-NN)

base_models = [

    LogisticRegression(lr=0.1, n_iters=3000, verbose=True),

    LDA(),

    GaussianNB(),

    DecisionTree(max_depth=7),

    RandomForest(n_estimators=10, max_depth=7, max_features=int(np.sqrt(X_train.shape[1])))

]

 

# Мета-модель — логистическая регрессия

meta_model = LogisticRegression(lr=0.1, n_iters=3000)

 

# Создаём ансамбль

stacking = StackingEnsemble(base_models, meta_model)

 

# Обучаем

stacking.fit(X_train, y_train)

 

# Оцениваем

stacking.evaluate(X_test, y_test)

 

# Сохраняем

stacking.save('stacking_ensemble_v1.pkl')

 

# Загрузка и использование

loaded_stacking = StackingEnsemble.load('stacking_ensemble_v1.pkl')

y_pred_loaded = loaded_stacking.predict(X_test)

 

Отличные результаты! Ваш стекинг показал заметное улучшение по сравнению с базовыми моделями:

Accuracy вырос до 70.23%

F1-score — 0.6729

Recall — 0.6125

AUC-ROC — 0.7292

Это говорит о том, что объединение моделей действительно усилило качество предсказаний.

# # Более прогрессивный подход

 

# In[ ]:

 

import numpy as np

import pickle

import matplotlib.pyplot as plt

 

class StackingEnsembleWithViz:

    def __init__(self, base_models, meta_model):

        self.base_models = base_models

        self.meta_model = meta_model

        self.is_fitted = False

        self.metrics_history = {}

 

    def fit(self, X_train, y_train, val_size=0.5, random_state=42):

        from sklearn.model_selection import train_test_split

 

        print("Разбиваем данные на train и валидацию для стекинга...")

        X_train1, X_train2, y_train1, y_train2 = train_test_split(

            X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train

        )

 

        print("Обучаем базовые модели на первой части...")

        for model in self.base_models:

            print(f"Обучаем {model.__class__.__name__}...")

            model.fit(X_train1, y_train1)

 

        print("Формируем мета-признаки на второй части...")

        meta_features_train = []

        for model in self.base_models:

            if hasattr(model, 'predict_prob'):

                probs = model.predict_prob(X_train2)

            else:

                probs = model.predict(X_train2)

            meta_features_train.append(probs)

        meta_features_train = np.column_stack(meta_features_train)

 

        print("Обучаем мета-модель...")

        self.meta_model.fit(meta_features_train, y_train2)

 

        self.is_fitted = True

        print("Обучение стекинга завершено.")

 

    def predict(self, X):

        if not self.is_fitted:

            raise Exception("Модель не обучена! Вызовите fit() перед predict().")

 

        meta_features = []

        for model in self.base_models:

            if hasattr(model, 'predict_prob'):

                probs = model.predict_prob(X)

            else:

                probs = model.predict(X)

            meta_features.append(probs)

        meta_features = np.column_stack(meta_features)

 

        return self.meta_model.predict(meta_features)

 

    def predict_prob(self, X):

        if not self.is_fitted:

            raise Exception("Модель не обучена! Вызовите fit() перед predict_prob().")

 

        meta_features = []

        for model in self.base_models:

            if hasattr(model, 'predict_prob'):

                probs = model.predict_prob(X)

            else:

                probs = model.predict(X)

            meta_features.append(probs)

        meta_features = np.column_stack(meta_features)

 

        if hasattr(self.meta_model, 'predict_prob'):

            return self.meta_model.predict_prob(meta_features)

        else:

            return None

 

    def evaluate(self, X_test, y_test):

        y_pred = self.predict(X_test)

        y_prob = self.predict_prob(X_test)

        metrics = evaluate_model(y_test, y_pred, y_prob, model_name='Stacking Ensemble')

        self.metrics_history = metrics

        return metrics

 

 

    def save(self, filename):

        with open(filename, 'wb') as f:

            pickle.dump(self, f)

        print(f"Ансамбль сохранён в '{filename}'")

 

    @staticmethod

    def load(filename):

        with open(filename, 'rb') as f:

            ensemble = pickle.load(f)

        print(f"Ансамбль загружен из '{filename}'")

        return ensemble

 

 

# In[ ]:

 

# Создаём базовые модели (без k-NN)

base_models = [

    LogisticRegression(lr=0.1, n_iters=3000, verbose=True),

    LDA(),

    GaussianNB(),

    DecisionTree(max_depth=7),

    RandomForest(n_estimators=10, max_depth=7, max_features=int(np.sqrt(X_train.shape[1])))

]

 

# Мета-модель — логистическая регрессия

meta_model = LogisticRegression(lr=0.1, n_iters=3000)

 

# Создаём ансамбль

stacking = StackingEnsembleWithViz(base_models, meta_model)

 

# Обучаем

stacking.fit(X_train, y_train)

 

# Оцениваем и сохраняем метрики

metrics = stacking.evaluate(X_test, y_test)

 

# Сохраняем ансамбль

stacking.save('stacking_ensemble_v2.pkl')

 

# Загрузка и использование

loaded_stacking = StackingEnsembleWithViz.load('stacking_ensemble_v2.pkl')

y_pred_loaded = loaded_stacking.predict(X_test)

 

 

# In[ ]:

 

 

 

 

 

# In[ ]:

 

 

 

 

# In[ ]:

 

 

 

 

# # Бустинг

 

# In[ ]:

 

import numpy as np

 

class Node:

    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):

        self.feature_idx = feature_idx      # Индекс признака для разбиения

        self.threshold = threshold          # Порог разбиения

        self.left = left                    # Левый потомок

        self.right = right                  # Правый потомок

        self.value = value                  # Значение в листе (среднее целевой переменной)

 

class DecisionTreeRegressorScratch:

    def __init__(self, max_depth=3, min_samples_split=10):

        self.max_depth = max_depth

        self.min_samples_split = min_samples_split

        self.root = None

 

    def fit(self, X, y):

        self.root = self._build_tree(X, y, depth=0)

 

    def _build_tree(self, X, y, depth):

        n_samples, n_features = X.shape

        if depth >= self.max_depth or n_samples < self.min_samples_split:

            return Node(value=np.mean(y))

 

        best_feature, best_threshold = self._best_split(X, y, n_features)

        if best_feature is None:

            return Node(value=np.mean(y))

 

        left_idx = X[:, best_feature] < best_threshold

        right_idx = ~left_idx

 

        left_child = self._build_tree(X[left_idx], y[left_idx], depth + 1)

        right_child = self._build_tree(X[right_idx], y[right_idx], depth + 1)

 

        return Node(feature_idx=best_feature, threshold=best_threshold, left=left_child, right=right_child)

 

    def _best_split(self, X, y, n_features):

        best_mse = float('inf')

        best_feature, best_threshold = None, None

 

        for feature in range(n_features):

            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:

                left_idx = X[:, feature] < threshold

                right_idx = ~left_idx

                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:

                    continue

                mse = self._calculate_mse(y[left_idx], y[right_idx])

                if mse < best_mse:

                    best_mse = mse

                    best_feature = feature

                    best_threshold = threshold

        return best_feature, best_threshold

 

    def _calculate_mse(self, left_y, right_y):

        left_mse = np.var(left_y) * len(left_y)

        right_mse = np.var(right_y) * len(right_y)

        return (left_mse + right_mse) / (len(left_y) + len(right_y))

 

    def predict(self, X):

        return np.array([self._predict_sample(x, self.root) for x in X])

 

    def _predict_sample(self, x, node):

        if node.value is not None:

            return node.value

        if x[node.feature_idx] < node.threshold:

            return self._predict_sample(x, node.left)

        else:

            return self._predict_sample(x, node.right)

 

class GBMClassifier:

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=10,

                 early_stopping_rounds=None, verbose=False):

        self.n_estimators = n_estimators

        self.learning_rate = learning_rate

        self.max_depth = max_depth

        self.min_samples_split = min_samples_split

        self.early_stopping_rounds = early_stopping_rounds

        self.verbose = verbose

        self.trees = []

        self.F0 = None

 

    def _sigmoid(self, x):

        return 1 / (1 + np.exp(-x))

 

    def fit(self, X_train, y_train, X_val=None, y_val=None):

        p = np.mean(y_train)

        self.F0 = np.log(p / (1 - p))

        Fm = np.full(shape=y_train.shape, fill_value=self.F0)

 

        best_loss = float('inf')

        rounds_without_improve = 0

 

        for m in range(self.n_estimators):

            p_pred = self._sigmoid(Fm)

            residuals = y_train - p_pred

 

            tree = DecisionTreeRegressorScratch(

                max_depth=self.max_depth,

                min_samples_split=self.min_samples_split

            )

            tree.fit(X_train, residuals)

            update = tree.predict(X_train)

 

            Fm += self.learning_rate * update

            self.trees.append(tree)

 

            if self.verbose and (m + 1) % 10 == 0:

                loss = -np.mean(y_train * np.log(p_pred + 1e-15) + (1 - y_train) * np.log(1 - p_pred + 1e-15))

                print(f'Итерация {m+1}/{self.n_estimators}, LogLoss train: {loss:.4f}')

 

            # Ранняя остановка активна только при переданных валидационных данных и параметре early_stopping_rounds

            if (X_val is not None and y_val is not None and self.early_stopping_rounds is not None):

                val_pred = self.predict_prob(X_val)

                val_loss = -np.mean(y_val * np.log(val_pred + 1e-15) + (1 - y_val) * np.log(1 - val_pred + 1e-15))

 

                if self.verbose and (m + 1) % 10 == 0:

                    print(f'Итерация {m+1}/{self.n_estimators}, LogLoss val: {val_loss:.4f}')

 

                if val_loss < best_loss:

                    best_loss = val_loss

                    rounds_without_improve = 0

                else:

                    rounds_without_improve += 1

 

                if rounds_without_improve >= self.early_stopping_rounds:

                    if self.verbose:

                        print(f'Ранняя остановка на итерации {m+1}')

                    break

 

    def predict_prob(self, X):

        Fm = np.full(shape=(X.shape[0],), fill_value=self.F0)

        for tree in self.trees:

            Fm += self.learning_rate * tree.predict(X)

        return self._sigmoid(Fm)

 

    def predict(self, X, threshold=0.5):

        probs = self.predict_prob(X)

        return (probs >= threshold).astype(int)

 

 

# In[ ]:

 

from sklearn.model_selection import train_test_split

 

# Исходное разбиение на train и test

X_train_full, X_test, y_train_full, y_test = train_test_split(

    X_prepared, y_prepared, test_size=0.2, random_state=42, stratify=y_prepared

)

 

# Дополнительное разбиение train на train и val

X_train, X_val, y_train, y_val = train_test_split(

    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full

)

# Обучение с ранней остановкой

model = GBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=3,

                      early_stopping_rounds=20, verbose=True)

 

# Если у вас нет валидационного набора, просто вызывайте без него:

# model.fit(X_train, y_train)

 

# Если есть валидационный набор:

model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

 

# Предсказания и оценка

y_pred = model.predict(X_test)

y_prob = model.predict_prob(X_test)

evaluate_model(y_test, y_pred, y_prob, model_name='GBMClassifier Improved')

   

# Сохранение модели

import pickle

with open('gbm_classifier_model.pkl', 'wb') as f:

    pickle.dump(model, f)

 

GBMClassifier Improved — Accuracy: 0.7062, F1: 0.6646, Recall: 0.5823, AUC-ROC: 0.7660

# In[ ]:

 

 

 

 

# # Бэггинг

 

# In[ ]:

 

# Список базовых моделей и их параметров

base_models_with_params = [

    (LogisticRegression, {'lr': 0.1, 'n_iters': 3000, 'verbose': False}),

    (LDA, {}),

    (GaussianNB, {}),

    (DecisionTree, {'max_depth': 7}),

    (RandomForest, {'n_estimators': 10, 'max_depth': 7, 'max_features': int(np.sqrt(X_train.shape[1]))})

]

 

# Класс ансамбля, который умеет обучать разные модели (нужно реализовать)

class BaggingClassifierEnsemble:

    def __init__(self, base_models_with_params, n_estimators=10, random_state=42):

        self.base_models_with_params = base_models_with_params

        self.n_estimators = n_estimators

        self.random_state = random_state

        self.models = []

        self.is_fitted = False

 

    def fit(self, X, y):

        np.random.seed(self.random_state)

        self.models = []

 

        print("Обучаем ансамбль бэггинг с разными базовыми моделями...")

        for model_class, params in self.base_models_with_params:

            print(f"Обучаем модели класса {model_class.__name__}...")

            for i in range(self.n_estimators):

               indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)

                X_sample = X[indices]

                y_sample = y[indices]

                model = model_class(**params)

                model.fit(X_sample, y_sample)

                self.models.append(model)

        self.is_fitted = True

        print("Обучение завершено.")

 

    def predict_prob(self, X):

        if not self.is_fitted:

            raise Exception("Модель не обучена!")

        probs = []

        for model in self.models:

            if hasattr(model, 'predict_prob'):

                probs.append(model.predict_prob(X))

            else:

                preds = model.predict(X)

                probs.append(preds)

        avg_probs = np.mean(probs, axis=0)

        return avg_probs

 

    def predict(self, X, threshold=0.5):

        avg_probs = self.predict_prob(X)

        return (avg_probs >= threshold).astype(int)

 

    def evaluate(self, X_test, y_test):

        y_pred = self.predict(X_test)

        y_prob = self.predict_prob(X_test)

        evaluate_model(y_test, y_pred, y_prob, model_name='BaggingClassifierEnsemble')

 

    def save(self, filename):

        import pickle

        with open(filename, 'wb') as f:

            pickle.dump(self, f)

        print(f"Ансамбль сохранён в '{filename}'")

 

    @staticmethod

    def load(filename):

        import pickle

        with open(filename, 'rb') as f:

            ensemble = pickle.load(f)

        print(f"Ансамбль загружен из '{filename}'")

        return ensemble

 

# Создаём и обучаем ансамбль

bagging_ensemble = BaggingClassifierEnsemble(base_models_with_params, n_estimators=10, random_state=42)

bagging_ensemble.fit(X_train, y_train)

 

# Оцениваем

bagging_ensemble.evaluate(X_test, y_test)

 

# Сохраняем

bagging_ensemble.save('bagging_classifier_ensemble.pkl')

 

# Загружаем и используем

loaded_ensemble = BaggingClassifierEnsemble.load('bagging_classifier_ensemble.pkl')

y_pred_loaded = loaded_ensemble.predict(X_test)

 

 

# In[ ]:

 

 

 

 

# # Сравнение всех моделей

 

# In[ ]:

 

import pickle

import matplotlib.pyplot as plt

import numpy as np

 

# Список файлов с сохранёнными моделями и их имена для отображения

model_files = {

    'Logistic Regression'     : 'logistic_regression_model.pkl',

    'LDA'                     : 'lda_model.pkl',

    'GaussianNB'              : 'gaussian_nb_model.pkl',

    'DecisionTree'            : 'decision_tree_model.pkl',

    'RandomForest'            : 'random_forest_model.pkl',

    'Stacking Ensemble Sample': 'stacking_ensemble_v1.pkl',  

    'Stacking Ensemble Prob'  : 'stacking_ensemble_v2.pkl',  

    'GBM classifier'          : 'gbm_classifier_model.pkl',

    'Bagging'                 : 'bagging_classifier_ensemble.pkl'

}

 

metrics_results = {}

 

for name, filepath in model_files.items():

    with open(filepath, 'rb') as f:

        model = pickle.load(f)

    y_pred = model.predict(X_test)

    y_prob = model.predict_prob(X_test) if hasattr(model, 'predict_prob') else None

    metrics = evaluate_model(y_test, y_pred, y_prob, model_name=name)

    metrics_results[name] = metrics

 

   

 

 

def plot_metrics(metrics_dict):

    """

    Визуализация метрик для нескольких моделей.

    metrics_dict: словарь вида

        {

            'ModelName1': {'Accuracy': 0.7, 'F1': 0.65, 'Recall': 0.6, 'AUC-ROC': 0.72},

            'ModelName2': {...},

            ...

        }

    """

    metrics = list(next(iter(metrics_dict.values())).keys())

    models = list(metrics_dict.keys())

    x = np.arange(len(models))  # числовые позиции для столбцов

 

    for metric in metrics:

        values = [metrics_dict[m][metric] for m in models]

        plt.figure(figsize=(8, 4))

        plt.bar(x, values, color='skyblue')

        plt.title(f'{metric} для моделей')

        plt.ylabel(metric)

        plt.xticks(x, models, rotation=45, ha='right')  # задаём метки по оси X

        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()

        plt.show()

# Визуализация метрик

plot_metrics(metrics_results)