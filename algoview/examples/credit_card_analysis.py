import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from ..visualization.logistic_viz import LogisticRegressionVisualizer
from ..visualization.lda_viz import LDAVisualizer
from ..visualization.knn_viz import KNNVisualizer
from ..visualization.decision_tree_viz import DecisionTreeVisualizer
from ..visualization.random_forest_viz import RandomForestVisualizer
from ..visualization.gaussian_nb_viz import GaussianNBVisualizer

class CreditCardAnalyzer:
    def __init__(self):
        # Получаем абсолютный путь к файлу данных
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        data_path = os.path.join(project_root, 'data', 'UCI_Credit_Card.csv')
        
        # Загружаем данные
        self.df = pd.read_csv(data_path)
        
        # Подготовка данных
        X = self.df.drop(['ID', 'default.payment.next.month'], axis=1).values
        y = self.df['default.payment.next.month'].values
        
        # Стандартизация признаков
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Разделение на обучающую и тестовую выборки
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Для визуализации возьмем только два самых важных признака
        # LIMIT_BAL и PAY_AMT1 (лимит по карте и сумма последнего платежа)
        self.X_train_2d = self.X_train[:, [0, 18]]
        self.X_test_2d = self.X_test[:, [0, 18]]

    def visualize_all(self):
        print("Визуализация логистической регрессии...")
        log_viz = LogisticRegressionVisualizer()
        log_viz.X = self.X_train_2d
        log_viz.y = self.y_train
        log_viz.model.fit(log_viz.X, log_viz.y)
        log_viz.visualize()
        
        print("\nВизуализация LDA...")
        lda_viz = LDAVisualizer()
        lda_viz.X = self.X_train_2d
        lda_viz.y = self.y_train
        lda_viz.X1 = self.X_train_2d[self.y_train == 0]
        lda_viz.X2 = self.X_train_2d[self.y_train == 1]
        lda_viz.lda.fit(lda_viz.X, lda_viz.y)
        lda_viz.visualize()
        
        print("\nВизуализация KNN...")
        knn_viz = KNNVisualizer()
        # Возьмем подвыборку для наглядности
        sample_idx = np.random.choice(len(self.X_train_2d), 100, replace=False)
        knn_viz.X_train = self.X_train_2d[sample_idx]
        knn_viz.y_train = self.y_train[sample_idx]
        knn_viz.X_test = self.X_test_2d[:1]
        knn_viz.knn.fit(knn_viz.X_train, knn_viz.y_train)
        knn_viz.visualize()
        
        print("\nВизуализация дерева решений...")
        dt_viz = DecisionTreeVisualizer()
        dt_viz.X = self.X_train_2d[sample_idx]
        dt_viz.y = self.y_train[sample_idx]
        dt_viz.tree.fit(dt_viz.X, dt_viz.y)
        dt_viz.visualize()
        
        print("\nВизуализация случайного леса...")
        rf_viz = RandomForestVisualizer()
        rf_viz.X = self.X_train_2d[sample_idx]
        rf_viz.y = self.y_train[sample_idx]
        rf_viz.forest.fit(rf_viz.X, rf_viz.y)
        rf_viz.visualize()
        
        print("\nВизуализация наивного байесовского классификатора...")
        nb_viz = GaussianNBVisualizer()
        nb_viz.X = self.X_train_2d[sample_idx]
        nb_viz.y = self.y_train[sample_idx]
        nb_viz.nb.fit(nb_viz.X, nb_viz.y)
        nb_viz.visualize()

if __name__ == "__main__":
    analyzer = CreditCardAnalyzer()
    analyzer.visualize_all() 