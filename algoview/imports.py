# Стандартные библиотеки Python
import os
import pickle
from typing import List, Dict, Tuple, Optional, Union

# Основные библиотеки для анализа данных
import numpy as np
import pandas as pd

# Библиотеки для визуализации
import matplotlib.pyplot as plt
import seaborn as sns

# Метрики и утилиты sklearn
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    recall_score, 
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc
)

# Импорты из нашего пакета
from .base_algorithms.linear.logistic_regression import LogisticRegression
from .base_algorithms.linear.lda import LDA
from .base_algorithms.trees.decision_tree import DecisionTree
from .base_algorithms.trees.regression_tree import DecisionTreeRegressorScratch
from .base_algorithms.probabilistic.naive_bayes import GaussianNB
from .base_algorithms.neighbors.knn import KNN

from .ensemble.bagging import RandomForest, BaggingClassifierEnsemble
from .ensemble.boosting import GBMClassifier
from .ensemble.stacking import StackingEnsemble, StackingEnsembleWithViz

from .metrics.classification import evaluate_model
from .utils.data import analyze_dataset, feature_summary, feature_selector
from .utils.preprocessing import simple_oversample, prepare_data_classic

# Настройки для визуализации
plt.style.use('seaborn')
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12 