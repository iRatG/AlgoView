# Структура проекта AlgoView

```
algoview/
├── base_algorithms/              # Уровень 1: Базовые алгоритмы
│   ├── __init__.py
│   ├── linear/                  # Линейные модели
│   │   ├── __init__.py
│   │   ├── logistic.py         # Логистическая регрессия
│   │   └── lda.py              # Линейный дискриминантный анализ
│   ├── trees/                   # Древовидные модели
│   │   ├── __init__.py
│   │   ├── decision_tree.py    # Дерево решений (классификация)
│   │   └── regression_tree.py  # Дерево решений (регрессия)
│   ├── probabilistic/           # Вероятностные модели
│   │   ├── __init__.py
│   │   └── naive_bayes.py      # Наивный Байес (Гауссовский)
│   └── neighbors/               # Модели на основе соседей
│       ├── __init__.py
│       └── knn.py              # K-ближайших соседей
│
├── ensemble/                    # Уровень 2: Ансамбли
│   ├── __init__.py
│   ├── bagging.py              # Bagging и Random Forest
│   ├── boosting.py             # Gradient Boosting Machine
│   └── stacking.py             # Stacking с визуализацией
│
├── metrics/                     # Метрики качества
│   ├── __init__.py
│   ├── classification.py        # Метрики классификации (accuracy, f1, recall и др.)
│   ├── regression.py           # Метрики регрессии (mse, mae и др.)
│   └── tracker.py              # Система отслеживания метрик во время обучения
│
├── visualization/               # Визуализация алгоритмов
│   ├── __init__.py
│   ├── base/                   # Базовые компоненты визуализации
│   │   ├── __init__.py
│   │   ├── plot_utils.py       # Общие утилиты для построения графиков
│   │   └── animation.py        # Базовые классы для анимации
│   ├── linear/                 # Визуализация линейных моделей
│   │   ├── __init__.py
│   │   ├── logistic_viz.py
│   │   └── lda_viz.py
│   ├── trees/                  # Визуализация деревьев
│   │   ├── __init__.py
│   │   └── tree_viz.py
│   └── ensemble/               # Визуализация ансамблей
│       ├── __init__.py
│       ├── bagging_viz.py
│       ├── boosting_viz.py
│       └── stacking_viz.py
│
├── utils/                      # Общие утилиты
│   ├── __init__.py
│   ├── data.py                # Работа с данными (analyze_dataset, feature_summary)
│   ├── preprocessing.py       # Подготовка данных (prepare_data_classic, simple_oversample)
│   └── serialization.py       # Сохранение/загрузка моделей
│
├── tests/                      # Тесты
│   ├── __init__.py
│   ├── test_base_algorithms/
│   ├── test_ensemble/
│   ├── test_metrics/
│   └── test_visualization/
│
├── examples/                   # Примеры использования
│   ├── basic_usage.ipynb
│   ├── advanced_ensembles.ipynb
│   └── visualization_demo.ipynb
│
├── setup.py                    # Установка пакета
├── requirements.txt            # Зависимости
└── README.md                   # Документация
```

## Существующие алгоритмы из source.py:

1. **Базовые алгоритмы**:
   - `LogisticRegression`: Логистическая регрессия
   - `LDA`: Линейный дискриминантный анализ
   - `GaussianNB`: Наивный Байес (Гауссовский)
   - `DecisionTree`: Дерево решений для классификации
   - `DecisionTreeRegressorScratch`: Дерево решений для регрессии
   - `KNN`: К-ближайших соседей

2. **Ансамбли**:
   - `RandomForest`: Случайный лес
   - `GBMClassifier`: Градиентный бустинг
   - `StackingEnsemble`: Базовый стекинг
   - `StackingEnsembleWithViz`: Стекинг с визуализацией
   - `BaggingClassifierEnsemble`: Бэггинг классификатор

3. **Утилиты**:
   - `analyze_dataset`: Анализ набора данных
   - `feature_summary`: Сводка по признакам
   - `feature_selector`: Отбор признаков
   - `simple_oversample`: Простой оверсэмплинг
   - `prepare_data_classic`: Подготовка данных
   - `evaluate_model`: Оценка модели
   - `plot_metrics`: Визуализация метрик

## Особенности структуры:

1. **Модульность**: Каждый компонент в отдельном пакете
2. **Четкое разделение**: Базовые алгоритмы, ансамбли, метрики, визуализация
3. **Единая система метрик**: Централизованный пакет для всех алгоритмов
4. **Гибкая визуализация**: Соответствует структуре алгоритмов

## Взаимодействие компонентов:

1. **Алгоритмы → Метрики**:
   ```python
   from algoview.metrics import ClassificationMetrics
   
   class LogisticRegression:
       def fit(self, X, y):
           self.metric_tracker = ClassificationMetrics()
           # ... обучение ...
           self.metric_tracker.update(y_true, y_pred)
   ```

2. **Алгоритмы → Визуализация**:
   ```python
   from algoview.base_algorithms.linear import LogisticRegression
   from algoview.visualization.linear import LogisticViz
   
   model = LogisticRegression()
   viz = LogisticViz(model)
   viz.plot_decision_boundary()
   ```

3. **Ансамбли → Базовые алгоритмы**:
   ```python
   from algoview.base_algorithms.trees import DecisionTree
   from algoview.ensemble import RandomForest
   
   rf = RandomForest(base_estimator=DecisionTree())
   ```

## Преимущества структуры:

1. **Масштабируемость**: Легко добавлять новые алгоритмы и функциональность
2. **Переиспользование**: Общие компоненты доступны всем модулям
3. **Тестируемость**: Четкое разделение облегчает написание тестов
4. **Понятность**: Логичная организация помогает в обучении 