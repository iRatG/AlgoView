# AlgoView

AlgoView - это образовательный проект для изучения алгоритмов машинного обучения с нуля. Проект включает реализации популярных алгоритмов классификации и их визуализацию для лучшего понимания принципов работы.

## Реализованные алгоритмы

### Базовые алгоритмы
- Линейные модели
  - Логистическая регрессия
  - Линейный дискриминантный анализ (LDA)
- Деревья решений
  - Классификатор на основе дерева решений
  - Регрессор на основе дерева решений
- Вероятностные модели
  - Наивный байесовский классификатор (Gaussian NB)
- Методы ближайших соседей
  - K-ближайших соседей (KNN)

### Ансамблевые методы
- Бэггинг
  - Случайный лес (Random Forest)
  - Баггинг классификатор
- Бустинг
  - Градиентный бустинг (GBM)
- Стекинг
  - Стекинг ансамбль

## Визуализация

Каждый алгоритм сопровождается визуализатором, который помогает понять:
- Процесс принятия решений
- Границы классификации
- Внутреннее устройство модели
- Распределение данных и их особенности

## Установка

```bash
# Клонирование репозитория
git clone https://github.com/iRatG/AlgoView.git
cd AlgoView

# Установка зависимостей
pip install -r requirements.txt
```

## Использование

### Пример анализа данных о кредитных картах

```python
from algoview.examples.credit_card_analysis import CreditCardAnalyzer

# Создаем анализатор
analyzer = CreditCardAnalyzer()

# Визуализируем работу всех алгоритмов
analyzer.visualize_all()
```

### Использование отдельных алгоритмов

```python
from algoview.base_algorithms.linear.logistic import LogisticRegression
from algoview.visualization.logistic_viz import LogisticRegressionVisualizer

# Создаем и обучаем модель
model = LogisticRegression()
model.fit(X_train, y_train)

# Визуализируем процесс обучения и принятия решений
visualizer = LogisticRegressionVisualizer()
visualizer.visualize()
```

## Структура проекта

```
algoview/
├── base_algorithms/
│   ├── linear/ (LogisticRegression, LDA)
│   ├── trees/ (DecisionTree, DecisionTreeRegressor)
│   ├── probabilistic/ (GaussianNB)
│   └── neighbors/ (KNN)
├── ensemble/
│   ├── bagging.py (RandomForest, BaggingClassifier)
│   ├── boosting.py (GBMClassifier)
│   └── stacking.py (StackingEnsemble)
├── metrics/
├── utils/
└── visualization/
```

## Требования

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn (только для сравнения результатов)
- Seaborn (для визуализации)

## Лицензия

MIT License

## Авторы

- [iRatG](https://github.com/iRatG)

## Вклад в проект

Мы приветствуем вклад в развитие проекта! Если вы хотите добавить новый алгоритм, улучшить визуализацию или исправить ошибку:

1. Форкните репозиторий
2. Создайте ветку для ваших изменений
3. Внесите изменения и создайте pull request

## Цитирование

Если вы используете этот проект в своих исследованиях или обучении, пожалуйста, ссылайтесь на него:

```bibtex
@misc{algoview2024,
  author = {iRatG},
  title = {AlgoView: Educational Machine Learning Algorithms Visualization},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/iRatG/AlgoView}
}
``` 