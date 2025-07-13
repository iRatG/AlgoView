import numpy as np

def simple_oversample(X, y, random_state=42):
    """
    Простой оверсэмплинг для балансировки классов.
    """
    np.random.seed(random_state)
    
    # Определение размеров классов
    unique_classes = np.unique(y)
    class_counts = [np.sum(y == cls) for cls in unique_classes]
    max_size = max(class_counts)
    
    # Оверсэмплинг каждого класса до размера наибольшего
    X_resampled = []
    y_resampled = []
    
    for cls in unique_classes:
        # Индексы текущего класса
        idx = np.where(y == cls)[0]
        
        # Если нужен оверсэмплинг
        if len(idx) < max_size:
            # Случайный выбор с повторением
            resample_idx = np.random.choice(idx, size=max_size-len(idx), replace=True)
            X_resampled.extend(X[idx])
            X_resampled.extend(X[resample_idx])
            y_resampled.extend(y[idx])
            y_resampled.extend(y[resample_idx])
        else:
            X_resampled.extend(X[idx])
            y_resampled.extend(y[idx])
    
    return np.array(X_resampled), np.array(y_resampled)

def prepare_data_classic(df, selected_features, target_col):
    """
    Классическая подготовка данных для обучения.
    """
    # Разделение на признаки и целевую переменную
    X = df[selected_features].values
    y = df[target_col].values
    
    # Балансировка классов
    X_balanced, y_balanced = simple_oversample(X, y)
    
    return X_balanced, y_balanced 