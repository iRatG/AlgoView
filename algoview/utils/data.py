import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        # Пропуск целевой переменной
        if col == target:
            continue
            
        # Базовая информация
        d = {
            'feature': col,
            'dtype': str(df[col].dtype),
            'nunique': df[col].nunique(),
            'missing': df[col].isnull().sum() / n
        }
        
        # Определение типа признака
        is_categorical = df[col].nunique() <= categorical_threshold and df[col].dtype != 'float64'
        is_numeric = df[col].dtype in ['int64', 'float64']
        
        if is_numeric:
            d.update({
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            })
            
            # Корреляция с целевой переменной для числовых признаков
            if target and df[target].dtype in ['int64', 'float64']:
                d['target_corr'] = df[col].corr(df[target])
                
        if is_categorical:
            # Мода и её частота для категориальных признаков
            mode_val = df[col].mode().iloc[0]
            mode_freq = df[col].value_counts().iloc[0] / n
            d.update({
                'mode': mode_val,
                'mode_freq': mode_freq
            })
            
            # Связь с целевой переменной для категориальных признаков
            if target and df[target].dtype in ['int64', 'float64']:
                target_means = df.groupby(col)[target].mean()
                d['target_mean_diff'] = target_means.max() - target_means.min()
        
        summary.append(d)
    
    return pd.DataFrame(summary)

def feature_selector(summary_df, target_col, max_missing_ratio=0.1, min_variance=1e-5,
                    corr_threshold=0.1, max_unique_cat=15, exclude_features=None):
    """
    Отбор признаков на основе их характеристик.
    """
    selected_features = []
    exclude_features = exclude_features or []
    
    for _, row in summary_df.iterrows():
        feature = row['feature']
        
        # Пропуск исключенных признаков и целевой переменной
        if feature in exclude_features or feature == target_col:
            continue
            
        # Фильтр по пропущенным значениям
        if row['missing'] > max_missing_ratio:
            continue
            
        # Фильтр по дисперсии для числовых признаков
        if 'std' in row and row['std'] < min_variance:
            continue
            
        # Фильтр по корреляции с целевой переменной
        if 'target_corr' in row and abs(row['target_corr']) < corr_threshold:
            continue
            
        # Фильтр по количеству уникальных значений для категориальных
        if row['nunique'] > max_unique_cat and row['dtype'] != 'float64':
            continue
            
        selected_features.append(feature)
    
    return selected_features 