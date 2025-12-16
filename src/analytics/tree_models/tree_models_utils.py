"""
Utilidades para Modelos de Árvore (Tree Models)
Decision Tree, Random Forest e Gradient Boosting
"""

from typing import Dict, List, Tuple, Any
from collections import OrderedDict
import numpy as np
import pandas as pd


# ========== LAZY IMPORTS ==========
def get_pandas():
    """Lazy import do pandas"""
    import pandas as pd
    return pd


def get_numpy():
    """Lazy import do numpy"""
    import numpy as np
    return np


def get_sklearn_tree():
    """Lazy import do sklearn.tree"""
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, _tree
    return DecisionTreeClassifier, DecisionTreeRegressor, _tree


def get_sklearn_ensemble():
    """Lazy import do sklearn.ensemble"""
    from sklearn.ensemble import (
        RandomForestClassifier, 
        RandomForestRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor
    )
    return (RandomForestClassifier, RandomForestRegressor, 
            GradientBoostingClassifier, GradientBoostingRegressor)


def get_sklearn_model_selection():
    """Lazy import do sklearn.model_selection"""
    from sklearn.model_selection import train_test_split, GridSearchCV
    return train_test_split, GridSearchCV


def get_sklearn_metrics():
    """Lazy import do sklearn.metrics"""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report,
        mean_squared_error, r2_score, mean_absolute_error,
        roc_auc_score, roc_curve
    )
    return (accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, classification_report,
            mean_squared_error, r2_score, mean_absolute_error,
            roc_auc_score, roc_curve)


def get_sklearn_preprocessing():
    """Lazy import do sklearn.preprocessing"""
    from sklearn.preprocessing import LabelEncoder
    return LabelEncoder


def get_sklearn_inspection():
    """Lazy import do sklearn.inspection"""
    from sklearn.inspection import permutation_importance
    return permutation_importance


# ========== FUNÇÕES DE PROCESSAMENTO ==========

def is_categorical_target(y: pd.Series) -> bool:
    """Verifica se a variável alvo é categórica"""
    pd = get_pandas()
    
    # Verifica tipo
    if y.dtype == 'object' or y.dtype.name == 'category':
        return True
    
    # Verifica número de valores únicos
    unique_values = y.unique()
    if len(unique_values) <= 10 and all(isinstance(v, (str, int, np.integer)) for v in unique_values):
        return True
    
    return False


def encode_categorical_columns(df: pd.DataFrame, categorical_cols: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """Codifica colunas categóricas usando LabelEncoder"""
    LabelEncoder = get_sklearn_preprocessing()
    
    df_encoded = df.copy()
    encoders = {}
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
    
    return df_encoded, encoders


def calculate_feature_importance(model, X: pd.DataFrame) -> Dict:
    """Calcula importância das features do modelo"""
    np = get_numpy()
    
    importance_dict = {
        name: float(score)
        for name, score in zip(X.columns, model.feature_importances_)
    }
    
    # Ordena por importância
    importance_dict = OrderedDict(sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True))
    
    return importance_dict


def calculate_metrics_classification(y_true, y_pred, model=None, X_test=None) -> Dict:
    """Calcula métricas para classificação"""
    np = get_numpy()
    (accuracy_score, precision_score, recall_score, f1_score,
     confusion_matrix, classification_report, _, _, _,
     roc_auc_score, roc_curve) = get_sklearn_metrics()
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    # ROC AUC para classificação binária
    is_binary = len(np.unique(y_true)) == 2
    if is_binary and model is not None and X_test is not None:
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba))
                
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                metrics['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist()
                }
        except Exception as e:
            print(f"Erro ao calcular ROC AUC: {e}")
            metrics['roc_auc'] = 0
    
    return metrics


def calculate_metrics_regression(y_true, y_pred) -> Dict:
    """Calcula métricas para regressão"""
    np = get_numpy()
    (_, _, _, _, _, _, mean_squared_error, r2_score, mean_absolute_error, _, _) = get_sklearn_metrics()
    
    mse = mean_squared_error(y_true, y_pred)
    
    metrics = {
        'r2_score': float(r2_score(y_true, y_pred)),
        'mse': float(mse),
        'rmse': float(np.sqrt(mse)),
        'mae': float(mean_absolute_error(y_true, y_pred))
    }
    
    return metrics


def train_decision_tree(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_cols: List[str],
    test_size: float = 0.3,
    max_depth: int = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42
) -> Dict:
    """
    Treina um modelo de Decision Tree
    
    Args:
        X: DataFrame com variáveis independentes
        y: Série com variável dependente
        categorical_cols: Lista de colunas categóricas
        test_size: Proporção do conjunto de teste
        max_depth: Profundidade máxima da árvore
        min_samples_split: Mínimo de amostras para split
        min_samples_leaf: Mínimo de amostras em folha
        random_state: Seed para reprodutibilidade
        
    Returns:
        Dict com modelo, métricas e informações
    """
    np = get_numpy()
    train_test_split, GridSearchCV = get_sklearn_model_selection()
    DecisionTreeClassifier, DecisionTreeRegressor, _tree = get_sklearn_tree()
    
    # Encode categóricas
    if categorical_cols:
        X, encoders = encode_categorical_columns(X, categorical_cols)
    else:
        encoders = {}
    
    # Verifica tipo de problema
    is_classification = is_categorical_target(y)
    
    # Encode y se categórico
    LabelEncoder = get_sklearn_preprocessing()
    y_encoder = None
    if is_classification:
        y_encoder = LabelEncoder()
        y_encoded = y_encoder.fit_transform(y)
    else:
        y_encoded = y.values
    
    # Split dos dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state
    )
    
    # Grid Search para melhor modelo
    param_grid = {
        'max_depth': [3, 5, 10, 15, None] if max_depth is None else [max_depth],
        'min_samples_split': [2, 5, 10] if min_samples_split == 2 else [min_samples_split],
        'min_samples_leaf': [1, 2, 4] if min_samples_leaf == 1 else [min_samples_leaf],
        'criterion': ['gini', 'entropy'] if is_classification else ['squared_error', 'absolute_error']
    }
    
    model = (
        DecisionTreeClassifier(random_state=random_state)
        if is_classification
        else DecisionTreeRegressor(random_state=random_state)
    )
    
    # Define número de folds
    n_samples = len(X_train)
    cv = 5 if n_samples > 1000 else 3
    
    if is_classification:
        from collections import Counter
        class_counts = Counter(y_train)
        min_class_count = min(class_counts.values())
        cv = min(cv, min_class_count)
    
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring='accuracy' if is_classification else 'r2',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Predições
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    # Calcula métricas
    if is_classification:
        train_metrics = calculate_metrics_classification(y_train, y_pred_train)
        test_metrics = calculate_metrics_classification(y_test, y_pred_test, best_model, X_test)
    else:
        train_metrics = calculate_metrics_regression(y_train, y_pred_train)
        test_metrics = calculate_metrics_regression(y_test, y_pred_test)
    
    # Feature importance
    feature_importance = calculate_feature_importance(best_model, X)
    
    # Informações do modelo
    model_info = {
        'model_type': 'Decision Tree',
        'max_depth': int(best_model.get_depth()),
        'n_leaves': int(best_model.get_n_leaves()),
        'n_features': int(X.shape[1]),
        'n_samples_train': int(len(X_train)),
        'n_samples_test': int(len(X_test)),
        'best_params': grid_search.best_params_
    }
    
    return {
        'model': best_model,
        'encoders': encoders,
        'y_encoder': y_encoder,
        'is_classification': is_classification,
        'feature_names': X.columns.tolist(),
        'feature_importance': feature_importance,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'model_info': model_info,
        'y_pred_test': y_pred_test.tolist(),
        'y_test': y_test.tolist()
    }


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_cols: List[str],
    test_size: float = 0.3,
    n_estimators: int = 100,
    max_depth: int = None,
    min_samples_leaf: int = 1,
    random_state: int = 42
) -> Dict:
    """
    Treina um modelo de Random Forest
    """
    np = get_numpy()
    train_test_split, _ = get_sklearn_model_selection()
    RandomForestClassifier, RandomForestRegressor, _, _ = get_sklearn_ensemble()
    
    # Encode categóricas
    if categorical_cols:
        X, encoders = encode_categorical_columns(X, categorical_cols)
    else:
        encoders = {}
    
    # Verifica tipo de problema
    is_classification = is_categorical_target(y)
    
    # Encode y se categórico
    LabelEncoder = get_sklearn_preprocessing()
    y_encoder = None
    if is_classification:
        y_encoder = LabelEncoder()
        y_encoded = y_encoder.fit_transform(y)
    else:
        y_encoded = y.values
    
    # Split dos dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state
    )
    
    # Treina modelo
    if is_classification:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        )
    else:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        )
    
    model.fit(X_train, y_train)
    
    # Predições
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calcula métricas
    if is_classification:
        train_metrics = calculate_metrics_classification(y_train, y_pred_train)
        test_metrics = calculate_metrics_classification(y_test, y_pred_test, model, X_test)
    else:
        train_metrics = calculate_metrics_regression(y_train, y_pred_train)
        test_metrics = calculate_metrics_regression(y_test, y_pred_test)
    
    # Feature importance
    feature_importance = calculate_feature_importance(model, X)
    
    # Informações do modelo
    model_info = {
        'model_type': 'Random Forest',
        'n_estimators': int(n_estimators),
        'max_depth': max_depth if max_depth is not None else 'None',
        'n_features': int(X.shape[1]),
        'n_samples_train': int(len(X_train)),
        'n_samples_test': int(len(X_test))
    }
    
    return {
        'model': model,
        'encoders': encoders,
        'y_encoder': y_encoder,
        'is_classification': is_classification,
        'feature_names': X.columns.tolist(),
        'feature_importance': feature_importance,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'model_info': model_info,
        'y_pred_test': y_pred_test.tolist(),
        'y_test': y_test.tolist()
    }


def train_gradient_boosting(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_cols: List[str],
    test_size: float = 0.3,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    subsample: float = 1.0,
    random_state: int = 42
) -> Dict:
    """
    Treina um modelo de Gradient Boosting com Grid Search
    """
    np = get_numpy()
    train_test_split, GridSearchCV = get_sklearn_model_selection()
    _, _, GradientBoostingClassifier, GradientBoostingRegressor = get_sklearn_ensemble()
    
    # Encode categóricas
    if categorical_cols:
        X, encoders = encode_categorical_columns(X, categorical_cols)
    else:
        encoders = {}
    
    # Verifica tipo de problema
    is_classification = is_categorical_target(y)
    
    # Encode y se categórico
    LabelEncoder = get_sklearn_preprocessing()
    y_encoder = None
    if is_classification:
        y_encoder = LabelEncoder()
        y_encoded = y_encoder.fit_transform(y)
    else:
        y_encoded = y.values
    
    # Split dos dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state
    )
    
    # Grid Search para melhor modelo
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.5],
        'subsample': [0.5, 0.8, 1.0],
        'max_depth': [3, 5, 7]
    }
    
    if is_classification:
        model = GradientBoostingClassifier(
            random_state=random_state,
            n_iter_no_change=10
        )
    else:
        model = GradientBoostingRegressor(
            random_state=random_state,
            n_iter_no_change=10
        )
    
    # Define número de folds
    n_samples = len(X_train)
    cv = 5 if n_samples > 1000 else 3
    
    if is_classification:
        from collections import Counter
        class_counts = Counter(y_train)
        min_class_count = min(class_counts.values())
        cv = min(cv, min_class_count)
    
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring='accuracy' if is_classification else 'r2',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Predições
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    # Calcula métricas
    if is_classification:
        train_metrics = calculate_metrics_classification(y_train, y_pred_train)
        test_metrics = calculate_metrics_classification(y_test, y_pred_test, best_model, X_test)
    else:
        train_metrics = calculate_metrics_regression(y_train, y_pred_train)
        test_metrics = calculate_metrics_regression(y_test, y_pred_test)
    
    # Feature importance
    feature_importance = calculate_feature_importance(best_model, X)
    
    # Informações do modelo
    model_info = {
        'model_type': 'Gradient Boosting',
        'n_estimators': int(best_model.n_estimators),
        'learning_rate': float(best_model.learning_rate),
        'max_depth': int(best_model.max_depth),
        'subsample': float(best_model.subsample),
        'n_features': int(X.shape[1]),
        'n_samples_train': int(len(X_train)),
        'n_samples_test': int(len(X_test)),
        'best_params': grid_search.best_params_
    }
    
    return {
        'model': best_model,
        'encoders': encoders,
        'y_encoder': y_encoder,
        'is_classification': is_classification,
        'feature_names': X.columns.tolist(),
        'feature_importance': feature_importance,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'model_info': model_info,
        'y_pred_test': y_pred_test.tolist(),
        'y_test': y_test.tolist()
    }
