"""
Utilidades para análise de Redes Neurais
Suporta MLPClassifier e MLPRegressor com Holdout e K-Fold
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from collections import OrderedDict

from src.utils.lazy_imports import get_numpy, get_pandas


def get_sklearn_neural_network():
    """Lazy import do sklearn.neural_network"""
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    return MLPClassifier, MLPRegressor


def get_sklearn_preprocessing():
    """Lazy import do sklearn.preprocessing"""
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    return OneHotEncoder, LabelEncoder


def get_sklearn_model_selection():
    """Lazy import do sklearn.model_selection"""
    from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
    return train_test_split, KFold, StratifiedKFold, GridSearchCV


def get_sklearn_metrics():
    """Lazy import do sklearn.metrics"""
    from sklearn.metrics import (
        mean_squared_error, r2_score,
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_curve, roc_auc_score
    )
    return (mean_squared_error, r2_score, accuracy_score, precision_score,
            recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score)


def get_sklearn_compose():
    """Lazy import do sklearn.compose"""
    from sklearn.compose import ColumnTransformer
    return ColumnTransformer


def get_sklearn_inspection():
    """Lazy import do sklearn.inspection"""
    from sklearn.inspection import permutation_importance
    return permutation_importance


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
    _, LabelEncoder = get_sklearn_preprocessing()
    
    df_encoded = df.copy()
    encoders = {}
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
    
    return df_encoded, encoders


def transform_features(X: pd.DataFrame, categorical_cols: List[str]) -> Tuple[np.ndarray, List[str], Any]:
    """Transforma features aplicando OneHotEncoder para categóricas"""
    OneHotEncoder, _ = get_sklearn_preprocessing()
    ColumnTransformer = get_sklearn_compose()
    
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    transformers = []
    if len(categorical_cols) > 0:
        transformers.append(('onehot', OneHotEncoder(sparse_output=False), categorical_cols))
    
    preprocessor = ColumnTransformer(transformers, remainder='passthrough')
    X_transformed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    
    return X_transformed, feature_names.tolist(), preprocessor


def calculate_metrics_classification(y_true, y_pred, y_proba, model_classes) -> Dict:
    """Calcula métricas para classificação"""
    (_, _, accuracy_score, precision_score, recall_score, 
     f1_score, confusion_matrix, roc_curve, roc_auc_score) = get_sklearn_metrics()
    np = get_numpy()
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    # ROC AUC
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 2:
        # Classificação binária
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba[:, 1]))
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1], pos_label=unique_classes[1])
        metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
    else:
        # Multiclasse
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted'))
        metrics['roc_curve'] = None
    
    return metrics


def calculate_metrics_regression(y_true, y_pred) -> Dict:
    """Calcula métricas para regressão"""
    mean_squared_error, r2_score, _, _, _, _, _, _, _ = get_sklearn_metrics()
    np = get_numpy()
    
    metrics = {
        'mse': float(mean_squared_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2': float(r2_score(y_true, y_pred)),
        'mean': float(np.mean(y_pred)),
        'std': float(np.std(y_pred))
    }
    
    return metrics


def calculate_feature_importance(model, X_test, y_test, feature_names: List[str]) -> Dict:
    """Calcula importância das features usando permutation importance"""
    permutation_importance = get_sklearn_inspection()
    np = get_numpy()
    
    perm_importance = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )
    
    importance_dict = {
        name.replace('onehot__', '').replace('remainder__', ''): float(score)
        for name, score in zip(feature_names, perm_importance.importances_mean)
    }
    
    # Ordena por importância
    importance_dict = OrderedDict(sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True))
    
    return importance_dict


def train_neural_network_holdout(
    df: pd.DataFrame,
    x_columns: List[str],
    y_column: str,
    categorical_cols: List[str],
    activation: str,
    test_size: float,
    max_iter: int = 500
) -> Dict[str, Any]:
    """
    Treina rede neural usando método Holdout
    
    Args:
        df: DataFrame com os dados
        x_columns: Lista de colunas X
        y_column: Nome da coluna Y
        categorical_cols: Colunas categóricas para encoding
        activation: Função de ativação (relu, tanh, logistic, identity)
        test_size: Proporção do conjunto de teste
        max_iter: Número máximo de iterações
    
    Returns:
        Dicionário com resultados da análise
    """
    pd = get_pandas()
    np = get_numpy()
    MLPClassifier, MLPRegressor = get_sklearn_neural_network()
    train_test_split, _, _, GridSearchCV = get_sklearn_model_selection()
    
    # Prepara dados
    X = df[x_columns].copy()
    y = df[y_column].copy()
    
    # Verifica se é classificação ou regressão
    is_classification = is_categorical_target(y)
    
    # Codifica variáveis categóricas em X
    if categorical_cols:
        X, encoders = encode_categorical_columns(X, categorical_cols)
    
    # Transforma features
    X_transformed, feature_names, preprocessor = transform_features(X, [])
    
    # Divide em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=test_size, random_state=42, stratify=y if is_classification else None
    )
    
    # Define modelo base
    if is_classification:
        base_model = MLPClassifier(random_state=42, max_iter=max_iter)
        scoring = 'accuracy'
    else:
        base_model = MLPRegressor(random_state=42, max_iter=max_iter)
        scoring = 'r2'
    
    # Grid de hiperparâmetros
    param_grid = {
        'hidden_layer_sizes': [
            (5,), (10,), (15,),
            (5, 3), (10, 5), (15, 10),
            (10, 5, 3), (15, 10, 5),
        ],
        'activation': [activation],
        'solver': ['adam', 'sgd'],
        'learning_rate': ['constant', 'adaptive'],
        'alpha': [0.0001, 0.001, 0.01]
    }
    
    # Grid Search
    grid_search = GridSearchCV(
        base_model, param_grid,
        scoring=scoring,
        cv=3,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Predições
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    # Calcula métricas
    if is_classification:
        y_proba_train = best_model.predict_proba(X_train)
        y_proba_test = best_model.predict_proba(X_test)
        
        metrics_train = calculate_metrics_classification(y_train, y_pred_train, y_proba_train, best_model.classes_)
        metrics_test = calculate_metrics_classification(y_test, y_pred_test, y_proba_test, best_model.classes_)
    else:
        metrics_train = calculate_metrics_regression(y_train, y_pred_train)
        metrics_test = calculate_metrics_regression(y_test, y_pred_test)
    
    # Feature importance
    feature_importance = calculate_feature_importance(best_model, X_test, y_test, feature_names)
    
    # Informações do modelo
    model_info = {
        'hidden_layers': best_model.hidden_layer_sizes,
        'n_layers': len(best_model.hidden_layer_sizes) if isinstance(best_model.hidden_layer_sizes, tuple) else 1,
        'n_iter': best_model.n_iter_,
        'loss': float(best_model.loss_),
        'best_params': grid_search.best_params_
    }
    
    return {
        'model': best_model,
        'is_classification': is_classification,
        'metrics_train': metrics_train,
        'metrics_test': metrics_test,
        'feature_importance': feature_importance,
        'model_info': model_info,
        'y_train': y_train.tolist(),
        'y_test': y_test.tolist(),
        'y_pred_train': y_pred_train.tolist(),
        'y_pred_test': y_pred_test.tolist(),
        'feature_names': feature_names,
        'preprocessor': preprocessor
    }


def train_neural_network_kfold(
    df: pd.DataFrame,
    x_columns: List[str],
    y_column: str,
    categorical_cols: List[str],
    activation: str,
    n_folds: int,
    max_iter: int = 500
) -> Dict[str, Any]:
    """
    Treina rede neural usando método K-Fold
    
    Args:
        df: DataFrame com os dados
        x_columns: Lista de colunas X
        y_column: Nome da coluna Y
        categorical_cols: Colunas categóricas para encoding
        activation: Função de ativação
        n_folds: Número de folds
        max_iter: Número máximo de iterações
    
    Returns:
        Dicionário com resultados da análise
    """
    pd = get_pandas()
    np = get_numpy()
    MLPClassifier, MLPRegressor = get_sklearn_neural_network()
    _, KFold, StratifiedKFold, GridSearchCV = get_sklearn_model_selection()
    
    # Prepara dados
    X = df[x_columns].copy()
    y = df[y_column].copy()
    
    # Verifica se é classificação ou regressão
    is_classification = is_categorical_target(y)
    
    # Codifica variáveis categóricas em X
    if categorical_cols:
        X, encoders = encode_categorical_columns(X, categorical_cols)
    
    # Transforma features
    X_transformed, feature_names, preprocessor = transform_features(X, [])
    
    # Define modelo base
    if is_classification:
        base_model = MLPClassifier(random_state=42, max_iter=max_iter)
        scoring = 'accuracy'
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        base_model = MLPRegressor(random_state=42, max_iter=max_iter)
        scoring = 'r2'
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Grid de hiperparâmetros
    param_grid = {
        'hidden_layer_sizes': [
            (5,), (10,), (15,),
            (5, 3), (10, 5), (15, 10),
            (10, 5, 3), (15, 10, 5),
        ],
        'activation': [activation],
        'solver': ['adam', 'sgd'],
        'learning_rate': ['constant', 'adaptive'],
        'alpha': [0.0001, 0.001, 0.01]
    }
    
    # Grid Search com K-Fold
    grid_search = GridSearchCV(
        base_model, param_grid,
        scoring=scoring,
        cv=kf,
        n_jobs=-1
    )
    
    grid_search.fit(X_transformed, y)
    best_model = grid_search.best_estimator_
    
    # Avalia em cada fold
    fold_metrics = []
    all_y_true = []
    all_y_pred = []
    
    for train_idx, test_idx in kf.split(X_transformed, y):
        X_train_fold = X_transformed[train_idx]
        X_test_fold = X_transformed[test_idx]
        y_train_fold = y.iloc[train_idx]
        y_test_fold = y.iloc[test_idx]
        
        best_model.fit(X_train_fold, y_train_fold)
        y_pred_fold = best_model.predict(X_test_fold)
        
        if is_classification:
            y_proba_fold = best_model.predict_proba(X_test_fold)
            metrics_fold = calculate_metrics_classification(y_test_fold, y_pred_fold, y_proba_fold, best_model.classes_)
        else:
            metrics_fold = calculate_metrics_regression(y_test_fold, y_pred_fold)
        
        fold_metrics.append(metrics_fold)
        all_y_true.extend(y_test_fold.tolist())
        all_y_pred.extend(y_pred_fold.tolist())
    
    # Calcula médias das métricas
    avg_metrics = {}
    for key in fold_metrics[0].keys():
        if key not in ['confusion_matrix', 'roc_curve']:
            values = [m[key] for m in fold_metrics]
            avg_metrics[key] = float(np.mean(values))
            avg_metrics[key + '_std'] = float(np.std(values))
    
    # Confusion matrix agregada (se classificação)
    if is_classification:
        avg_metrics['confusion_matrix'] = fold_metrics[-1]['confusion_matrix']
        avg_metrics['roc_curve'] = fold_metrics[-1]['roc_curve']
    
    # Feature importance
    feature_importance = calculate_feature_importance(best_model, X_transformed, y, feature_names)
    
    # Informações do modelo
    model_info = {
        'hidden_layers': best_model.hidden_layer_sizes,
        'n_layers': len(best_model.hidden_layer_sizes) if isinstance(best_model.hidden_layer_sizes, tuple) else 1,
        'n_iter': best_model.n_iter_,
        'loss': float(best_model.loss_),
        'best_params': grid_search.best_params_,
        'n_folds': n_folds
    }
    
    return {
        'model': best_model,
        'is_classification': is_classification,
        'metrics': avg_metrics,
        'feature_importance': feature_importance,
        'model_info': model_info,
        'y_true': all_y_true,
        'y_pred': all_y_pred,
        'feature_names': feature_names,
        'preprocessor': preprocessor
    }
