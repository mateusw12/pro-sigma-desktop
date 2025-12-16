"""
Gaussian Process Regression Utilities
Regressão usando Processos Gaussianos com intervalos de confiança
"""
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, ExpSineSquared, 
    DotProduct, WhiteKernel, ConstantKernel
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def get_available_kernels():
    """
    Retorna dicionário com kernels disponíveis
    
    Returns:
        dict: Nome do kernel e função construtora
    """
    return {
        'RBF': lambda: ConstantKernel(1.0) * RBF(length_scale=1.0),
        'Matern (ν=1.5)': lambda: ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5),
        'Matern (ν=2.5)': lambda: ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5),
        'Rational Quadratic': lambda: ConstantKernel(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0),
        'Exp-Sine-Squared': lambda: ConstantKernel(1.0) * ExpSineSquared(length_scale=1.0, periodicity=1.0),
        'RBF + White Noise': lambda: ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0),
        'Dot Product': lambda: ConstantKernel(1.0) * DotProduct(sigma_0=1.0),
        'RBF + Rational Quadratic': lambda: (
            ConstantKernel(1.0) * RBF(length_scale=1.0) + 
            ConstantKernel(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0)
        )
    }


def train_gaussian_process(data, x_columns, y_column, kernel_name='RBF', 
                          test_size=0.2, scale=True, n_restarts=10, alpha=1e-10):
    """
    Treina modelo Gaussian Process Regression
    
    Args:
        data: DataFrame com os dados
        x_columns: Lista de colunas para features (X)
        y_column: Nome da coluna target (Y)
        kernel_name: Nome do kernel a usar
        test_size: Proporção do conjunto de teste
        scale: Se deve normalizar os dados
        n_restarts: Número de reinícios para otimização
        alpha: Regularização (ruído)
    
    Returns:
        dict com resultados do modelo
    """
    # Preparar dados
    X = data[x_columns].values
    y = data[y_column].values
    
    # Verificar dados válidos
    if np.any(np.isnan(X)) or np.any(np.isinf(X)) or np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Dados contêm valores NaN ou infinitos.")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Normalizar se necessário
    scaler_X = None
    scaler_y = None
    
    if scale:
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
        y_train_scaled = y_train
        y_test_scaled = y_test
    
    # Selecionar kernel
    kernels = get_available_kernels()
    if kernel_name not in kernels:
        kernel_name = 'RBF'
    
    kernel = kernels[kernel_name]()
    
    # Treinar modelo
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        alpha=alpha,
        random_state=42,
        normalize_y=False  # Já normalizamos manualmente
    )
    
    gpr.fit(X_train_scaled, y_train_scaled)
    
    # Predições
    y_train_pred_scaled, y_train_std_scaled = gpr.predict(X_train_scaled, return_std=True)
    y_test_pred_scaled, y_test_std_scaled = gpr.predict(X_test_scaled, return_std=True)
    
    # Reverter normalização se necessário
    if scale:
        y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
        y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()
        
        # Escalar desvio padrão
        y_train_std = y_train_std_scaled * scaler_y.scale_[0]
        y_test_std = y_test_std_scaled * scaler_y.scale_[0]
    else:
        y_train_pred = y_train_pred_scaled
        y_test_pred = y_test_pred_scaled
        y_train_std = y_train_std_scaled
        y_test_std = y_test_std_scaled
    
    # Calcular métricas
    train_metrics = {
        'mse': mean_squared_error(y_train, y_train_pred),
        'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'mae': mean_absolute_error(y_train, y_train_pred),
        'r2': r2_score(y_train, y_train_pred)
    }
    
    test_metrics = {
        'mse': mean_squared_error(y_test, y_test_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'mae': mean_absolute_error(y_test, y_test_pred),
        'r2': r2_score(y_test, y_test_pred)
    }
    
    # Log-likelihood marginal
    log_marginal_likelihood = gpr.log_marginal_likelihood_value_
    
    # Kernel otimizado
    optimized_kernel = str(gpr.kernel_)
    
    # Calcular intervalo de confiança (95%)
    train_lower_95 = y_train_pred - 1.96 * y_train_std
    train_upper_95 = y_train_pred + 1.96 * y_train_std
    test_lower_95 = y_test_pred - 1.96 * y_test_std
    test_upper_95 = y_test_pred + 1.96 * y_test_std
    
    # Calcular cobertura do intervalo de confiança
    train_coverage = np.mean((y_train >= train_lower_95) & (y_train <= train_upper_95))
    test_coverage = np.mean((y_test >= test_lower_95) & (y_test <= test_upper_95))
    
    # Interpretações
    interpretations = generate_interpretations(
        train_metrics, test_metrics, log_marginal_likelihood, 
        train_coverage, test_coverage
    )
    
    return {
        'model': gpr,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'y_train_std': y_train_std,
        'y_test_std': y_test_std,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'log_marginal_likelihood': log_marginal_likelihood,
        'kernel_name': kernel_name,
        'optimized_kernel': optimized_kernel,
        'train_coverage': train_coverage,
        'test_coverage': test_coverage,
        'interpretations': interpretations,
        'x_columns': x_columns,
        'y_column': y_column
    }


def predict_with_uncertainty(results, X_new):
    """
    Faz predições com incerteza para novos dados
    
    Args:
        results: Resultado do train_gaussian_process
        X_new: Novos dados (array ou DataFrame)
    
    Returns:
        tuple: (predições, desvio padrão)
    """
    model = results['model']
    scaler_X = results['scaler_X']
    scaler_y = results['scaler_y']
    
    # Converter para array se necessário
    if isinstance(X_new, pd.DataFrame):
        X_new = X_new.values
    
    # Normalizar se necessário
    if scaler_X is not None:
        X_new_scaled = scaler_X.transform(X_new)
    else:
        X_new_scaled = X_new
    
    # Predizer
    y_pred_scaled, y_std_scaled = model.predict(X_new_scaled, return_std=True)
    
    # Reverter normalização
    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_std = y_std_scaled * scaler_y.scale_[0]
    else:
        y_pred = y_pred_scaled
        y_std = y_std_scaled
    
    return y_pred, y_std


def generate_interpretations(train_metrics, test_metrics, log_likelihood, 
                            train_coverage, test_coverage):
    """
    Gera interpretações dos resultados
    
    Args:
        train_metrics: Métricas de treino
        test_metrics: Métricas de teste
        log_likelihood: Log-likelihood marginal
        train_coverage: Cobertura do IC em treino
        test_coverage: Cobertura do IC em teste
    
    Returns:
        dict com interpretações
    """
    interpretations = {}
    
    # R² do teste
    r2_test = test_metrics['r2']
    if r2_test >= 0.9:
        interpretations['r2'] = {
            'status': 'Excelente',
            'color': 'green',
            'message': f'R² = {r2_test:.4f} - Modelo explica >90% da variância'
        }
    elif r2_test >= 0.7:
        interpretations['r2'] = {
            'status': 'Bom',
            'color': 'blue',
            'message': f'R² = {r2_test:.4f} - Bom ajuste do modelo'
        }
    elif r2_test >= 0.5:
        interpretations['r2'] = {
            'status': 'Razoável',
            'color': 'yellow',
            'message': f'R² = {r2_test:.4f} - Ajuste moderado'
        }
    else:
        interpretations['r2'] = {
            'status': 'Fraco',
            'color': 'red',
            'message': f'R² = {r2_test:.4f} - Modelo pouco preditivo, considere outro kernel'
        }
    
    # Overfitting
    r2_diff = train_metrics['r2'] - test_metrics['r2']
    if r2_diff < 0.05:
        interpretations['overfitting'] = {
            'status': 'Sem Overfitting',
            'color': 'green',
            'message': f'Diferença R² Train-Test = {r2_diff:.4f} - Generalização boa'
        }
    elif r2_diff < 0.15:
        interpretations['overfitting'] = {
            'status': 'Leve Overfitting',
            'color': 'yellow',
            'message': f'Diferença R² Train-Test = {r2_diff:.4f} - Leve overfitting'
        }
    else:
        interpretations['overfitting'] = {
            'status': 'Overfitting',
            'color': 'red',
            'message': f'Diferença R² Train-Test = {r2_diff:.4f} - Overfitting detectado, aumente alpha'
        }
    
    # Cobertura do intervalo de confiança
    if test_coverage >= 0.93 and test_coverage <= 0.97:
        interpretations['confidence'] = {
            'status': 'Calibrado',
            'color': 'green',
            'message': f'Cobertura IC 95% = {test_coverage:.1%} - Intervalos bem calibrados'
        }
    elif test_coverage >= 0.85:
        interpretations['confidence'] = {
            'status': 'Aceitável',
            'color': 'blue',
            'message': f'Cobertura IC 95% = {test_coverage:.1%} - Intervalos razoáveis'
        }
    elif test_coverage > 0.97:
        interpretations['confidence'] = {
            'status': 'Conservador',
            'color': 'yellow',
            'message': f'Cobertura IC 95% = {test_coverage:.1%} - Intervalos muito largos (conservador)'
        }
    else:
        interpretations['confidence'] = {
            'status': 'Mal Calibrado',
            'color': 'red',
            'message': f'Cobertura IC 95% = {test_coverage:.1%} - Intervalos mal calibrados'
        }
    
    # Log-likelihood marginal (quanto maior, melhor)
    interpretations['likelihood'] = {
        'value': log_likelihood,
        'message': f'Log-Likelihood Marginal = {log_likelihood:.2f} (maior = melhor kernel)'
    }
    
    return interpretations
