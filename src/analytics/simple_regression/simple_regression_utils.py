"""
Simple Regression Utilities
Functions for simple linear regression analysis
"""
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy import stats
from sklearn.metrics import mean_squared_error


def calculate_simple_regression(X: np.ndarray, y: np.ndarray) -> Dict:
    """
    Calculate simple linear regression
    
    Args:
        X: Independent variable (n x 1)
        y: Dependent variable (n,)
        
    Returns:
        Dictionary with regression results
    """
    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n = len(y)
    
    # Add intercept term (column of ones)
    X_with_intercept = np.column_stack([np.ones(n), X])
    
    # Calculate coefficients using OLS: β = (X'X)^-1 X'y
    XtX = X_with_intercept.T @ X_with_intercept
    Xty = X_with_intercept.T @ y
    coefficients = np.linalg.solve(XtX, Xty)
    
    # Predictions
    y_pred = X_with_intercept @ coefficients
    
    # Residuals
    residuals = y - y_pred
    
    # Sum of Squares
    SS_total = np.sum((y - np.mean(y)) ** 2)
    SS_residual = np.sum(residuals ** 2)
    SS_regression = SS_total - SS_residual
    
    # Degrees of freedom
    df_regression = 1  # Simple regression has 1 predictor
    df_residual = n - 2  # n - (number of parameters)
    df_total = n - 1
    
    # Mean Squares
    MS_regression = SS_regression / df_regression
    MS_residual = SS_residual / df_residual
    
    # F-statistic for ANOVA
    F_statistic = MS_regression / MS_residual
    p_value_F = 1 - stats.f.cdf(F_statistic, df_regression, df_residual)
    
    # R-squared
    R_squared = 1 - (SS_residual / SS_total)
    R_squared_adj = 1 - ((SS_residual / df_residual) / (SS_total / df_total))
    
    # Standard error of regression
    SE_regression = np.sqrt(MS_residual)
    
    # Standard errors of coefficients
    var_coef = MS_residual * np.linalg.inv(XtX)
    SE_coefficients = np.sqrt(np.diag(var_coef))
    
    # t-statistics
    t_statistics = coefficients / SE_coefficients
    
    # p-values for coefficients
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_statistics), df_residual))
    
    # Confidence intervals (95%)
    alpha = 0.05
    t_critical = stats.t.ppf(1 - alpha/2, df_residual)
    CI_lower = coefficients - t_critical * SE_coefficients
    CI_upper = coefficients + t_critical * SE_coefficients
    
    # VIF (Variance Inflation Factor) - For simple regression, VIF = 1
    VIF = 1.0  # No multicollinearity in simple regression
    
    # Correlation coefficient
    correlation = np.corrcoef(X.flatten(), y)[0, 1]
    
    # RMSE and MAE
    RMSE = np.sqrt(mean_squared_error(y, y_pred))
    MAE = np.mean(np.abs(residuals))
    
    results = {
        'coefficients': coefficients,
        'SE_coefficients': SE_coefficients,
        't_statistics': t_statistics,
        'p_values': p_values,
        'CI_lower': CI_lower,
        'CI_upper': CI_upper,
        'predictions': y_pred,
        'residuals': residuals,
        'R_squared': R_squared,
        'R_squared_adj': R_squared_adj,
        'correlation': correlation,
        'n': n,
        'df_regression': df_regression,
        'df_residual': df_residual,
        'df_total': df_total,
        'SS_regression': SS_regression,
        'SS_residual': SS_residual,
        'SS_total': SS_total,
        'MS_regression': MS_regression,
        'MS_residual': MS_residual,
        'F_statistic': F_statistic,
        'p_value_F': p_value_F,
        'SE_regression': SE_regression,
        'RMSE': RMSE,
        'MAE': MAE,
        'VIF': VIF,
        'mean_y': np.mean(y),
        'mean_X': np.mean(X)
    }
    
    return results


def create_anova_table(results: Dict) -> pd.DataFrame:
    """
    Create ANOVA table for regression
    
    Args:
        results: Dictionary with regression results
        
    Returns:
        DataFrame with ANOVA table
    """
    anova_data = {
        'Fonte': ['Regressão', 'Resíduo', 'Total'],
        'DF': [
            results['df_regression'],
            results['df_residual'],
            results['df_total']
        ],
        'SS (Soma dos Quadrados)': [
            results['SS_regression'],
            results['SS_residual'],
            results['SS_total']
        ],
        'MS (Quadrado Médio)': [
            results['MS_regression'],
            results['MS_residual'],
            ''
        ],
        'F': [
            results['F_statistic'],
            '',
            ''
        ],
        'p-valor': [
            results['p_value_F'],
            '',
            ''
        ]
    }
    
    df = pd.DataFrame(anova_data)
    return df


def create_coefficients_table(results: Dict, X_name: str, y_name: str) -> pd.DataFrame:
    """
    Create table with regression coefficients
    
    Args:
        results: Dictionary with regression results
        X_name: Name of X variable
        y_name: Name of y variable
        
    Returns:
        DataFrame with coefficients
    """
    coef_data = {
        'Termo': ['Intercepto', X_name],
        'Coeficiente': results['coefficients'],
        'Erro Padrão': results['SE_coefficients'],
        't Ratio': results['t_statistics'],
        'p-valor': results['p_values'],
        'IC 95% Inferior': results['CI_lower'],
        'IC 95% Superior': results['CI_upper'],
        'VIF': ['', results['VIF']]  # VIF só para o preditor
    }
    
    df = pd.DataFrame(coef_data)
    return df


def create_summary_table(results: Dict, X_name: str, y_name: str) -> pd.DataFrame:
    """
    Create summary statistics table
    
    Args:
        results: Dictionary with regression results
        X_name: Name of X variable
        y_name: Name of y variable
        
    Returns:
        DataFrame with summary statistics
    """
    summary_data = {
        'Métrica': [
            'R²',
            'R² Ajustado',
            'Erro Padrão da Regressão',
            'RMSE',
            'MAE',
            'Correlação',
            'Número de Observações',
            f'Média de {y_name}',
            f'Média de {X_name}'
        ],
        'Valor': [
            f'{results["R_squared"]:.6f}',
            f'{results["R_squared_adj"]:.6f}',
            f'{results["SE_regression"]:.6f}',
            f'{results["RMSE"]:.6f}',
            f'{results["MAE"]:.6f}',
            f'{results["correlation"]:.6f}',
            f'{results["n"]}',
            f'{results["mean_y"]:.6f}',
            f'{results["mean_X"]:.6f}'
        ]
    }
    
    df = pd.DataFrame(summary_data)
    return df


def create_regression_plot(
    X: np.ndarray,
    y: np.ndarray,
    results: Dict,
    X_name: str,
    y_name: str,
    figsize: Tuple[int, int] = (10, 7)
) -> Figure:
    """
    Create scatter plot with regression line
    
    Args:
        X: Independent variable
        y: Dependent variable
        results: Regression results
        X_name: Name of X variable
        y_name: Name of y variable
        figsize: Figure size
        
    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Flatten X if needed
    X_plot = X.flatten()
    
    # Sort data for smooth line
    sort_idx = np.argsort(X_plot)
    X_sorted = X_plot[sort_idx]
    y_sorted = y[sort_idx]
    y_pred_sorted = results['predictions'][sort_idx]
    
    # Scatter plot
    ax.scatter(X_plot, y, alpha=0.6, s=50, color='steelblue', 
               edgecolors='black', linewidth=0.5, label='Dados Observados', zorder=3)
    
    # Regression line
    ax.plot(X_sorted, y_pred_sorted, color='red', linewidth=2, 
            label=f'Linha de Regressão\n$R^2$ = {results["R_squared"]:.4f}', zorder=4)
    
    # Confidence interval (95%)
    # Calculate standard error of prediction
    n = len(y)
    X_with_intercept = np.column_stack([np.ones(n), X])
    X_sorted_with_intercept = np.column_stack([np.ones(len(X_sorted)), X_sorted.reshape(-1, 1)])
    
    # Variance of predictions
    XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
    var_pred = results['MS_residual'] * np.sum(
        (X_sorted_with_intercept @ XtX_inv) * X_sorted_with_intercept, axis=1
    )
    SE_pred = np.sqrt(var_pred)
    
    # 95% CI
    t_critical = stats.t.ppf(0.975, results['df_residual'])
    CI_lower = y_pred_sorted - t_critical * SE_pred
    CI_upper = y_pred_sorted + t_critical * SE_pred
    
    ax.fill_between(X_sorted, CI_lower, CI_upper, alpha=0.2, color='red', 
                    label='IC 95%', zorder=2)
    
    # Labels and title
    ax.set_xlabel(X_name, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_name, fontsize=12, fontweight='bold')
    ax.set_title('Análise de Regressão Linear Simples', fontsize=14, fontweight='bold', pad=20)
    
    # Add regression equation
    intercept = results['coefficients'][0]
    slope = results['coefficients'][1]
    equation = f'{y_name} = {intercept:.4f} + {slope:.4f} × {X_name}'
    ax.text(0.05, 0.95, equation, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    return fig


def create_residuals_plot(
    X: np.ndarray,
    results: Dict,
    X_name: str,
    figsize: Tuple[int, int] = (12, 10)
) -> Figure:
    """
    Create residual diagnostic plots
    
    Args:
        X: Independent variable
        results: Regression results
        X_name: Name of X variable
        figsize: Figure size
        
    Returns:
        Matplotlib Figure with 4 subplots
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    residuals = results['residuals']
    y_pred = results['predictions']
    X_plot = X.flatten()
    
    # 1. Residuals vs Fitted
    ax1 = axes[0, 0]
    ax1.scatter(y_pred, residuals, alpha=0.6, s=50, color='steelblue',
                edgecolors='black', linewidth=0.5)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Valores Ajustados', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Resíduos', fontsize=11, fontweight='bold')
    ax1.set_title('Resíduos vs Valores Ajustados', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add loess smooth line
    from scipy.signal import savgol_filter
    if len(y_pred) > 5:
        sort_idx = np.argsort(y_pred)
        y_pred_sorted = y_pred[sort_idx]
        residuals_sorted = residuals[sort_idx]
        if len(y_pred_sorted) > 7:
            window = min(len(y_pred_sorted) // 3, 51)
            if window % 2 == 0:
                window += 1
            if window >= 3:
                smooth = savgol_filter(residuals_sorted, window, 2)
                ax1.plot(y_pred_sorted, smooth, color='blue', linewidth=2, alpha=0.7)
    
    # 2. Q-Q Plot (Normal probability plot)
    ax2 = axes[0, 1]
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Gráfico Q-Q Normal', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 3. Scale-Location (Spread-Location)
    ax3 = axes[1, 0]
    sqrt_abs_residuals = np.sqrt(np.abs(residuals / np.std(residuals)))
    ax3.scatter(y_pred, sqrt_abs_residuals, alpha=0.6, s=50, color='steelblue',
                edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Valores Ajustados', fontsize=11, fontweight='bold')
    ax3.set_ylabel('√|Resíduos Padronizados|', fontsize=11, fontweight='bold')
    ax3.set_title('Gráfico Scale-Location', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 4. Residuals vs X
    ax4 = axes[1, 1]
    ax4.scatter(X_plot, residuals, alpha=0.6, s=50, color='steelblue',
                edgecolors='black', linewidth=0.5)
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel(X_name, fontsize=11, fontweight='bold')
    ax4.set_ylabel('Resíduos', fontsize=11, fontweight='bold')
    ax4.set_title(f'Resíduos vs {X_name}', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.suptitle('Diagnóstico de Resíduos', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig


def create_histogram_residuals(
    results: Dict,
    figsize: Tuple[int, int] = (10, 6)
) -> Figure:
    """
    Create histogram of residuals with normal curve overlay
    
    Args:
        results: Regression results
        figsize: Figure size
        
    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    residuals = results['residuals']
    
    # Histogram
    n, bins, patches = ax.hist(residuals, bins=30, density=True, alpha=0.7, 
                                color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Fit normal distribution
    mu, sigma = np.mean(residuals), np.std(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
            label=f'Normal(μ={mu:.4f}, σ={sigma:.4f})')
    
    # Labels and title
    ax.set_xlabel('Resíduos', fontsize=12, fontweight='bold')
    ax.set_ylabel('Densidade', fontsize=12, fontweight='bold')
    ax.set_title('Distribuição dos Resíduos', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    
    plt.tight_layout()
    return fig
