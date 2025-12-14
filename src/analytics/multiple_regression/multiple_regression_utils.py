"""
Multiple Regression Utilities
Calculations for multiple linear regression analysis
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import List, Tuple, Dict

def create_interaction_terms(X_df: pd.DataFrame, interactions: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Create interaction terms from selected variable pairs
    
    Args:
        X_df: DataFrame with predictor variables
        interactions: List of tuples (var1, var2) for interactions
    
    Returns:
        DataFrame with original variables and interaction terms
    """
    result_df = X_df.copy()
    
    for var1, var2 in interactions:
        interaction_name = f"{var1} × {var2}"
        result_df[interaction_name] = X_df[var1] * X_df[var2]
    
    return result_df


def calculate_vif(X: np.ndarray, variable_names: List[str]) -> Dict[str, float]:
    """
    Calculate Variance Inflation Factor for each predictor
    
    Args:
        X: Design matrix (without intercept column)
        variable_names: Names of variables
    
    Returns:
        Dictionary with VIF for each variable
    """
    vif_dict = {}
    
    # Need at least 2 predictors for VIF
    if X.shape[1] < 2:
        return {var: 1.0 for var in variable_names}
    
    for i, var_name in enumerate(variable_names):
        # Regress variable i on all other variables
        X_i = X[:, i]
        X_others = np.delete(X, i, axis=1)
        
        # Add intercept to X_others
        X_others_with_intercept = np.column_stack([np.ones(len(X_others)), X_others])
        
        try:
            # OLS for this variable
            coeffs = np.linalg.lstsq(X_others_with_intercept, X_i, rcond=None)[0]
            predictions = X_others_with_intercept @ coeffs
            
            # Calculate R²
            ss_res = np.sum((X_i - predictions) ** 2)
            ss_tot = np.sum((X_i - np.mean(X_i)) ** 2)
            
            if ss_tot == 0:
                r_squared = 0
            else:
                r_squared = 1 - (ss_res / ss_tot)
            
            # VIF = 1 / (1 - R²)
            if r_squared >= 0.9999:  # Avoid division by near-zero
                vif = 999.99
            else:
                vif = 1 / (1 - r_squared)
            
            vif_dict[var_name] = vif
        except:
            vif_dict[var_name] = np.nan
    
    return vif_dict


def calculate_multiple_regression(X: np.ndarray, y: np.ndarray, variable_names: List[str]) -> Dict:
    """
    Calculate multiple linear regression using OLS
    
    Args:
        X: Design matrix (n x p) without intercept
        y: Response vector (n,)
        variable_names: Names of predictor variables
    
    Returns:
        Dictionary with regression results
    """
    n = len(y)
    
    # Add intercept column
    X_with_intercept = np.column_stack([np.ones(n), X])
    
    # Calculate coefficients: β = (X'X)^-1 X'y
    XtX = X_with_intercept.T @ X_with_intercept
    Xty = X_with_intercept.T @ y
    
    try:
        coefficients = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        # Singular matrix - use pseudoinverse
        coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
    
    # Predictions
    y_pred = X_with_intercept @ coefficients
    
    # Residuals
    residuals = y - y_pred
    
    # Sum of squares
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum(residuals ** 2)
    ss_regression = ss_total - ss_residual
    
    # Degrees of freedom
    p = X.shape[1]  # Number of predictors (excluding intercept)
    df_regression = p
    df_residual = n - p - 1
    df_total = n - 1
    
    # Mean squares
    ms_regression = ss_regression / df_regression if df_regression > 0 else 0
    ms_residual = ss_residual / df_residual if df_residual > 0 else 0
    
    # F-statistic
    if ms_residual > 0:
        f_statistic = ms_regression / ms_residual
        f_pvalue = 1 - stats.f.cdf(f_statistic, df_regression, df_residual)
    else:
        f_statistic = np.inf
        f_pvalue = 0.0
    
    # R² and adjusted R²
    r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    
    if n > p + 1:
        r_squared_adj = 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))
    else:
        r_squared_adj = r_squared
    
    # Standard error of coefficients
    if ms_residual > 0:
        var_coeff = ms_residual * np.linalg.inv(XtX)
        se_coefficients = np.sqrt(np.diag(var_coeff))
    else:
        se_coefficients = np.zeros(len(coefficients))
    
    # t-statistics and p-values
    t_statistics = np.zeros(len(coefficients))
    p_values = np.ones(len(coefficients))
    
    for i in range(len(coefficients)):
        if se_coefficients[i] > 0:
            t_statistics[i] = coefficients[i] / se_coefficients[i]
            p_values[i] = 2 * (1 - stats.t.cdf(abs(t_statistics[i]), df_residual))
        else:
            t_statistics[i] = np.inf if coefficients[i] != 0 else 0
            p_values[i] = 0.0 if coefficients[i] != 0 else 1.0
    
    # 95% Confidence intervals
    t_critical = stats.t.ppf(0.975, df_residual)
    ci_lower = coefficients - t_critical * se_coefficients
    ci_upper = coefficients + t_critical * se_coefficients
    
    # Calculate VIF for predictors (excluding intercept)
    vif_dict = calculate_vif(X, variable_names)
    
    # Additional metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    # AIC and BIC
    n_params = len(coefficients)
    if ms_residual > 0:
        log_likelihood = -n/2 * (np.log(2*np.pi) + np.log(ss_residual/n) + 1)
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n) - 2 * log_likelihood
    else:
        aic = np.inf
        bic = np.inf
    
    # Standardized residuals
    if ms_residual > 0:
        standardized_residuals = residuals / np.sqrt(ms_residual)
    else:
        standardized_residuals = np.zeros(n)
    
    return {
        'coefficients': coefficients,
        'se_coefficients': se_coefficients,
        't_statistics': t_statistics,
        'p_values': p_values,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'vif': vif_dict,
        'r_squared': r_squared,
        'r_squared_adj': r_squared_adj,
        'f_statistic': f_statistic,
        'f_pvalue': f_pvalue,
        'ss_regression': ss_regression,
        'ss_residual': ss_residual,
        'ss_total': ss_total,
        'ms_regression': ms_regression,
        'ms_residual': ms_residual,
        'df_regression': df_regression,
        'df_residual': df_residual,
        'df_total': df_total,
        'rmse': rmse,
        'mae': mae,
        'aic': aic,
        'bic': bic,
        'y_pred': y_pred,
        'residuals': residuals,
        'standardized_residuals': standardized_residuals,
        'n_obs': n,
        'n_predictors': p,
        'y_mean': np.mean(y),
        'variable_names': ['Intercepto'] + variable_names
    }


def backward_elimination(X: np.ndarray, y: np.ndarray, variable_names: List[str], 
                        alpha: float = 0.05) -> Tuple[List[int], Dict]:
    """
    Perform backward elimination to select significant variables
    
    Args:
        X: Design matrix (without intercept)
        y: Response vector
        variable_names: Names of variables
        alpha: Significance level for removal
    
    Returns:
        Tuple of (indices of selected variables, final model results)
    """
    n_vars = X.shape[1]
    selected_indices = list(range(n_vars))
    
    while len(selected_indices) > 0:
        # Fit model with current variables
        X_selected = X[:, selected_indices]
        current_names = [variable_names[i] for i in selected_indices]
        
        results = calculate_multiple_regression(X_selected, y, current_names)
        
        # Find variable with highest p-value (excluding intercept)
        p_values = results['p_values'][1:]  # Exclude intercept
        
        if len(p_values) == 0:
            break
        
        max_pvalue_idx = np.argmax(p_values)
        max_pvalue = p_values[max_pvalue_idx]
        
        # If highest p-value > alpha, remove that variable
        if max_pvalue > alpha:
            removed_idx = selected_indices[max_pvalue_idx]
            selected_indices.remove(removed_idx)
        else:
            # All remaining variables are significant
            break
    
    # Final model
    if len(selected_indices) > 0:
        X_final = X[:, selected_indices]
        final_names = [variable_names[i] for i in selected_indices]
        final_results = calculate_multiple_regression(X_final, y, final_names)
    else:
        # No variables selected - use only intercept
        final_results = {
            'coefficients': np.array([np.mean(y)]),
            'variable_names': ['Intercepto'],
            'n_predictors': 0,
            'r_squared': 0,
            'r_squared_adj': 0
        }
    
    return selected_indices, final_results


def create_anova_table(results: Dict) -> pd.DataFrame:
    """Create ANOVA table DataFrame"""
    anova_data = {
        'Fonte': ['Regressão', 'Residual', 'Total'],
        'DF': [results['df_regression'], results['df_residual'], results['df_total']],
        'SS': [results['ss_regression'], results['ss_residual'], results['ss_total']],
        'MS': [results['ms_regression'], results['ms_residual'], ''],
        'F-Statistic': [results['f_statistic'], '', ''],
        'P-Value': [results['f_pvalue'], '', '']
    }
    
    return pd.DataFrame(anova_data)


def create_coefficients_table(results: Dict) -> pd.DataFrame:
    """Create coefficients table DataFrame"""
    coef_data = {
        'Termo': results['variable_names'],
        'Estimativa': results['coefficients'],
        'Erro Padrão': results['se_coefficients'],
        't-Ratio': results['t_statistics'],
        'P-Value': results['p_values'],
        'IC 95% Inferior': results['ci_lower'],
        'IC 95% Superior': results['ci_upper']
    }
    
    # Add VIF column
    vif_values = ['-']  # Intercept has no VIF
    for var_name in results['variable_names'][1:]:
        vif_values.append(results['vif'].get(var_name, np.nan))
    
    coef_data['VIF'] = vif_values
    
    return pd.DataFrame(coef_data)


def create_summary_table(results: Dict, y_name: str) -> pd.DataFrame:
    """Create model summary table DataFrame"""
    summary_data = {
        'Métrica': [
            'R²',
            'R² Ajustado',
            'RMSE',
            'MAE',
            'AIC',
            'BIC',
            'Nº Observações',
            'Nº Preditores',
            f'Média de {y_name}'
        ],
        'Valor': [
            results['r_squared'],
            results['r_squared_adj'],
            results['rmse'],
            results['mae'],
            results['aic'],
            results['bic'],
            results['n_obs'],
            results['n_predictors'],
            results['y_mean']
        ]
    }
    
    return pd.DataFrame(summary_data)


def create_regression_plot(y_true: np.ndarray, y_pred: np.ndarray, y_name: str, 
                          results: Dict):
    """Create actual vs predicted plot"""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='black', linewidths=0.5, s=50)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predição Perfeita')
    
    # Labels and title
    ax.set_xlabel(f'{y_name} (Real)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{y_name} (Predito)', fontsize=12, fontweight='bold')
    ax.set_title(f'Valores Preditos vs Reais\nR² = {results["r_squared"]:.5f}', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_residuals_plot(y_true: np.ndarray, results: Dict, y_name: str):
    """Create 4-panel residual diagnostic plots"""
    import matplotlib.pyplot as plt
    
    y_pred = results['y_pred']
    residuals = results['residuals']
    standardized_residuals = results['standardized_residuals']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Residuals vs Fitted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6, edgecolors='black', linewidths=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 0].set_xlabel('Valores Ajustados', fontweight='bold')
    axes[0, 0].set_ylabel('Resíduos', fontweight='bold')
    axes[0, 0].set_title('Resíduos vs Valores Ajustados', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Q-Q Plot
    stats.probplot(standardized_residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot Normal', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Scale-Location (sqrt of standardized residuals)
    sqrt_std_resid = np.sqrt(np.abs(standardized_residuals))
    axes[1, 0].scatter(y_pred, sqrt_std_resid, alpha=0.6, edgecolors='black', linewidths=0.5)
    axes[1, 0].set_xlabel('Valores Ajustados', fontweight='bold')
    axes[1, 0].set_ylabel('√|Resíduos Padronizados|', fontweight='bold')
    axes[1, 0].set_title('Scale-Location', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Residuals vs Order
    axes[1, 1].scatter(range(len(residuals)), residuals, alpha=0.6, edgecolors='black', linewidths=0.5)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Ordem das Observações', fontweight='bold')
    axes[1, 1].set_ylabel('Resíduos', fontweight='bold')
    axes[1, 1].set_title('Resíduos vs Ordem', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_histogram_residuals(results: Dict):
    """Create histogram of residuals with normal curve overlay"""
    import matplotlib.pyplot as plt
    
    residuals = results['residuals']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Histogram
    n, bins, patches = ax.hist(residuals, bins=30, density=True, alpha=0.7, 
                               color='skyblue', edgecolor='black')
    
    # Fit normal distribution
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
    
    ax.set_xlabel('Resíduos', fontsize=12, fontweight='bold')
    ax.set_ylabel('Densidade', fontsize=12, fontweight='bold')
    ax.set_title('Distribuição dos Resíduos', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
