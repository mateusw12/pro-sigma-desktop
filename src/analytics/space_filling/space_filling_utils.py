"""
Space Filling Design - Funções de cálculo
Reaproveitamento do backend Python existente
"""

import statsmodels.api as sm
from src.utils.lazy_imports import get_numpy, get_scipy_stats, get_scipy_spatial_distance, get_pyDOE2


def generate_lhs(n_samples, n_factors, response_values=None, n_response_cols=1):
    """Gera Latin Hypercube Sample"""
    pyDOE2 = get_pyDOE2()
    design = pyDOE2.lhs(n_factors, samples=n_samples, criterion=None)
    
    # Converte para DataFrame-like dict
    result = {}
    for i in range(n_factors):
        result[f"X{i+1}"] = design[:, i].tolist()
    
    # Adiciona colunas de resposta
    for i in range(n_response_cols):
        if response_values and len(response_values) > 0:
            result[f"Y{i+1}"] = response_values
        else:
            result[f"Y{i+1}"] = [None] * n_samples
    
    return result


def generate_lhs_min(n_samples, n_factors, iterations, response_values=None, n_response_cols=1):
    """Gera LHS minimizando distância mínima"""
    np = get_numpy()
    pyDOE2 = get_pyDOE2()
    scipy_distance = get_scipy_spatial_distance()
    
    best_design = None
    best_min_dist = 0
    
    for _ in range(iterations):
        design = pyDOE2.lhs(n_factors, samples=n_samples, criterion=None)
        distances = scipy_distance.pdist(design)
        min_dist = np.min(distances)
        
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_design = design
    
    result = {}
    for i in range(n_factors):
        result[f"X{i+1}"] = best_design[:, i].tolist()
    
    for i in range(n_response_cols):
        if response_values and len(response_values) > 0:
            result[f"Y{i+1}"] = response_values
        else:
            result[f"Y{i+1}"] = [None] * n_samples
    
    return result


def generate_lhs_max(n_samples, n_factors, iterations, response_values=None, n_response_cols=1):
    """Gera LHS maximizando distância média"""
    np = get_numpy()
    pyDOE2 = get_pyDOE2()
    scipy_distance = get_scipy_spatial_distance()
    
    best_design = None
    best_mean_dist = 0
    
    for _ in range(iterations):
        design = pyDOE2.lhs(n_factors, samples=n_samples, criterion=None)
        distances = scipy_distance.pdist(design)
        mean_dist = np.mean(distances)
        
        if mean_dist > best_mean_dist:
            best_mean_dist = mean_dist
            best_design = design
    
    result = {}
    for i in range(n_factors):
        result[f"X{i+1}"] = best_design[:, i].tolist()
    
    for i in range(n_response_cols):
        if response_values and len(response_values) > 0:
            result[f"Y{i+1}"] = response_values
        else:
            result[f"Y{i+1}"] = [None] * n_samples
    
    return result


def generate_sphere_packing(n_factors, n_samples, n_response_cols, response_values=None):
    """Gera Sphere Packing design"""
    pyDOE2 = get_pyDOE2()
    
    # Implementação simplificada - usa LHS como base
    design = pyDOE2.lhs(n_factors, samples=n_samples, criterion='maximin')
    
    result = {}
    for i in range(n_factors):
        result[f"X{i+1}"] = design[:, i].tolist()
    
    for i in range(n_response_cols):
        if response_values and len(response_values) > 0:
            result[f"Y{i+1}"] = response_values
        else:
            result[f"Y{i+1}"] = [None] * n_samples
    
    return result


def scale_to_range(values, min_val, max_val):
    """Escala valores de [0,1] para [min_val, max_val]"""
    return [min_val + v * (max_val - min_val) for v in values]


def calculate_space_filling_analysis(data, response_columns, interaction_columns=None, recalculate=False):
    """
    Calcula análise Space Filling Design
    
    Args:
        data: DataFrame com os dados
        response_columns: Lista de colunas de resposta
        interaction_columns: Lista de interações (opcional)
        recalculate: Se True, é modelo reduzido
    
    Returns:
        Dict com resultados para cada Y
    """
    np = get_numpy()
    pd = __import__('pandas')
    stats = get_scipy_stats()
    
    df = pd.DataFrame(data)
    
    results = {}
    
    for response_col in response_columns:
        # Separa X e Y
        y = df[response_col].values
        X_cols = [col for col in df.columns if col not in response_columns]
        X = df[X_cols].values
        
        # Calcula médias para centralização
        means = {col: float(df[col].mean()) for col in X_cols}
        
        # Constrói matriz X com intercepto e termos lineares
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        # Lista para armazenar nomes dos parâmetros
        param_names = ['Intercept'] + X_cols.copy()
        
        # Adiciona interações/quadráticos se especificados
        quadratic_terms = {}  # {nome: (idx_coluna, centralizado)}
        interaction_names = []
        
        if interaction_columns:
            for interaction in interaction_columns:
                # Termos quadráticos são identificados por "/" (ex: "A/A")
                if '/' in interaction:
                    var_name = interaction.split('/')[0]
                    if var_name in X_cols:
                        idx = X_cols.index(var_name)
                        # Termo quadrático CENTRALIZADO: (X - mean)^2
                        centered_term = (X[:, idx] - means[var_name]) ** 2
                        X_with_intercept = np.column_stack([X_with_intercept, centered_term])
                        quadratic_terms[interaction] = idx
                        param_names.append(interaction)
                        interaction_names.append(interaction)
                # Termos de interação (X1*X2)
                elif '*' in interaction:
                    terms = interaction.split('*')
                    if len(terms) == 2:
                        col1, col2 = terms
                        if col1 in X_cols and col2 in X_cols:
                            idx1 = X_cols.index(col1)
                            idx2 = X_cols.index(col2)
                            interaction_term = X[:, idx1] * X[:, idx2]
                            X_with_intercept = np.column_stack([X_with_intercept, interaction_term])
                            param_names.append(interaction)
                            interaction_names.append(interaction)
        
        # Calcula betas (regressão linear)
        try:
            betas = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        except:
            results[response_col] = {"error": "Erro ao calcular betas"}
            continue
        
        # Criar modelo statsmodels para profiler incluindo TODOS os termos
        # Precisamos criar DataFrame com termos lineares + quadráticos + interações
        X_df_full = pd.DataFrame(X, columns=X_cols)
        
        # Adicionar termos quadráticos e interações ao DataFrame para statsmodels
        if interaction_columns:
            for interaction in interaction_columns:
                if '/' in interaction:
                    # Termo quadrático centralizado
                    var_name = interaction.split('/')[0]
                    if var_name in X_cols:
                        X_df_full[interaction] = (X_df_full[var_name] - means[var_name]) ** 2
                elif '*' in interaction:
                    # Termo de interação
                    terms = interaction.split('*')
                    if len(terms) == 2:
                        col1, col2 = terms
                        if col1 in X_cols and col2 in X_cols:
                            X_df_full[interaction] = X_df_full[col1] * X_df_full[col2]
        
        X_with_const = sm.add_constant(X_df_full)
        ols_model = sm.OLS(y, X_with_const).fit()
        
        # Armazenar informações sobre termos quadráticos para o profiler
        quadratic_info = {name: means[name.split('/')[0]] for name in quadratic_terms.keys()}
        
        # Predições
        y_pred = X_with_intercept @ betas
        
        # Resíduos
        residuals = y - y_pred
        
        # ANOVA
        y_mean = np.mean(y)
        ss_total = np.sum((y - y_mean) ** 2)
        ss_model = np.sum((y_pred - y_mean) ** 2)
        ss_error = np.sum(residuals ** 2)
        
        df_model = X_with_intercept.shape[1] - 1
        df_error = len(y) - X_with_intercept.shape[1]
        df_total = len(y) - 1
        
        ms_model = ss_model / df_model if df_model > 0 else 0
        ms_error = ss_error / df_error if df_error > 0 else 0
        
        f_ratio = ms_model / ms_error if ms_error > 0 else 0
        p_value = 1 - stats.f.cdf(f_ratio, df_model, df_error) if f_ratio > 0 else 1
        
        # R-squared
        r_squared = ss_model / ss_total if ss_total > 0 else 0
        r_squared_adj = 1 - (1 - r_squared) * (df_total / df_error) if df_error > 0 else 0
        
        # RMSE
        rmse = np.sqrt(ms_error)
        
        # Parameter Estimates
        # Matriz de covariância dos betas
        try:
            cov_matrix = ms_error * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            std_errors = np.sqrt(np.diag(cov_matrix))
            t_stats = betas / std_errors
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_error))
        except:
            std_errors = np.zeros(len(betas))
            t_stats = np.zeros(len(betas))
            p_values = np.ones(len(betas))
        
        # Nomeia parâmetros (já definidos anteriormente em param_names)
        parameter_estimates = {}
        for i, name in enumerate(param_names[:len(betas)]):
            parameter_estimates[name] = {
                'estimates': float(betas[i]),
                'stdError': float(std_errors[i]),
                'tRatio': float(t_stats[i]),
                'pValue': float(p_values[i])
            }
        
        # Lack of Fit (simplificado)
        lack_of_fit = {
            'grausLiberdade': {'lackOfFit': 0, 'pureError': 0},
            'sQuadrados': {'lackOfFit': 0, 'pureError': 0},
            'mQuadrados': {'lackOfFit': 0, 'pureError': 0},
            'fRatio': 0,
            'probF': 1
        }
        
        # Ordena Y e Y_pred para gráfico overlay
        sorted_indices = np.argsort(y)
        y_sorted = y[sorted_indices].tolist()
        y_pred_sorted = y_pred[sorted_indices].tolist()
        
        results[response_col] = {
            'model': ols_model,  # Statsmodels model for profiler (com todos os termos)
            'X_cols': X_cols,  # Column names for profiler (apenas fatores principais)
            'quadratic_terms': list(quadratic_terms.keys()) if quadratic_terms else [],  # Termos quadráticos
            'interaction_terms': interaction_names,  # Termos de interação
            'means': means,  # Médias para centralização dos quadráticos
            'param_names': param_names,  # Nomes dos parâmetros na ordem correta
            'betas': betas.tolist(),
            'anovaTable': {
                'grausLiberdade': {
                    'modelo': df_model,
                    'erro': df_error,
                    'total': df_total
                },
                'sQuadrados': {
                    'modelo': float(ss_model),
                    'erro': float(ss_error),
                    'total': float(ss_total)
                },
                'mQuadrados': {
                    'modelo': float(ms_model),
                    'erro': float(ms_error)
                },
                'fRatio': float(f_ratio),
                'probF': float(p_value)
            },
            'summarOfFit': {
                'rQuadrado': float(r_squared),
                'rQuadradoAjustado': float(r_squared_adj),
                'rmse': float(rmse),
                'media': float(y_mean),
                'observacoes': len(y)
            },
            'lackOfFit': lack_of_fit,
            'parameterEstimates': parameter_estimates,
            'isRecalculate': recalculate,
            'yPredicteds': y_pred.tolist(),
            'yPredictedsOdered': y_pred_sorted,
            'y': y_sorted,
            'mean': means
        }
    
    return results


def validate_space_filling_data(data, response_columns):
    """Valida dados para Space Filling"""
    if data is None or len(data) == 0:
        return False, "Dados vazios"
    
    if not response_columns or len(response_columns) == 0:
        return False, "Nenhuma coluna de resposta selecionada"
    
    # Verifica se há pelo menos 2 colunas X
    x_columns = [col for col in data.columns if col not in response_columns]
    if len(x_columns) < 1:
        return False, "É necessária pelo menos 1 variável X"
    
    # Verifica valores faltantes
    if data.isnull().any().any():
        return False, "Dados contêm valores faltantes (NaN)"
    
    return True, ""


def generate_equation(betas, param_names, means):
    """Gera equação do modelo no formato JMP (termos lineares não centralizados, quadráticos centralizados)"""
    equation = "Y = "
    
    for i, (beta, name) in enumerate(zip(betas, param_names)):
        if i == 0:
            # Intercepto
            equation += f"{beta:.4f}"
        elif '/' in name:
            # Termo quadrático: (X - média)²
            var_name = name.split('/')[0]
            equation += f" + ({var_name} - ({means[var_name]:.7f})) * (({var_name} - ({means[var_name]:.7f})) * ({beta:.7f}))"
        else:
            # Termo linear (não centralizado) ou interação
            sign = "+" if beta >= 0 else "-"
            equation += f" {sign} {abs(beta):.7f} * {name}"
    
    return equation
