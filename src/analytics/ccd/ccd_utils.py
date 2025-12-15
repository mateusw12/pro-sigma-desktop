"""
Funções utilitárias para Central Composite Design (CCD)
Gera experimentos e calcula ANOVA com Lack of Fit
"""

from src.utils.lazy_imports import get_numpy, get_pandas, get_scipy_stats
from typing import List, Dict
import itertools


def generate_ccd_design(n_factors, n_center, design_type="rotatable", n_responses=1, response_values=None):
    """
    Gera um design Central Composite Design (CCD) ou Box-Behnken
    
    Args:
        n_factors: Número de fatores
        n_center: Número de pontos centrais
        design_type: "rotatable", "orthogonal", ou "bbd" (Box-Behnken)
        n_responses: Número de colunas de resposta
        response_values: Lista de valores para preencher respostas (ou None para 99999)
    
    Returns:
        DataFrame com o design gerado
    """
    np = get_numpy()
    pd = get_pandas()
    
    if design_type == "bbd":
        # Box-Behnken Design
        design = _generate_box_behnken(n_factors, n_center)
    else:
        # CCD padrão (rotatable ou orthogonal)
        design = _generate_ccd(n_factors, n_center, design_type)
    
    # Cria DataFrame
    columns = [f'x{i+1}' for i in range(n_factors)]
    df = pd.DataFrame(design, columns=columns)
    
    # Embaralha as linhas
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Adiciona colunas de resposta
    if response_values and len(response_values) > 0:
        for i in range(n_responses):
            col_name = f'Y{i+1}'
            # Amostra aleatória dos valores fornecidos
            df[col_name] = np.random.choice(response_values, size=len(df))
    else:
        for i in range(n_responses):
            col_name = f'Y{i+1}'
            df[col_name] = 99999
    
    return df


def _generate_ccd(n_factors, n_center, alpha_type):
    """Gera CCD padrão"""
    np = get_numpy()
    
    # Pontos fatoriais (2^k)
    factorial = list(itertools.product([-1, 1], repeat=n_factors))
    factorial = np.array(factorial)
    
    # Pontos axiais
    if alpha_type == "rotatable":
        alpha = n_factors ** 0.25  # Rotacional
    else:  # orthogonal
        # Para ortogonal, alpha depende do número de pontos
        n_factorial = 2 ** n_factors
        n_axial = 2 * n_factors
        alpha = ((n_factorial * (1 + n_factorial)) / (n_factorial + n_axial + n_center)) ** 0.25
    
    axial = []
    for i in range(n_factors):
        point_pos = [0] * n_factors
        point_pos[i] = alpha
        axial.append(point_pos)
        
        point_neg = [0] * n_factors
        point_neg[i] = -alpha
        axial.append(point_neg)
    
    axial = np.array(axial)
    
    # Pontos centrais
    center = np.zeros((n_center, n_factors))
    
    # Combina todos os pontos
    design = np.vstack([factorial, axial, center])
    
    return design


def _generate_box_behnken(n_factors, n_center):
    """Gera Box-Behnken Design"""
    np = get_numpy()
    
    if n_factors < 3:
        raise ValueError("Box-Behnken requer pelo menos 3 fatores")
    
    # Gera combinações de 2 fatores por vez
    design = []
    
    for i, j in itertools.combinations(range(n_factors), 2):
        # Para cada par de fatores, cria um design fatorial 2x2
        for xi, xj in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            point = [0] * n_factors
            point[i] = xi
            point[j] = xj
            design.append(point)
    
    design = np.array(design)
    
    # Remove duplicatas
    design = np.unique(design, axis=0)
    
    # Adiciona pontos centrais
    center = np.zeros((n_center, n_factors))
    design = np.vstack([design, center])
    
    return design


def calculate_ccd_analysis(df, response_column, quadratic_terms=None, interaction_terms=None):
    """
    Calcula análise CCD com ANOVA completa
    
    Args:
        df: DataFrame com dados
        response_column: Nome da coluna de resposta (Y)
        quadratic_terms: Lista de colunas para termos quadráticos (None = nenhum)
        interaction_terms: Lista de tuplas (col1, col2) para interações (None = nenhum)
    
    Returns:
        dict com parameter_estimates, anova_table, summary_of_fit, lack_of_fit, predictions
    """
    np = get_numpy()
    pd = get_pandas()
    from scipy import stats
    
    df = df.copy()
    
    # Renomeia coluna de resposta para Y
    df = df.rename(columns={response_column: 'Y'})
    
    # Identifica colunas X (não Y)
    x_columns = [col for col in df.columns if not col.startswith('Y')]
    
    # Adiciona termos quadráticos selecionados
    if quadratic_terms:
        for col in quadratic_terms:
            if col in x_columns:
                df[f'{col}_squared'] = df[col] ** 2
    
    # Adiciona termos de interação selecionados
    if interaction_terms:
        for col1, col2 in interaction_terms:
            if col1 in x_columns and col2 in x_columns:
                df[f'{col1}_interaction_{col2}'] = df[col1] * df[col2]
    
    # Prepara X e Y para regressão
    feature_columns = [col for col in df.columns if col != 'Y']
    X = df[feature_columns].values
    y = df['Y'].values
    
    # Adiciona intercepto
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    
    # Regressão linear (mínimos quadrados)
    coefficients, residuals, rank, s = np.linalg.lstsq(X_with_intercept, y, rcond=None)
    
    # Predições
    y_pred = X_with_intercept @ coefficients
    
    # Resíduos
    residuals_array = y - y_pred
    
    # Graus de liberdade
    n = len(y)
    k = len(coefficients) - 1  # número de preditores (sem intercepto)
    df_model = k
    df_error = n - k - 1
    df_total = n - 1
    
    # Soma de quadrados
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_model = np.sum((y_pred - np.mean(y)) ** 2)
    ss_error = np.sum(residuals_array ** 2)
    
    # Média dos quadrados
    ms_model = ss_model / df_model if df_model > 0 else 0
    ms_error = ss_error / df_error if df_error > 0 else 0
    
    # F-statistic
    f_value = ms_model / ms_error if ms_error > 0 else 0
    p_value = stats.f.sf(f_value, df_model, df_error) if f_value > 0 else 1
    
    # R²
    r_squared = 1 - (ss_error / ss_total) if ss_total > 0 else 0
    
    # R² Ajustado
    r_squared_adj = 1 - ((1 - r_squared) * (n - 1) / (n - k - 1)) if (n - k - 1) > 0 else 0
    
    # RMSE
    rmse = np.sqrt(ms_error)
    
    # Erros padrão dos coeficientes
    if ms_error > 0:
        # Covariância dos coeficientes
        XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        var_coef = ms_error * np.diag(XtX_inv)
        se_coef = np.sqrt(var_coef)
        
        # t-statistic
        t_values = coefficients / se_coef
        p_values_coef = 2 * (1 - stats.t.cdf(np.abs(t_values), df_error))
    else:
        se_coef = np.zeros_like(coefficients)
        t_values = np.zeros_like(coefficients)
        p_values_coef = np.ones_like(coefficients)
    
    # Tabela de estimativas de parâmetros
    param_names = ['Intercept'] + feature_columns
    parameter_estimates = pd.DataFrame({
        'term': param_names,
        'estimate': coefficients,
        'stdError': se_coef,
        't_value': t_values,
        'prob': p_values_coef
    })
    
    # Tabela ANOVA
    anova_table = pd.DataFrame({
        'source': ['Model', 'Error', 'Total'],
        'df': [df_model, df_error, df_total],
        'sm': [ss_model, ss_error, ss_total],
        'ms': [ms_model, ms_error, 0],
        'fValue': [f_value, 0, 0],
        'prob': [p_value, 0, 0]
    })
    
    # Summary of Fit
    summary_of_fit = pd.DataFrame({
        'metric': ['r2', 'r2adjust', 'rmse', 'mean', 'observation'],
        'value': [r_squared, r_squared_adj, rmse, np.mean(y), n]
    })
    
    # Lack of Fit
    lack_of_fit = calculate_lack_of_fit(df, y, y_pred, df_error, ss_error)
    
    # Equação do modelo
    equation = _generate_equation(coefficients, param_names)
    
    return {
        'parameter_estimates': parameter_estimates,
        'anova_table': anova_table,
        'summary_of_fit': summary_of_fit,
        'lack_of_fit': lack_of_fit,
        'y': y.tolist(),
        'y_predicted': y_pred.tolist(),
        'equation': equation
    }


def _add_all_terms(df, x_columns):
    """Adiciona todos os termos quadráticos e de interação"""
    np = get_numpy()
    
    # Termos quadráticos
    for col in x_columns:
        df[f'{col}_squared'] = df[col] ** 2
    
    # Termos de interação (2 a 2)
    for col1, col2 in itertools.combinations(x_columns, 2):
        df[f'{col1}_interaction_{col2}'] = df[col1] * df[col2]
    
    return df


def _add_specific_terms(df, columns_used, x_columns):
    """Adiciona apenas termos específicos"""
    np = get_numpy()
    
    if not columns_used:
        return df
    
    for term in columns_used:
        if '_squared' in term or '/' in term:
            # Termo quadrático
            base_col = term.replace('_squared', '').replace('/', '-').split('-')[0]
            if base_col in x_columns:
                df[f'{base_col}_squared'] = df[base_col] ** 2
        
        elif '_interaction_' in term or '-' in term:
            # Termo de interação
            if '_interaction_' in term:
                cols = term.split('_interaction_')
            else:
                cols = term.split('-')
            
            if len(cols) == 2 and all(c in x_columns for c in cols):
                df[f'{cols[0]}_interaction_{cols[1]}'] = df[cols[0]] * df[cols[1]]
    
    return df


def calculate_lack_of_fit(df, y, y_pred, df_error, ss_error):
    """
    Calcula Lack of Fit
    
    Args:
        df: DataFrame original (sem Y)
        y: Valores observados
        y_pred: Valores preditos
        df_error: Graus de liberdade do erro
        ss_error: Soma de quadrados do erro
    
    Returns:
        dict com estatísticas de Lack of Fit
    """
    np = get_numpy()
    pd = get_pandas()
    from scipy import stats
    
    # Remove coluna Y se existir
    df_x = df.drop(columns=['Y'], errors='ignore')
    
    # Encontra pontos duplicados (mesmas coordenadas X)
    duplicates = {}
    for idx, row in df_x.iterrows():
        key = '-'.join([str(v) for v in row.values])
        if key not in duplicates:
            duplicates[key] = []
        duplicates[key].append(idx)
    
    # Filtra apenas duplicatas reais (2+ pontos)
    duplicates = {k: v for k, v in duplicates.items() if len(v) >= 2}
    
    if not duplicates:
        # Sem duplicatas, não é possível calcular erro puro
        return {
            'grausLiberdade': {'lackOfFit': 0, 'erroPuro': 0, 'total': 0},
            'sQuadrados': {'lackOfFit': 0, 'erroPuro': 0, 'total': 0},
            'mQuadrados': {'lackOfFit': 0, 'erroPuro': 0},
            'fRatio': 0,
            'probF': 0
        }
    
    # Calcula erro puro
    pure_error_ss = 0
    pure_error_df = 0
    
    for indices in duplicates.values():
        y_group = y[indices]
        mean_group = np.mean(y_group)
        pure_error_ss += np.sum((y_group - mean_group) ** 2)
        pure_error_df += len(indices) - 1
    
    # Lack of Fit
    lack_of_fit_ss = ss_error - pure_error_ss
    lack_of_fit_df = df_error - pure_error_df
    
    # Média dos quadrados
    ms_lack_of_fit = lack_of_fit_ss / lack_of_fit_df if lack_of_fit_df > 0 else 0
    ms_pure_error = pure_error_ss / pure_error_df if pure_error_df > 0 else 0
    
    # F-ratio
    f_ratio = ms_lack_of_fit / ms_pure_error if ms_pure_error > 0 else 0
    
    # p-value
    if f_ratio > 0 and pure_error_df > 0:
        prob_f = stats.f.sf(f_ratio, lack_of_fit_df, pure_error_df)
    else:
        prob_f = 0
    
    return {
        'grausLiberdade': {
            'lackOfFit': int(lack_of_fit_df),
            'erroPuro': int(pure_error_df),
            'total': int(df_error)
        },
        'sQuadrados': {
            'lackOfFit': float(lack_of_fit_ss),
            'erroPuro': float(pure_error_ss),
            'total': float(ss_error)
        },
        'mQuadrados': {
            'lackOfFit': float(ms_lack_of_fit),
            'erroPuro': float(ms_pure_error)
        },
        'fRatio': float(f_ratio),
        'probF': float(prob_f)
    }


def _generate_equation(coefficients, param_names):
    """Gera equação do modelo em formato legível"""
    equation = f"Y = {coefficients[0]:.4f}"
    
    for i in range(1, len(coefficients)):
        coef = coefficients[i]
        name = param_names[i]
        
        # Formata nome do termo
        name = name.replace('_interaction_', ' * ')
        name = name.replace('_squared', '²')
        
        if coef >= 0:
            equation += f" + {coef:.4f}*{name}"
        else:
            equation += f" - {abs(coef):.4f}*{name}"
    
    return equation


def split_dataframes_by_response(df, response_columns):
    """
    Divide DataFrame em múltiplos DataFrames, um para cada coluna de resposta
    
    Args:
        df: DataFrame original
        response_columns: Lista de colunas de resposta (Y1, Y2, etc)
    
    Returns:
        dict com DataFrames separados por coluna de resposta
    """
    pd = get_pandas()
    
    result = {}
    
    # Colunas X (não-resposta)
    x_columns = [col for col in df.columns if col not in response_columns]
    
    for response_col in response_columns:
        # Cria DataFrame com X + essa resposta específica
        selected_cols = x_columns + [response_col]
        result[response_col] = df[selected_cols].copy()
    
    return result
