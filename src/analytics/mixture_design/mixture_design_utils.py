"""
Mixture Design Utilities
Geração de designs de mistura e cálculos de modelos
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy import stats


def generate_mixture_design(
    n_factors: int,
    n_runs: int,
    design_type: str = "space_filling",
    constraints: Optional[pd.DataFrame] = None,
    n_replicates: int = 1
) -> pd.DataFrame:
    """
    Gera design de mistura
    
    Args:
        n_factors: Número de fatores
        n_runs: Número de experimentos
        design_type: 'space_filling' ou 'optimal'
        constraints: DataFrame com minLevelValue e maxLevelValue
        n_replicates: Número de réplicas
    
    Returns:
        DataFrame com o design gerado
    """
    if design_type == "space_filling":
        design = _generate_space_filling(n_factors, n_runs)
    elif design_type == "optimal":
        design = _generate_optimal(n_factors, n_runs)
    else:
        raise ValueError(f"Tipo de design desconhecido: {design_type}")
    
    # Aplicar restrições se fornecidas
    if constraints is not None and len(constraints) > 0:
        design = _apply_constraints(design, constraints)
    
    # Normalizar para garantir soma = 1
    design = _normalize_mixture(design)
    
    # Criar DataFrame
    columns = [f"X{i+1}" for i in range(n_factors)]
    df = pd.DataFrame(design, columns=columns)
    
    # Adicionar réplicas
    if n_replicates > 1:
        df = pd.concat([df] * n_replicates, ignore_index=True)
    
    return df


def _generate_space_filling(n_factors: int, n_runs: int) -> np.ndarray:
    """Gera design usando Latin Hypercube Sampling"""
    # Latin Hypercube Sampling simplificado
    design = np.zeros((n_runs, n_factors))
    
    for i in range(n_factors):
        # Gerar valores uniformes e embaralhar
        intervals = np.arange(n_runs) + np.random.uniform(0, 1, n_runs)
        intervals = intervals / n_runs
        np.random.shuffle(intervals)
        design[:, i] = intervals
    
    return design


def _generate_optimal(n_factors: int, n_runs: int) -> np.ndarray:
    """Gera design D-optimal simplificado"""
    # Gerar candidatos
    n_candidates = n_runs * 10
    candidates = np.random.dirichlet(np.ones(n_factors), n_candidates)
    
    # Selecionar n_runs pontos que maximizam determinante de X'X
    selected_indices = []
    
    # Primeiro ponto: vértice
    selected_indices.append(0)
    
    # Selecionar pontos adicionais
    for _ in range(n_runs - 1):
        best_det = -np.inf
        best_idx = None
        
        for idx in range(n_candidates):
            if idx in selected_indices:
                continue
            
            # Testar adicionar este ponto
            test_design = candidates[selected_indices + [idx]]
            
            # Calcular determinante
            try:
                det = np.linalg.det(test_design.T @ test_design)
                if det > best_det:
                    best_det = det
                    best_idx = idx
            except:
                continue
        
        if best_idx is not None:
            selected_indices.append(best_idx)
    
    return candidates[selected_indices]


def _apply_constraints(design: np.ndarray, constraints: pd.DataFrame) -> np.ndarray:
    """Aplica restrições de min/max aos fatores"""
    n_factors = design.shape[1]
    
    for i in range(min(n_factors, len(constraints))):
        min_val = constraints.iloc[i]['minLevelValue']
        max_val = constraints.iloc[i]['maxLevelValue']
        range_val = max_val - min_val
        
        design[:, i] = min_val + design[:, i] * range_val
    
    return design


def _normalize_mixture(design: np.ndarray) -> np.ndarray:
    """Normaliza para que soma de cada linha = 1"""
    row_sums = design.sum(axis=1, keepdims=True)
    return design / row_sums


def calculate_mixture_model(
    df: pd.DataFrame,
    response_col: str,
    x_cols: List[str],
    include_interactions: bool = False,
    custom_terms: Optional[List[str]] = None
) -> Dict:
    """
    Calcula modelo de mistura (Scheffé polynomial)
    
    Args:
        df: DataFrame com dados
        response_col: Nome da coluna resposta
        x_cols: Colunas dos fatores
        include_interactions: Incluir interações de 2ª ordem
        custom_terms: Termos customizados (ex: ['X1*X2'])
    
    Returns:
        Dict com resultados
    """
    import statsmodels.api as sm
    from sklearn.metrics import r2_score, mean_squared_error
    
    # Preparar dados
    df_work = df.copy()
    X_data = df_work[x_cols].values
    y = df_work[response_col].values
    
    # Criar termos de interação se solicitado
    if custom_terms:
        for term in custom_terms:
            if '*' in term:
                vars_in_term = term.split('*')
                if len(vars_in_term) == 2:
                    col1, col2 = vars_in_term[0].strip(), vars_in_term[1].strip()
                    if col1 in df_work.columns and col2 in df_work.columns:
                        interaction_name = f"{col1}_{col2}"
                        df_work[interaction_name] = df_work[col1] * df_work[col2]
                        x_cols = list(x_cols) + [interaction_name]
    
    # Para mixture design, não incluir intercepto (restrição de soma=1)
    X = df_work[x_cols].values
    
    # Adicionar interações automáticas se solicitado
    if include_interactions and not custom_terms:
        n_factors = len(x_cols)
        for i in range(n_factors):
            for j in range(i+1, n_factors):
                interaction_name = f"{x_cols[i]}_{x_cols[j]}"
                df_work[interaction_name] = X[:, i] * X[:, j]
        
        X = df_work[[col for col in df_work.columns if col.startswith('X')]].values
        x_cols = [col for col in df_work.columns if col.startswith('X')]
    
    # Ajustar modelo sem intercepto (modelo de Scheffé)
    model = sm.OLS(y, X).fit()
    
    # Predições
    y_pred = model.predict(X)
    
    # Métricas
    n = len(y)
    p = X.shape[1]
    
    r2 = r2_score(y, y_pred)
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mean_y = np.mean(y)
    
    # Estimativas dos parâmetros
    param_estimates = []
    for i, col in enumerate(x_cols):
        param_estimates.append({
            'term': col.replace('_', '*'),
            'estimate': model.params[i],
            'stdError': model.bse[i],
            'tValue': model.tvalues[i],
            'prob': model.pvalues[i]
        })
    
    # ANOVA
    ss_total = np.sum((y - mean_y) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)
    ss_model = ss_total - ss_residual
    
    df_model = p
    df_error = n - p
    df_total = n - 1
    
    ms_model = ss_model / df_model if df_model > 0 else 0
    ms_error = ss_residual / df_error if df_error > 0 else 0
    
    f_value = ms_model / ms_error if ms_error > 0 else 0
    prob_f = 1 - stats.f.cdf(f_value, df_model, df_error) if ms_error > 0 else 1.0
    
    anova_table = [
        {
            'source': 'Modelo',
            'df': df_model,
            'sumSquares': ss_model,
            'meanSquares': ms_model,
            'fValue': f_value,
            'prob': prob_f
        },
        {
            'source': 'Erro',
            'df': df_error,
            'sumSquares': ss_residual,
            'meanSquares': ms_error,
            'fValue': None,
            'prob': None
        },
        {
            'source': 'Total',
            'df': df_total,
            'sumSquares': ss_total,
            'meanSquares': None,
            'fValue': None,
            'prob': None
        }
    ]
    
    # Summary of fit
    summary_of_fit = {
        'r2': r2,
        'r2_adj': r2_adj,
        'rmse': rmse,
        'mean': mean_y,
        'observations': n
    }
    
    # Equação
    equation = _generate_equation(param_estimates, response_col)
    
    # Dados para gráficos de predição
    prediction_data = {}
    for col in x_cols:
        if col in df_work.columns:
            unique_vals = np.sort(df_work[col].unique())
            pred_vals = []
            for val in unique_vals:
                mask = df_work[col] == val
                pred_vals.append(np.mean(y_pred[mask]))
            
            prediction_data[col] = {
                'x': unique_vals.tolist(),
                'y': pred_vals
            }
    
    return {
        'parameterEstimates': param_estimates,
        'anovaTable': anova_table,
        'summaryOfFit': summary_of_fit,
        'equation': equation,
        'predictions': y_pred.tolist(),
        'residuals': (y - y_pred).tolist(),
        'predictionData': prediction_data,
        'model': model,
        'yActual': y.tolist()
    }


def _generate_equation(param_estimates: List[Dict], response_col: str) -> str:
    """Gera equação do modelo"""
    equation = f"{response_col} = "
    
    terms = []
    for param in param_estimates:
        coef = param['estimate']
        term = param['term']
        
        if coef >= 0 and len(terms) > 0:
            terms.append(f"+ {coef:.6f}*{term}")
        else:
            terms.append(f"{coef:.6f}*{term}")
    
    equation += " ".join(terms)
    return equation


def interpret_mixture_results(results: Dict) -> List[str]:
    """Gera interpretações dos resultados"""
    interpretations = []
    
    summary = results['summaryOfFit']
    anova = results['anovaTable'][0]  # Linha do modelo
    
    # Interpretação do modelo
    if anova['prob'] < 0.001:
        interpretations.append("O modelo é altamente significativo (p < 0.001)")
    elif anova['prob'] < 0.05:
        interpretations.append(f"O modelo é significativo (p = {anova['prob']:.4f})")
    else:
        interpretations.append(f"O modelo não é significativo (p = {anova['prob']:.4f})")
    
    # Interpretação do R²
    r2 = summary['r2']
    if r2 > 0.9:
        interpretations.append(f"Excelente ajuste do modelo (R² = {r2:.4f})")
    elif r2 > 0.7:
        interpretations.append(f"Bom ajuste do modelo (R² = {r2:.4f})")
    elif r2 > 0.5:
        interpretations.append(f"Ajuste moderado do modelo (R² = {r2:.4f})")
    else:
        interpretations.append(f"Ajuste fraco do modelo (R² = {r2:.4f})")
    
    # Fatores significativos
    significant = [p['term'] for p in results['parameterEstimates'] if p['prob'] < 0.05]
    if significant:
        interpretations.append(f"Fatores significativos (p < 0.05): {', '.join(significant)}")
    else:
        interpretations.append("Nenhum fator individualmente significativo")
    
    return interpretations
