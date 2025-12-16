"""
Utilidades para Gage R&R (Repeatability and Reproducibility)
Análise de Sistema de Medição (MSA)
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from collections import OrderedDict


# ========== LAZY IMPORTS ==========
def get_pandas():
    """Lazy import do pandas"""
    import pandas as pd
    return pd


def get_numpy():
    """Lazy import do numpy"""
    import numpy as np
    return np


def get_scipy_stats():
    """Lazy import do scipy.stats"""
    from scipy import stats
    return stats


# ========== FUNÇÕES DE CÁLCULO ==========

def calculate_gage_rr(
    data: pd.DataFrame,
    part_col: str,
    operator_col: str,
    measurement_col: str,
    tolerance: float = None,
    spec_range: float = None
) -> Dict:
    """
    Calcula Gage R&R usando método ANOVA
    
    Args:
        data: DataFrame com os dados
        part_col: Nome da coluna de peças
        operator_col: Nome da coluna de operadores
        measurement_col: Nome da coluna de medições
        tolerance: Tolerância total (USL - LSL)
        spec_range: Range de especificação (6σ do processo)
        
    Returns:
        Dict com todos os resultados da análise
    """
    np = get_numpy()
    pd = get_pandas()
    
    # Prepara dados
    parts = data[part_col].unique()
    operators = data[operator_col].unique()
    n_parts = len(parts)
    n_operators = len(operators)
    n_trials = len(data) // (n_parts * n_operators)
    
    # Cria tabela pivotada para cálculos
    pivot_data = []
    for part in parts:
        for operator in operators:
            measurements = data[(data[part_col] == part) & (data[operator_col] == operator)][measurement_col].values
            pivot_data.append({
                'Part': part,
                'Operator': operator,
                'Measurements': measurements.tolist(),
                'Average': np.mean(measurements),
                'Range': np.max(measurements) - np.min(measurements)
            })
    
    pivot_df = pd.DataFrame(pivot_data)
    
    # ========== ANOVA Calculations ==========
    
    # Grand Average
    grand_avg = data[measurement_col].mean()
    
    # Sum of Squares
    SS_total = np.sum((data[measurement_col] - grand_avg) ** 2)
    
    # SS Parts
    part_means = data.groupby(part_col)[measurement_col].mean()
    n_measurements_per_part = n_operators * n_trials
    SS_parts = n_measurements_per_part * np.sum((part_means - grand_avg) ** 2)
    
    # SS Operators
    operator_means = data.groupby(operator_col)[measurement_col].mean()
    n_measurements_per_operator = n_parts * n_trials
    SS_operators = n_measurements_per_operator * np.sum((operator_means - grand_avg) ** 2)
    
    # SS Interaction (Part * Operator)
    part_operator_means = data.groupby([part_col, operator_col])[measurement_col].mean().reset_index()
    SS_interaction = 0
    for _, row in part_operator_means.iterrows():
        part_mean = part_means[row[part_col]]
        operator_mean = operator_means[row[operator_col]]
        po_mean = row[measurement_col]
        SS_interaction += n_trials * (po_mean - part_mean - operator_mean + grand_avg) ** 2
    
    # SS Equipment (Repeatability)
    SS_equipment = SS_total - SS_parts - SS_operators - SS_interaction
    
    # Degrees of Freedom
    df_parts = n_parts - 1
    df_operators = n_operators - 1
    df_interaction = df_parts * df_operators
    df_equipment = n_parts * n_operators * (n_trials - 1)
    df_total = len(data) - 1
    
    # Mean Squares
    MS_parts = SS_parts / df_parts if df_parts > 0 else 0
    MS_operators = SS_operators / df_operators if df_operators > 0 else 0
    MS_interaction = SS_interaction / df_interaction if df_interaction > 0 else 0
    MS_equipment = SS_equipment / df_equipment if df_equipment > 0 else 0
    
    # Variance Components
    # Equipment Variation (EV) - Repeatability
    var_equipment = MS_equipment
    
    # Appraiser Variation (AV) - Reproducibility
    # Primeiro calcula variance do operador
    var_operator = max(0, (MS_operators - MS_interaction) / (n_parts * n_trials))
    
    # Interaction variance
    var_interaction = max(0, (MS_interaction - MS_equipment) / n_trials)
    
    # AV = Operator variance + Interaction variance
    var_appraiser = var_operator + var_interaction
    
    # Part Variation (PV)
    var_parts = max(0, (MS_parts - MS_interaction) / (n_operators * n_trials))
    
    # Gage R&R = EV + AV
    var_gage_rr = var_equipment + var_appraiser
    
    # Total Variation
    var_total = var_gage_rr + var_parts
    
    # Study Variation (SD)
    sd_equipment = np.sqrt(var_equipment)
    sd_appraiser = np.sqrt(var_appraiser)
    sd_gage_rr = np.sqrt(var_gage_rr)
    sd_parts = np.sqrt(var_parts)
    sd_total = np.sqrt(var_total)
    
    # Study Variation (6σ)
    sv_equipment = 6 * sd_equipment
    sv_appraiser = 6 * sd_appraiser
    sv_gage_rr = 6 * sd_gage_rr
    sv_parts = 6 * sd_parts
    sv_total = 6 * sd_total
    
    # Percentages (% Study Variation)
    pct_sv_equipment = (sv_equipment / sv_total * 100) if sv_total > 0 else 0
    pct_sv_appraiser = (sv_appraiser / sv_total * 100) if sv_total > 0 else 0
    pct_sv_gage_rr = (sv_gage_rr / sv_total * 100) if sv_total > 0 else 0
    pct_sv_parts = (sv_parts / sv_total * 100) if sv_total > 0 else 0
    
    # % Tolerance (if provided)
    pct_tol_equipment = (sv_equipment / tolerance * 100) if tolerance else None
    pct_tol_appraiser = (sv_appraiser / tolerance * 100) if tolerance else None
    pct_tol_gage_rr = (sv_gage_rr / tolerance * 100) if tolerance else None
    
    # Number of Distinct Categories (ndc)
    ndc = int(np.sqrt(2 * (var_parts / var_gage_rr))) if var_gage_rr > 0 else 0
    
    # Interpretação do Gage R&R
    if pct_sv_gage_rr < 10:
        grr_interpretation = "Excelente - Sistema de medição aceitável"
        grr_status = "accept"
    elif pct_sv_gage_rr < 30:
        grr_interpretation = "Marginal - Sistema pode ser aceitável dependendo da aplicação"
        grr_status = "marginal"
    else:
        grr_interpretation = "Inaceitável - Sistema de medição precisa melhorar"
        grr_status = "reject"
    
    # Interpretação do NDC
    if ndc >= 5:
        ndc_interpretation = "Excelente - Sistema pode discriminar bem"
    elif ndc >= 2:
        ndc_interpretation = "Marginal - Sistema tem discriminação limitada"
    else:
        ndc_interpretation = "Inaceitável - Sistema não discrimina adequadamente"
    
    # ANOVA Table
    anova_table = {
        'Source': ['Parts', 'Operators', 'Part*Operator', 'Repeatability', 'Total'],
        'DF': [df_parts, df_operators, df_interaction, df_equipment, df_total],
        'SS': [SS_parts, SS_operators, SS_interaction, SS_equipment, SS_total],
        'MS': [MS_parts, MS_operators, MS_interaction, MS_equipment, 0],
        'Var': [var_parts, var_operator, var_interaction, var_equipment, var_total]
    }
    
    # Variance Components Summary
    variance_components = {
        'Source': ['Gage R&R', '  Repeatability', '  Reproducibility', 'Part-to-Part', 'Total Variation'],
        'VarComp': [var_gage_rr, var_equipment, var_appraiser, var_parts, var_total],
        'StdDev': [sd_gage_rr, sd_equipment, sd_appraiser, sd_parts, sd_total],
        '6*SD': [sv_gage_rr, sv_equipment, sv_appraiser, sv_parts, sv_total],
        '%StudyVar': [pct_sv_gage_rr, pct_sv_equipment, pct_sv_appraiser, pct_sv_parts, 100.0]
    }
    
    if tolerance:
        variance_components['%Tolerance'] = [
            pct_tol_gage_rr, pct_tol_equipment, pct_tol_appraiser, None, None
        ]
    
    # Operator Statistics
    operator_stats = []
    for operator in operators:
        op_data = data[data[operator_col] == operator][measurement_col]
        operator_stats.append({
            'Operator': operator,
            'Average': np.mean(op_data),
            'StdDev': np.std(op_data, ddof=1),
            'Count': len(op_data)
        })
    
    # Part Statistics
    part_stats = []
    for part in parts:
        part_data = data[data[part_col] == part][measurement_col]
        part_stats.append({
            'Part': part,
            'Average': np.mean(part_data),
            'Range': np.max(part_data) - np.min(part_data),
            'StdDev': np.std(part_data, ddof=1)
        })
    
    # Range Control Chart Limits
    avg_range = pivot_df['Range'].mean()
    # D4 and D3 constants for control charts (n=trials)
    d2_values = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326}
    D4_values = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114}
    D3_values = {2: 0, 3: 0, 4: 0, 5: 0}
    
    d2 = d2_values.get(n_trials, 2.326)
    D4 = D4_values.get(n_trials, 2.114)
    D3 = D3_values.get(n_trials, 0)
    
    UCL_range = D4 * avg_range
    LCL_range = D3 * avg_range
    
    # X-bar Control Chart Limits
    A2_values = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577}
    A2 = A2_values.get(n_trials, 0.577)
    
    UCL_xbar = grand_avg + A2 * avg_range
    LCL_xbar = grand_avg - A2 * avg_range
    
    return {
        'anova_table': anova_table,
        'variance_components': variance_components,
        'operator_stats': operator_stats,
        'part_stats': part_stats,
        'pivot_data': pivot_df.to_dict('records'),
        'summary': {
            'n_parts': n_parts,
            'n_operators': n_operators,
            'n_trials': n_trials,
            'n_measurements': len(data),
            'grand_average': grand_avg,
            'grr_percent': pct_sv_gage_rr,
            'grr_interpretation': grr_interpretation,
            'grr_status': grr_status,
            'ndc': ndc,
            'ndc_interpretation': ndc_interpretation,
            'avg_range': avg_range,
            'UCL_range': UCL_range,
            'LCL_range': LCL_range,
            'UCL_xbar': UCL_xbar,
            'LCL_xbar': LCL_xbar
        },
        'raw_data': data.to_dict('records')
    }


def prepare_gage_rr_data(
    data: pd.DataFrame,
    part_col: str,
    operator_col: str,
    measurement_cols: List[str]
) -> pd.DataFrame:
    """
    Prepara dados para análise Gage R&R
    Transforma formato wide (múltiplas colunas de medição) para long format
    
    Args:
        data: DataFrame original
        part_col: Coluna de peças
        operator_col: Coluna de operadores
        measurement_cols: Lista de colunas de medições (trial1, trial2, etc)
        
    Returns:
        DataFrame no formato long com colunas: Part, Operator, Measurement
    """
    pd = get_pandas()
    
    records = []
    for _, row in data.iterrows():
        part = row[part_col]
        operator = row[operator_col]
        for measurement in measurement_cols:
            records.append({
                'Part': part,
                'Operator': operator,
                'Measurement': row[measurement]
            })
    
    return pd.DataFrame(records)
