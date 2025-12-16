"""
Utilidades para análise de Pareto
Cálculo de frequências, percentuais acumulados e regra 80/20
"""

import numpy as np
import pandas as pd
from typing import Dict


def calculate_pareto(data: pd.DataFrame, category_col: str, value_col: str = None) -> Dict:
    """
    Calcula análise de Pareto
    
    Args:
        data: DataFrame com os dados
        category_col: Coluna de categorias
        value_col: Coluna de valores (opcional, se None conta frequências)
        
    Returns:
        Dict com análise completa
    """
    # Se value_col não fornecida, conta frequências
    if value_col is None:
        freq_data = data[category_col].value_counts().reset_index()
        freq_data.columns = ['Category', 'Count']
    else:
        freq_data = data.groupby(category_col)[value_col].sum().reset_index()
        freq_data.columns = ['Category', 'Count']
    
    # Ordena por valor decrescente
    freq_data = freq_data.sort_values('Count', ascending=False).reset_index(drop=True)
    
    # Calcula percentuais
    total = freq_data['Count'].sum()
    freq_data['Percent'] = (freq_data['Count'] / total) * 100
    freq_data['Cumulative_Percent'] = freq_data['Percent'].cumsum()
    
    # Identifica categorias da regra 80/20
    vital_few_mask = freq_data['Cumulative_Percent'] <= 80
    vital_few_categories = freq_data[vital_few_mask]['Category'].tolist()
    n_vital_few = len(vital_few_categories)
    
    # Se nenhuma categoria atingiu 80%, pega até ultrapassar
    if n_vital_few == 0:
        vital_few_mask = freq_data['Cumulative_Percent'] <= freq_data['Cumulative_Percent'].iloc[0]
        vital_few_categories = freq_data[vital_few_mask]['Category'].tolist()
        n_vital_few = max(1, len(vital_few_categories))
    
    # Estatísticas
    n_categories = len(freq_data)
    percent_vital = (n_vital_few / n_categories) * 100 if n_categories > 0 else 0
    
    # Contribuição do vital few
    vital_contribution = freq_data[freq_data['Category'].isin(vital_few_categories)]['Count'].sum()
    vital_contribution_pct = (vital_contribution / total) * 100 if total > 0 else 0
    
    # Classificação ABC
    freq_data['ABC_Class'] = 'C'
    cum_pct = 0
    for i, row in freq_data.iterrows():
        cum_pct += row['Percent']
        if cum_pct <= 80:
            freq_data.at[i, 'ABC_Class'] = 'A'
        elif cum_pct <= 95:
            freq_data.at[i, 'ABC_Class'] = 'B'
    
    # Conta classes
    abc_counts = freq_data['ABC_Class'].value_counts().to_dict()
    
    return {
        'pareto_data': freq_data,
        'total': total,
        'n_categories': n_categories,
        'vital_few_categories': vital_few_categories,
        'n_vital_few': n_vital_few,
        'percent_vital': percent_vital,
        'vital_contribution': vital_contribution,
        'vital_contribution_pct': vital_contribution_pct,
        'abc_counts': abc_counts,
        'category_col': category_col,
        'value_col': value_col
    }


def analyze_pareto_principle(results: Dict) -> Dict:
    """
    Analisa aderência ao princípio de Pareto (80/20)
    
    Args:
        results: Resultado de calculate_pareto
        
    Returns:
        Dict com análise da regra 80/20
    """
    n_vital = results['n_vital_few']
    n_total = results['n_categories']
    percent_vital = results['percent_vital']
    contribution_pct = results['vital_contribution_pct']
    
    # Verifica aderência à regra 80/20
    # Ideal: ~20% das categorias representam ~80% dos valores
    ideal_vital_pct = 20
    ideal_contribution_pct = 80
    
    vital_deviation = abs(percent_vital - ideal_vital_pct)
    contribution_deviation = abs(contribution_pct - ideal_contribution_pct)
    
    # Classificação da aderência
    if vital_deviation <= 10 and contribution_deviation <= 10:
        adherence = "Forte"
        adherence_color = "#4ade80"
        description = "Segue fortemente o princípio de Pareto (80/20)"
    elif vital_deviation <= 20 and contribution_deviation <= 20:
        adherence = "Moderada"
        adherence_color = "#fbbf24"
        description = "Aderência moderada ao princípio de Pareto"
    else:
        adherence = "Fraca"
        adherence_color = "#f87171"
        description = "Distribuição não segue o princípio de Pareto"
    
    return {
        'adherence': adherence,
        'adherence_color': adherence_color,
        'description': description,
        'vital_deviation': vital_deviation,
        'contribution_deviation': contribution_deviation,
        'ideal_vital_pct': ideal_vital_pct,
        'ideal_contribution_pct': ideal_contribution_pct
    }
