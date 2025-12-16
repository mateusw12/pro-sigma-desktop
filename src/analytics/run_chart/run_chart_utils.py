"""
Utilidades para análise de Run Chart
Detecção de padrões, trends, shifts e runs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def analyze_run_chart(data: pd.Series) -> Dict:
    """
    Analisa Run Chart e detecta padrões
    
    Args:
        data: Série temporal de dados
        
    Returns:
        Dict com análise completa
    """
    n = len(data)
    median = data.median()
    
    # Classifica pontos acima/abaixo da mediana
    above_median = (data > median).astype(int)
    below_median = (data < median).astype(int)
    on_median = (data == median).astype(int)
    
    # Contagem de pontos
    n_above = above_median.sum()
    n_below = below_median.sum()
    n_on = on_median.sum()
    
    # Calcula runs (sequências)
    runs = []
    current_run = 1
    for i in range(1, n):
        if on_median[i] == 1:
            continue
        if above_median[i] == above_median[i-1] and on_median[i-1] == 0:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
            current_run = 1
    runs.append(current_run)
    
    n_runs = len(runs)
    longest_run = max(runs) if runs else 0
    
    # Teste de runs (randomness test)
    expected_runs = ((2 * n_above * n_below) / (n_above + n_below)) + 1
    variance_runs = ((2 * n_above * n_below * (2 * n_above * n_below - n_above - n_below)) / 
                     ((n_above + n_below)**2 * (n_above + n_below - 1)))
    std_runs = np.sqrt(variance_runs)
    
    z_score = (n_runs - expected_runs) / std_runs if std_runs > 0 else 0
    
    # Detecção de shift (mudança de nível)
    # Shift: 6+ pontos consecutivos acima ou abaixo da mediana
    shift_detected = longest_run >= 6
    
    # Detecção de trend (tendência)
    # Trend: 5+ pontos consecutivos subindo ou descendo
    trend_up = 0
    trend_down = 0
    max_trend_up = 0
    max_trend_down = 0
    
    for i in range(1, n):
        if data.iloc[i] > data.iloc[i-1]:
            trend_up += 1
            trend_down = 0
            max_trend_up = max(max_trend_up, trend_up)
        elif data.iloc[i] < data.iloc[i-1]:
            trend_down += 1
            trend_up = 0
            max_trend_down = max(max_trend_down, trend_down)
        else:
            trend_up = 0
            trend_down = 0
    
    trend_detected = max_trend_up >= 5 or max_trend_down >= 5
    
    # Detecção de ciclos (oscillation)
    # Muitos runs curtos indicam oscilação
    oscillation_detected = n_runs > expected_runs + 2 * std_runs
    
    # Interpretação
    interpretations = []
    
    if shift_detected:
        interpretations.append({
            'type': 'Shift',
            'description': f'Mudança de nível detectada (run de {longest_run} pontos)',
            'severity': 'high'
        })
    
    if trend_detected:
        if max_trend_up >= 5:
            interpretations.append({
                'type': 'Trend Up',
                'description': f'Tendência de alta detectada ({max_trend_up} pontos consecutivos subindo)',
                'severity': 'medium'
            })
        if max_trend_down >= 5:
            interpretations.append({
                'type': 'Trend Down',
                'description': f'Tendência de baixa detectada ({max_trend_down} pontos consecutivos descendo)',
                'severity': 'medium'
            })
    
    if oscillation_detected:
        interpretations.append({
            'type': 'Oscillation',
            'description': 'Oscilação excessiva detectada (muitas mudanças)',
            'severity': 'low'
        })
    
    if abs(z_score) < 1.96:
        interpretations.append({
            'type': 'Random',
            'description': 'Processo aparenta ser aleatório (sem padrões significativos)',
            'severity': 'none'
        })
    
    # Estatísticas descritivas
    stats = {
        'n': n,
        'median': median,
        'mean': data.mean(),
        'std': data.std(),
        'min': data.min(),
        'max': data.max(),
        'range': data.max() - data.min()
    }
    
    # Informações de runs
    runs_info = {
        'n_runs': n_runs,
        'expected_runs': expected_runs,
        'std_runs': std_runs,
        'z_score': z_score,
        'longest_run': longest_run,
        'n_above': n_above,
        'n_below': n_below,
        'n_on': n_on
    }
    
    # Detecção de padrões
    patterns = {
        'shift_detected': shift_detected,
        'trend_detected': trend_detected,
        'oscillation_detected': oscillation_detected,
        'max_trend_up': max_trend_up,
        'max_trend_down': max_trend_down
    }
    
    return {
        'stats': stats,
        'runs_info': runs_info,
        'patterns': patterns,
        'interpretations': interpretations
    }


def detect_astronomical_points(data: pd.Series, median: float) -> List[int]:
    """
    Detecta pontos astronômicos (muito distantes da mediana)
    
    Args:
        data: Série temporal
        median: Mediana dos dados
        
    Returns:
        Lista de índices dos pontos astronômicos
    """
    # Calcula MAD (Median Absolute Deviation)
    mad = np.median(np.abs(data - median))
    
    # Pontos a mais de 3.5 MAD são considerados astronômicos
    threshold = 3.5 * mad
    astronomical = []
    
    for i, value in enumerate(data):
        if abs(value - median) > threshold:
            astronomical.append(i)
    
    return astronomical
