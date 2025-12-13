"""
Control Charts Calculation Utilities
Funções para cálculo de cartas de controle
"""
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd


def get_control_constants() -> Dict[int, Dict[str, float]]:
    """Retorna constantes para cartas de controle"""
    return {
        2: {'A2': 1.880, 'd2': 1.128, 'D3': 0, 'D4': 3.267, 'A3': 2.659, 'B3': 0, 'B4': 3.267},
        3: {'A2': 1.023, 'd2': 1.693, 'D3': 0, 'D4': 2.574, 'A3': 1.954, 'B3': 0, 'B4': 2.568},
        4: {'A2': 0.729, 'd2': 2.059, 'D3': 0, 'D4': 2.282, 'A3': 1.628, 'B3': 0, 'B4': 2.266},
        5: {'A2': 0.577, 'd2': 2.326, 'D3': 0, 'D4': 2.114, 'A3': 1.427, 'B3': 0, 'B4': 2.089},
        6: {'A2': 0.483, 'd2': 2.534, 'D3': 0, 'D4': 2.004, 'A3': 1.287, 'B3': 0.030, 'B4': 1.970},
        7: {'A2': 0.419, 'd2': 2.704, 'D3': 0.076, 'D4': 1.924, 'A3': 1.182, 'B3': 0.118, 'B4': 1.882},
        8: {'A2': 0.373, 'd2': 2.847, 'D3': 0.136, 'D4': 1.864, 'A3': 1.099, 'B3': 0.185, 'B4': 1.815},
        9: {'A2': 0.337, 'd2': 2.970, 'D3': 0.184, 'D4': 1.816, 'A3': 1.032, 'B3': 0.239, 'B4': 1.761},
        10: {'A2': 0.308, 'd2': 3.078, 'D3': 0.223, 'D4': 1.777, 'A3': 0.975, 'B3': 0.284, 'B4': 1.716},
        11: {'A2': 0.285, 'd2': 3.173, 'D3': 0.256, 'D4': 1.744, 'A3': 0.927, 'B3': 0.321, 'B4': 1.679},
        12: {'A2': 0.266, 'd2': 3.258, 'D3': 0.283, 'D4': 1.717, 'A3': 0.886, 'B3': 0.354, 'B4': 1.646},
        13: {'A2': 0.249, 'd2': 3.336, 'D3': 0.307, 'D4': 1.693, 'A3': 0.850, 'B3': 0.382, 'B4': 1.618},
        14: {'A2': 0.235, 'd2': 3.407, 'D3': 0.328, 'D4': 1.672, 'A3': 0.817, 'B3': 0.406, 'B4': 1.594},
        15: {'A2': 0.223, 'd2': 3.472, 'D3': 0.347, 'D4': 1.653, 'A3': 0.789, 'B3': 0.428, 'B4': 1.572},
        20: {'A2': 0.180, 'd2': 3.735, 'D3': 0.415, 'D4': 1.585, 'A3': 0.680, 'B3': 0.510, 'B4': 1.490},
        25: {'A2': 0.153, 'd2': 3.931, 'D3': 0.459, 'D4': 1.541, 'A3': 0.606, 'B3': 0.565, 'B4': 1.435},
    }


def calculate_individual_mr(data: np.ndarray) -> Dict[str, Any]:
    """
    Calcula os limites de controle para carta Individual e Moving Range (I-MR).
    
    Args:
        data: Array numpy com os dados
        
    Returns:
        Dicionário com dados das cartas I e MR
    """
    # Calcula Moving Ranges
    MR = [np.nan]
    for i in range(1, len(data)):
        MR.append(abs(data[i] - data[i-1]))
    
    MR = np.array(MR)
    MR_valid = MR[1:]  # Remove o primeiro NaN
    
    # Constantes para n=2
    constants = get_control_constants()[2]
    d2 = constants['d2']  # 1.128
    D3 = constants['D3']  # 0
    D4 = constants['D4']  # 3.267
    
    # Cálculos para carta Individual
    mean_individual = np.mean(data)
    mean_MR = np.mean(MR_valid)
    
    # Limites sempre com 3 sigma
    ucl_individual = mean_individual + 3 * mean_MR / d2
    lcl_individual = mean_individual - 3 * mean_MR / d2
    
    # Limites para MR
    ucl_mr = D4 * mean_MR
    lcl_mr = D3 * mean_MR
    
    return {
        "individuals": {
            "data": data,
            "mean": mean_individual,
            "ucl": ucl_individual,
            "lcl": lcl_individual,
            "std": np.std(data)
        },
        "moving_range": {
            "data": MR_valid,
            "mean": mean_MR,
            "ucl": ucl_mr,
            "lcl": lcl_mr
        }
    }


def calculate_xbar_r(data: np.ndarray, subgroup_size: int) -> Dict[str, Any]:
    """
    Calcula os limites de controle para carta X-bar & R.
    
    Args:
        data: Array numpy com os dados
        subgroup_size: Tamanho do subgrupo
        
    Returns:
        Dicionário com dados das cartas X-bar e R
    """
    # Divide dados em subgrupos
    num_complete = len(data) // subgroup_size
    data_reshaped = data[:num_complete * subgroup_size].reshape(num_complete, subgroup_size)
    
    # Calcula médias e amplitudes
    xbar = np.mean(data_reshaped, axis=1)
    ranges = np.ptp(data_reshaped, axis=1)
    
    # Médias globais
    xbar_bar = np.mean(xbar)
    r_bar = np.mean(ranges)
    
    # Constantes de controle
    constants = get_control_constants()[subgroup_size]
    A2 = constants['A2']
    D3 = constants['D3']
    D4 = constants['D4']
    
    # Limites de controle
    ucl_xbar = xbar_bar + A2 * r_bar
    lcl_xbar = xbar_bar - A2 * r_bar
    ucl_r = D4 * r_bar
    lcl_r = D3 * r_bar
    
    return {
        "xbar": {
            "data": xbar,
            "mean": xbar_bar,
            "ucl": ucl_xbar,
            "lcl": lcl_xbar,
            "std": np.std(xbar)
        },
        "range": {
            "data": ranges,
            "mean": r_bar,
            "ucl": ucl_r,
            "lcl": lcl_r,
            "std": np.std(ranges)
        },
        "num_subgroups": num_complete
    }


def calculate_xbar_s(data: np.ndarray, subgroup_size: int) -> Dict[str, Any]:
    """
    Calcula os limites de controle para carta X-bar & S.
    
    Args:
        data: Array numpy com os dados
        subgroup_size: Tamanho do subgrupo
        
    Returns:
        Dicionário com dados das cartas X-bar e S
    """
    # Divide dados em subgrupos
    num_complete = len(data) // subgroup_size
    data_reshaped = data[:num_complete * subgroup_size].reshape(num_complete, subgroup_size)
    
    # Calcula médias e desvios padrão
    xbar = np.mean(data_reshaped, axis=1)
    s = np.std(data_reshaped, axis=1, ddof=1)
    
    # Médias globais
    xbar_bar = np.mean(xbar)
    s_bar = np.mean(s)
    
    # Constantes de controle
    constants = get_control_constants()[subgroup_size]
    A3 = constants['A3']
    B3 = constants['B3']
    B4 = constants['B4']
    
    # Limites de controle
    ucl_xbar = xbar_bar + A3 * s_bar
    lcl_xbar = xbar_bar - A3 * s_bar
    ucl_s = B4 * s_bar
    lcl_s = B3 * s_bar
    
    return {
        "xbar": {
            "data": xbar,
            "mean": xbar_bar,
            "ucl": ucl_xbar,
            "lcl": lcl_xbar,
            "std": np.std(xbar)
        },
        "stdev": {
            "data": s,
            "mean": s_bar,
            "ucl": ucl_s,
            "lcl": lcl_s,
            "std": np.std(s)
        },
        "num_subgroups": num_complete
    }


def calculate_p_chart(data: np.ndarray, sample_size: int) -> Dict[str, Any]:
    """
    Calcula os limites de controle para carta P (proporção de defeituosos).
    
    Args:
        data: Array numpy com contagem de defeituosos
        sample_size: Tamanho da amostra
        
    Returns:
        Dicionário com dados da carta P
    """
    # Calcula proporções
    proportions = data / sample_size
    p_bar = np.mean(proportions)
    
    # Limites de controle (3 sigma)
    ucl = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / sample_size)
    lcl = p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / sample_size)
    lcl = max(0, lcl)
    ucl = min(1, ucl)
    
    return {
        "proportions": proportions,
        "mean": p_bar,
        "ucl": ucl,
        "lcl": lcl,
        "sample_size": sample_size
    }


def calculate_np_chart(data: np.ndarray, sample_size: int) -> Dict[str, Any]:
    """
    Calcula os limites de controle para carta NP (número de defeituosos).
    
    Args:
        data: Array numpy com contagem de defeituosos
        sample_size: Tamanho da amostra
        
    Returns:
        Dicionário com dados da carta NP
    """
    np_bar = np.mean(data)
    p_bar = np_bar / sample_size
    
    # Limites de controle (3 sigma)
    ucl = np_bar + 3 * np.sqrt(sample_size * p_bar * (1 - p_bar))
    lcl = np_bar - 3 * np.sqrt(sample_size * p_bar * (1 - p_bar))
    lcl = max(0, lcl)
    
    return {
        "data": data,
        "mean": np_bar,
        "p_bar": p_bar,
        "ucl": ucl,
        "lcl": lcl,
        "sample_size": sample_size
    }


def calculate_c_chart(data: np.ndarray) -> Dict[str, Any]:
    """
    Calcula os limites de controle para carta C (número de defeitos).
    
    Args:
        data: Array numpy com contagem de defeitos
        
    Returns:
        Dicionário com dados da carta C
    """
    c_bar = np.mean(data)
    
    # Limites de controle (3 sigma)
    ucl = c_bar + 3 * np.sqrt(c_bar)
    lcl = c_bar - 3 * np.sqrt(c_bar)
    lcl = max(0, lcl)
    
    return {
        "data": data,
        "mean": c_bar,
        "ucl": ucl,
        "lcl": lcl
    }


def calculate_u_chart(data: np.ndarray, sample_size: int) -> Dict[str, Any]:
    """
    Calcula os limites de controle para carta U (defeitos por unidade).
    
    Args:
        data: Array numpy com contagem de defeitos
        sample_size: Tamanho da amostra
        
    Returns:
        Dicionário com dados da carta U
    """
    u_values = data / sample_size
    u_bar = np.mean(u_values)
    
    # Limites de controle (3 sigma)
    ucl = u_bar + 3 * np.sqrt(u_bar / sample_size)
    lcl = u_bar - 3 * np.sqrt(u_bar / sample_size)
    lcl = max(0, lcl)
    
    return {
        "u_values": u_values,
        "mean": u_bar,
        "ucl": ucl,
        "lcl": lcl,
        "sample_size": sample_size
    }


def count_out_of_control(data: np.ndarray, ucl: float, lcl: float) -> int:
    """
    Conta quantos pontos estão fora dos limites de controle.
    
    Args:
        data: Array com os dados
        ucl: Limite superior de controle
        lcl: Limite inferior de controle
        
    Returns:
        Número de pontos fora de controle
    """
    return np.sum((data > ucl) | (data < lcl))
