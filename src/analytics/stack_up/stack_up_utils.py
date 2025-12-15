"""
Stack-Up Analysis Utilities
Fornece funções para cálculo de empilhamento de tolerâncias
"""

import re
from typing import Any, Dict, List, Tuple
from src.utils.lazy_imports import get_numpy, get_pandas


def generate_data_frame(distributions: Dict[str, List[float]]):
    """
    Gera um DataFrame a partir de um dicionário contendo distribuições de valores.

    Args:
        distributions: Dicionário onde as chaves representam os nomes das colunas 
                      e os valores são listas de números.

    Returns:
        DataFrame contendo as distribuições fornecidas, com as chaves como nomes das colunas.
    """
    pd = get_pandas()
    column_names = {key: key for key, _ in distributions.items()}
    data = {column_names[key]: values for key, values in distributions.items()}
    df = pd.DataFrame(data)
    return df


def generate_distributions(
    rounds: int, 
    means: Dict[str, float], 
    stds: Dict[str, float]
) -> Dict[str, List[float]]:
    """
    Gera distribuições normais baseadas em médias e desvios padrão.

    Args:
        rounds: Número de amostras a serem geradas para cada distribuição.
        means: Dicionário contendo as médias de cada fator.
        stds: Dicionário contendo os desvios padrão de cada fator.

    Returns:
        Dicionário com distribuições normais geradas para cada fator.
    """
    np = get_numpy()
    distributions: Dict[str, List[float]] = {}
    
    for key, (mean, std) in zip(means.keys(), zip(means.values(), stds.values())):
        distributions[key] = np.random.normal(loc=mean, scale=std, size=rounds).tolist()
    
    return distributions


def normalize_column_name(column_name: str) -> str:
    """
    Normaliza o nome de uma coluna, substituindo caracteres especiais e espaços por underscores.

    Args:
        column_name: Nome original da coluna.

    Returns:
        Nome normalizado da coluna.
    """
    return re.sub(r'\W|^(?=\d)', '_', column_name)


def calculate_equation(factors: Dict[str, Any]) -> str:
    """
    Calcula a equação com base nos fatores fornecidos, usando nomes normalizados.

    Args:
        factors: Dicionário onde cada valor contém:
                - name (str): Nome do fator.
                - sensitivity (float): Sensibilidade do fator.

    Returns:
        String representando a equação calculada.
    """
    equation_parts = []
    for _, value in factors.items():
        column_name = normalize_column_name(value['name'])
        equation_parts.append(f"{column_name} * ({value['sensitivity']})")
    
    return " + ".join(equation_parts)


def calculate_equation_not_normalized(factors: Dict[str, Any]) -> str:
    """
    Calcula a equação com base nos fatores fornecidos, sem normalizar os nomes das colunas.

    Args:
        factors: Dicionário onde cada valor contém:
                - name (str): Nome do fator.
                - sensitivity (float): Sensibilidade do fator.

    Returns:
        String representando a equação calculada com nomes originais.
    """
    equation_parts = []
    for _, value in factors.items():
        equation_parts.append(f"{value['name']} * ({value['sensitivity']})")
    
    return " + ".join(equation_parts)


def calculate_means_std(factors: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calcula as médias e os desvios padrão para os fatores fornecidos.

    Args:
        factors: Dicionário onde cada valor contém:
                - name (str): Nome do fator.
                - min (float): Valor mínimo do fator.
                - max (float): Valor máximo do fator.
                - quota (float): Quota para cálculo do desvio padrão.

    Returns:
        Tupla contendo dois dicionários: (médias, desvios padrão).
    """
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}

    for _, value in factors.items():
        means[value['name']] = (value['max'] + value['min']) / 2
        stds[value['name']] = (value['max'] - value['min']) / (6 * float(value['quota']))

    return means, stds


def trim_names(factors: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove espaços dos nomes dos fatores.

    Args:
        factors: Dicionário onde cada valor contém:
                - name (str): Nome do fator.

    Returns:
        Dicionário com nomes dos fatores sem espaços.
    """
    for _, factor in factors.items():
        factor['name'] = factor['name'].replace(" ", "")
    
    return factors


def validate_factors(factors: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Valida os fatores fornecidos.

    Args:
        factors: Dicionário contendo os fatores a serem validados.

    Returns:
        Tupla (válido, mensagem de erro).
    """
    if not factors:
        return False, "Nenhum fator fornecido"
    
    for key, factor in factors.items():
        if factor['min'] > factor['max']:
            return False, f"Valor mínimo maior que máximo no fator {factor['name']}"
        
        if factor['sensitivity'] == 0:
            return False, f"Sensibilidade não pode ser zero no fator {factor['name']}"
    
    return True, ""


def calculate_stack_up(rounds: int, factors: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calcula o empilhamento de tolerâncias completo.

    Args:
        rounds: Número de rodadas para simulação.
        factors: Dicionário contendo os fatores com suas características.

    Returns:
        Dicionário com os resultados do cálculo.
    """
    # Valida fatores
    is_valid, error_msg = validate_factors(factors)
    if not is_valid:
        raise ValueError(error_msg)
    
    # Remove espaços em branco
    new_factors = trim_names(factors)

    # Calcula média e desvio padrão
    means, stds = calculate_means_std(new_factors)

    # Gera equação
    equation = calculate_equation(new_factors)

    # Calcula as distribuições
    distributions = generate_distributions(rounds, means, stds)

    # Gera data frame
    df = generate_data_frame(distributions)

    # Normaliza nomes das colunas
    df.columns = [normalize_column_name(col) for col in df.columns]

    # Calcula coluna resposta
    df["Y"] = df.eval(equation)

    # Gera equação não normalizada para exibição
    new_equation = calculate_equation_not_normalized(factors)

    return {
        "means": means,
        "stds": stds,
        "equation": new_equation,
        "dataframe": df,
        "distributions": distributions
    }
