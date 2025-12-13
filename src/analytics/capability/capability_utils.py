"""
Utility functions for Process Capability calculations
Adapted from FastAPI backend for desktop application
"""
from typing import List, Tuple
from src.utils.lazy_imports import get_numpy, get_pandas, get_scipy_stats

# Lazy imports - carregados apenas quando usados
np = None
pd = None
norm = None
weibull_min = None
chi2 = None

def _ensure_imports():
    """Garante que imports pesados estão carregados"""
    global np, pd, norm, weibull_min, chi2
    if np is None:
        np = get_numpy()
    if pd is None:
        pd = get_pandas()
    if norm is None:
        stats = get_scipy_stats()
        norm = stats.norm
        weibull_min = stats.weibull_min
        chi2 = stats.chi2
    return np, pd, norm, weibull_min, chi2


def data_frame_split_by_columns(df):
    """Separa um DataFrame em múltiplos DataFrames com base nos valores únicos da primeira coluna."""
    first_column_name = df.columns[0]
    unique_values = df[first_column_name].unique()
    split_data_frames = []
    for value in unique_values:
        split_df = df[df[first_column_name] == value]
        split_data_frames.append(split_df)
    return split_data_frames


def remove_last_column(df, last_column: str):
    """Remove a última coluna de um DataFrame com base no nome da coluna fornecido."""
    df.drop(columns=[last_column], inplace=True)


def calculate_pp_ppk(lse: float, lie: float, split_df):
    """Calcula o Índice de Capacidade do Processo (PP) e o Índice de Capacidade do Processo Ajustado (PPK)."""
    _ensure_imports()
    mean = split_df.mean().iloc[0]
    col = split_df.iloc[:, 0]
    std = np.std(col, ddof=1)

    pp = (lse - lie) / (6 * std)
    ppu, ppl = calculate_pp(lse, lie, mean, std)
    ppk = np.min([ppu, ppl])

    return pp, ppk, ppu, ppl


def calculate_pp(lse: float, lie: float, mean: float, std: float):
    """Calcula os índices de capacidade superior (PPU) e capacidade inferior (PPL) do processo."""
    ppu = (lse - mean) / (3 * std)
    ppl = (mean - lie) / (3 * std)
    return ppu, ppl


def calculate_cp_cpk(lse: float, lie: float, split_df, within_sigma: float):
    """Calcula os índices de capacidade do processo (CP) e capacidade do processo ajustado (CPK)."""
    _ensure_imports()
    mean = split_df.mean().iloc[0]
    cp = (lse - lie) / (6 * within_sigma)
    cpl, cpu = calculate_cpk(lse, lie, mean, within_sigma)
    cpk = np.min([cpl, cpu])
    return cp, cpk, cpl, cpu


def average(arr: List[float]):
    """Calcula a média de uma lista de valores numéricos."""
    total_sum = sum(arr)
    avg = total_sum / len(arr) if len(arr) > 0 else 0
    return avg


def calculate_cpk(lse: float, lie: float, mean: float, ranges: float):
    """Calcula os índices de capacidade do processo ajustado (CPL e CPU)."""
    cpl = (lse - mean) / (3 * ranges)
    cpu = (mean - lie) / (3 * ranges)
    return cpl, cpu


def calculate_process_summary(split_df):
    """Calcula o resumo do processo, incluindo a média, sigma dentro do processo, sigma geral e estabilidade."""
    _ensure_imports()
    col_name = split_df.columns[0]
    col = split_df[col_name]
    std = np.std(col, ddof=1)

    moving_ranges = calculate_moving_ranges(split_df)
    moving_mean = average(moving_ranges)

    mean = split_df.mean().iloc[0]
    within_sigma = moving_mean / 1.128
    overall_sigma = std
    stability = overall_sigma / within_sigma

    return mean, within_sigma, overall_sigma, stability


def calculate_moving_ranges(df):
    """Calcula as variações móveis (diferenças absolutas entre valores consecutivos) para a primeira coluna do DataFrame."""
    _ensure_imports()
    moving_range = []
    col_name = df.columns[0]
    col = df[col_name]

    if len(col) < 2:
        return moving_range

    for i in range(len(col) - 1):
        moving_ranges = abs(col.iloc[i] - col.iloc[i + 1])
        moving_range.append(moving_ranges)

    return moving_range


def calculate_inferior_limit(value: float, observations: int):
    """Calcula o limite inferior usando a fórmula de intervalo de confiança para uma distribuição normal."""
    return value - 1.96 * (1 / (observations * 3**2) + (value**2) / (2 * observations - 1))**0.5


def calculate_superior_limit(value: float, observations: int):
    """Calcula o limite superior usando a fórmula de intervalo de confiança para uma distribuição normal."""
    return value + 1.96 * (1 / (observations * 3**2) + (value**2) / (2 * observations - 1))**0.5


def calculate_se(observations: int, within_sigma: float, lie: float, lse: float, mean: float):
    """Calcula o erro padrão."""
    result = (((mean - lie)**2 + (lse - mean)**2) / (9 * observations * (within_sigma**2)) + (1) / (2 * observations))**0.5
    return result


def calculate_ppk_not_normal(lse: float, lie: float, toleranceType: str, P50: float, P99865: float, P000135: float, overall_sigma: float):
    """Calcula o índice de capacidade de processo (Ppk) e o valor de capacidade de processo (Pp) para dados não normais."""
    ppu = (lse - P50) / (P99865 - P000135)
    ppl = (P50 - lie) / (P99865 - P000135)
    pp = (lse - lie) / (6 * overall_sigma)

    if toleranceType == "bilateral":
        ppPpk_value = np.min([ppu, ppl])
    elif toleranceType == "inferiorUnilateral":
        ppPpk_value = ppl
    else:
        ppPpk_value = ppu
    return ppu, ppl, pp, ppPpk_value


def calculate_cpk_not_normal(lse: float, lie: float, tolerance_type: str, mean: float, within_sigma: float):
    """Calcula o índice de capacidade de processo (Cp, Cpk) para dados não normais."""
    cp = (lse - lie) / (6 * within_sigma)
    cpu = (lse - mean) / (3 * within_sigma)
    cpl = (mean - lie) / (3 * within_sigma)

    if tolerance_type == "bilateral":
        cpCPk_value = np.min([cpu, cpl])
    elif tolerance_type == "inferiorUnilateral":
        cpCPk_value = cpl
    else:
        cpCPk_value = cpu
    return cp, cpu, cpl, cpCPk_value


def calculate_rate(lse: float, lie: float, mean: float, sigma: float):
    """Calcula a taxa de falha baseada nos limites de especificação, média e desvio padrão do processo (PPM)."""
    overall_superior = (lse - mean) / sigma
    overall_inferior = (mean - lie) / sigma

    overall_percentil_inferior = (1 - norm.cdf(overall_superior)) * 1000000
    overall_percentil_superior = (1 - norm.cdf(overall_inferior)) * 1000000

    rate = overall_percentil_inferior + overall_percentil_superior
    return rate


def calculate_rate_inferior(lie: float, mean: float, sigma: float):
    """Calcula a taxa de falha para o limite inferior (PPM)."""
    overall_inferior = (mean - lie) / sigma
    overall_percentil_superior = (1 - norm.cdf(overall_inferior)) * 1000000
    return overall_percentil_superior


def calculate_rate_superior(lse: float, mean: float, sigma: float):
    """Calcula a taxa de falha para o limite superior (PPM)."""
    overall_superior = (lse - mean) / sigma
    overall_percentil_inferior = (1 - norm.cdf(overall_superior)) * 1000000
    return overall_percentil_inferior


def fit_weibull(data):
    """Ajusta uma distribuição Weibull aos dados fornecidos."""
    shape, _, scale = weibull_min.fit(data, floc=0)
    return shape, scale

