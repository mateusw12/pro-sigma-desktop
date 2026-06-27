"""
Sample Size Explorer utilities.
Ported from weg-statistics-python/app/sample_size_explorer/service/sample_size_explorer_service.py
"""
import math
from statistics import NormalDist
from scipy.stats import chi2


_nd = NormalDist()


def _z_from_confidence(confidence_level: float) -> float:
    if not (0 < confidence_level < 1):
        raise ValueError("Nível de confiança deve estar entre 0 e 1 (ex: 0.95)")
    cum_prob = 1 - (1 - confidence_level) / 2
    return _nd.inv_cdf(cum_prob)


def _sigma_error_pct(sample_size: int, confidence_level: float) -> float:
    if sample_size <= 1:
        return 0.0
    df = sample_size - 1
    upper_p = 1 - (1 - confidence_level) / 2
    lower_p = (1 - confidence_level) / 2
    upper_chi = chi2.ppf(upper_p, df)
    lower_chi = chi2.ppf(lower_p, df)
    upper_r = math.sqrt(df / upper_chi)
    lower_r = math.sqrt(df / lower_chi)
    return 0.5 * abs(upper_r - lower_r) * 100


def calc_sample_size(confidence_level: float, margin_pct: float) -> dict:
    """
    Calcula N dado nível de confiança e margem de erro (%).
    confidence_level: 0 < cl < 1 (ex: 0.95)
    margin_pct: % (ex: 5.0 para 5%)
    """
    if margin_pct <= 0:
        raise ValueError("Margem de erro deve ser maior que zero")
    z = _z_from_confidence(confidence_level)
    margin_dec = margin_pct / 100
    n = math.ceil((z / (6 * margin_dec)) ** 2)
    return {
        'sample_size': n,
        'z_score': round(z, 6),
        'sigma_error_pct': round(_sigma_error_pct(n, confidence_level), 6),
    }


def calc_margin_of_error(sample_size: int, confidence_level: float) -> dict:
    """
    Calcula margem de erro (%) dado N e nível de confiança.
    """
    if sample_size < 1:
        raise ValueError("N deve ser pelo menos 1")
    z = _z_from_confidence(confidence_level)
    margin_dec = z / (6 * math.sqrt(sample_size))
    return {
        'margin_pct': round(margin_dec * 100, 6),
        'z_score': round(z, 6),
        'sigma_error_pct': round(_sigma_error_pct(sample_size, confidence_level), 6),
    }


def calc_confidence_level(sample_size: int, margin_pct: float) -> dict:
    """
    Calcula nível de confiança dado N e margem de erro (%).
    """
    if sample_size < 1:
        raise ValueError("N deve ser pelo menos 1")
    if margin_pct <= 0:
        raise ValueError("Margem de erro deve ser maior que zero")
    margin_dec = margin_pct / 100
    z = 6 * margin_dec * math.sqrt(sample_size)
    cl = max(min(2 * _nd.cdf(z) - 1, 0.999999), 0.0)
    return {
        'confidence_level': round(cl, 6),
        'z_score': round(z, 6),
        'sigma_error_pct': round(_sigma_error_pct(sample_size, cl), 6),
    }


def sensitivity_table(confidence_levels: list, margins_pct: list) -> list:
    """
    Returns list of rows: {confidence_level, margin_pct, sample_size}
    """
    rows = []
    for cl in confidence_levels:
        for m in margins_pct:
            try:
                r = calc_sample_size(cl, m)
                rows.append({'confidence': cl, 'margin_pct': m, 'sample_size': r['sample_size']})
            except Exception:
                rows.append({'confidence': cl, 'margin_pct': m, 'sample_size': None})
    return rows
