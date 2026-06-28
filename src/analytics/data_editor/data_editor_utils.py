"""
Motor de fórmulas para o Editor de Dados.
Suporta distribuições estatísticas, sequências e expressões com referência a colunas.
"""
import numpy as np


# Funções disponíveis nas fórmulas
FORMULA_FUNCTIONS = {
    # Distribuições contínuas
    'NORMAL':       lambda mean, std, n=1:    np.random.normal(float(mean), float(std), int(n)),
    'UNIFORME':     lambda low, high, n=1:    np.random.uniform(float(low), float(high), int(n)),
    'LOGNORMAL':    lambda mean, sigma, n=1:  np.random.lognormal(float(mean), float(sigma), int(n)),
    'EXPONENCIAL':  lambda scale, n=1:        np.random.exponential(float(scale), int(n)),
    'TRIANGULAR':   lambda low, mode, high, n=1: np.random.triangular(float(low), float(mode), float(high), int(n)),
    'BETA':         lambda a, b, n=1:         np.random.beta(float(a), float(b), int(n)),
    'GAMMA':        lambda shape, scale, n=1: np.random.gamma(float(shape), float(scale), int(n)),
    'WEIBULL':      lambda a, n=1:            np.random.weibull(float(a), int(n)),
    # Distribuições discretas
    'POISSON':      lambda lam, n=1:          np.random.poisson(float(lam), int(n)).astype(float),
    'BINOMIAL':     lambda n_t, p, n=1:       np.random.binomial(int(n_t), float(p), int(n)).astype(float),
    'RANDINT':      lambda low, high, n=1:    np.random.randint(int(low), int(high) + 1, int(n)).astype(float),
    # Sequências
    'SEQUENCIA':    lambda start, stop, step=1: np.arange(float(start), float(stop) + float(step) * 0.5, float(step)),
    'LINSPACE':     lambda start, stop, n:    np.linspace(float(start), float(stop), int(n)),
    'ZEROS':        lambda n:                 np.zeros(int(n)),
    'UNS':          lambda n:                 np.ones(int(n)),
    'REPETIR':      lambda val, n:            np.repeat(float(val), int(n)),
    # Aliases em inglês
    'UNIFORM':      lambda low, high, n=1:    np.random.uniform(float(low), float(high), int(n)),
    'EXPONENTIAL':  lambda scale, n=1:        np.random.exponential(float(scale), int(n)),
    'SEQUENCE':     lambda start, stop, step=1: np.arange(float(start), float(stop) + float(step) * 0.5, float(step)),
    'REPEAT':       lambda val, n:            np.repeat(float(val), int(n)),
    # Matemática
    'ROUND':  np.round,
    'ABS':    np.abs,
    'SQRT':   np.sqrt,
    'LOG':    np.log,
    'LOG10':  np.log10,
    'EXP':    np.exp,
    'SIN':    np.sin,
    'COS':    np.cos,
    'TAN':    np.tan,
    'FLOOR':  np.floor,
    'CEIL':   np.ceil,
    'MIN':    np.minimum,
    'MAX':    np.maximum,
    'SOMA':   np.sum,
    'MEDIA':  np.mean,
    'STD':    np.std,
    # Constantes
    'PI': np.pi,
    'E':  np.e,
    'np': np,
}

FORMULA_HELP = [
    ("=NORMAL(média, desvio, n)",        "Ex: =NORMAL(100, 15, 50) → 50 valores normais"),
    ("=UNIFORME(min, max, n)",            "Ex: =UNIFORME(0, 100, 50)"),
    ("=LOGNORMAL(média, sigma, n)",       "Ex: =LOGNORMAL(0, 0.5, 50)"),
    ("=EXPONENCIAL(escala, n)",           "Ex: =EXPONENCIAL(10, 50)"),
    ("=TRIANGULAR(min, moda, max, n)",    "Ex: =TRIANGULAR(5, 10, 20, 50)"),
    ("=WEIBULL(a, n)",                    "Ex: =WEIBULL(1.5, 50)"),
    ("=POISSON(lambda, n)",               "Ex: =POISSON(5, 50)"),
    ("=BINOMIAL(n_tentativas, p, n)",     "Ex: =BINOMIAL(10, 0.3, 50)"),
    ("=RANDINT(min, max, n)",             "Ex: =RANDINT(1, 6, 50) → dado"),
    ("=SEQUENCIA(início, fim, passo)",    "Ex: =SEQUENCIA(1, 100, 1)"),
    ("=LINSPACE(início, fim, n)",         "Ex: =LINSPACE(0, 1, 100)"),
    ("=REPETIR(valor, n)",                "Ex: =REPETIR(3.14, 50)"),
    ("Referência de colunas",             "Ex: =ColA * 2 + ColB   (use o nome da coluna)"),
    ("Expressão matemática",              "Ex: =SQRT(ColA) + LOG(ColB)"),
]


def evaluate_formula(formula: str, columns: dict = None, n_rows: int = 10) -> list:
    """
    Avalia uma fórmula e retorna lista de valores float.

    Args:
        formula: string SEM o '=' inicial
        columns: dict {nome_coluna: list_de_valores}
        n_rows: número de linhas atual (usado como N padrão)

    Returns:
        list de float
    """
    safe_globals = dict(FORMULA_FUNCTIONS)
    safe_globals['__builtins__'] = {'range': range, 'int': int, 'float': float, 'len': len}

    # Injeta colunas como arrays numpy
    if columns:
        for col_name, values in columns.items():
            try:
                arr = np.array([float(v) if v not in ('', None) else np.nan for v in values])
                safe_globals[col_name] = arr
                # versão sem espaços
                safe_globals[col_name.replace(' ', '_')] = arr
            except Exception:
                pass

    # Substitui N pelo n_rows atual
    expr = formula.replace('{N}', str(n_rows)).replace(' N)', f' {n_rows})')

    try:
        result = eval(expr, safe_globals, {})
    except Exception as e:
        raise ValueError(f"Erro ao avaliar '{formula}': {e}")

    if isinstance(result, np.ndarray):
        return [float(x) if np.isfinite(x) else None for x in result]
    elif isinstance(result, (int, float)):
        return [float(result)] * n_rows
    elif isinstance(result, (list, tuple)):
        return [float(x) if x is not None else None for x in result]
    else:
        try:
            return [float(result)] * n_rows
        except Exception:
            raise ValueError(f"Resultado inesperado do tipo {type(result)}")


def format_value(v) -> str:
    """Formata valor para exibição na célula."""
    if v is None or v == '':
        return ''
    try:
        f = float(v)
        if f == int(f):
            return str(int(f))
        return f'{f:.6g}'
    except (TypeError, ValueError):
        return str(v)
