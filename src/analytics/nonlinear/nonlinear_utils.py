"""
Funções utilitárias para Regressão Não Linear
Implementa 25 modelos diferentes e calcula métricas (R², AIC, BIC)
"""

from src.utils.lazy_imports import get_numpy, get_pandas, get_scipy_optimize, get_scipy_special, get_scipy_stats


def get_model_equation(model_name):
    """Retorna a equação LaTeX para cada modelo"""
    equations = {
        "polynomial_2": r"$Y = \beta_0 + \beta_1 \cdot x + \beta_2 \cdot x^2$",
        "polynomial_3": r"$Y = \beta_0 + \beta_1 \cdot x + \beta_2 \cdot x^2 + \beta_3 \cdot x^3$",
        "gamma": r"$Y = e^{\beta_0 + \beta_1 \cdot x}$",
        "exponential_2p": r"$Y = a \cdot e^{b \cdot x}$",
        "exponential_3p": r"$Y = a \cdot (1 - e^{-b \cdot x})$",
        "biexponential": r"$Y = a \cdot e^{b \cdot x} + c \cdot e^{d \cdot x}$",
        "mechanistic_growth": r"$Y = a \cdot (1 - e^{-b \cdot x})$",
        "cell_growth": r"$Y = \frac{a \cdot e^{b \cdot x}}{1 + e^{c \cdot x}}$",
        "weibull_growth": r"$Y = a \cdot (1 - e^{-b \cdot x^c})$",
        "gompertz": r"$Y = a \cdot e^{-b \cdot e^{-c \cdot x}}$",
        "weibull": r"$Y = a \cdot x^b$",
        "logistic_2p": r"$Y = \frac{a}{1 + e^{-b \cdot (x - c)}}$",
        "logistic_4p": r"$Y = d + \frac{a - d}{1 + e^{-b \cdot (x - c)}}$",
        "probit_2p": r"$Y = \Phi(a + b \cdot x)$",
        "probit_4p": r"$Y = d + (a - d) \cdot \Phi(b \cdot (x - c))$",
        "gaussian_peak": r"$Y = a \cdot e^{-\frac{(x - b)^2}{2c^2}}$",
        "asymmetric_gaussian_peak": r"$Y = a \cdot e^{-\frac{(x - b)^2}{2c^2}} + d \cdot (x - b)$",
        "lorentzian_peak": r"$Y = \frac{a}{1 + (\frac{x - b}{c})^2}$",
        "logarithmic": r"$Y = a + b \cdot \ln(x)$",
        "inverse": r"$Y = a + \frac{b}{x}$",
        "sqrt_model": r"$Y = a + b \cdot \sqrt{x}$",
        "power_intercept": r"$Y = a + b \cdot x^c$",
        "michaelis_menten": r"$Y = \frac{V_{max} \cdot x}{K_m + x}$",
        "tanh_model": r"$Y = A + B \cdot \tanh(C \cdot (x - D))$",
    }
    return equations.get(model_name, "Y = f(x)")


def get_model_name_pt(model_name):
    """Retorna o nome em português para cada modelo"""
    names = {
        "polynomial_2": "Polinomial 2ª Ordem",
        "polynomial_3": "Polinomial 3ª Ordem",
        "gamma": "Gamma",
        "exponential_2p": "Exponencial 2 Parâmetros",
        "exponential_3p": "Exponencial 3 Parâmetros",
        "biexponential": "Bi-Exponencial",
        "mechanistic_growth": "Crescimento Mecanístico",
        "cell_growth": "Crescimento Celular",
        "weibull_growth": "Crescimento Weibull",
        "gompertz": "Gompertz",
        "weibull": "Weibull",
        "logistic_2p": "Logística 2 Parâmetros",
        "logistic_4p": "Logística 4 Parâmetros",
        "probit_2p": "Probit 2 Parâmetros",
        "probit_4p": "Probit 4 Parâmetros",
        "gaussian_peak": "Pico Gaussiano",
        "asymmetric_gaussian_peak": "Pico Gaussiano Assimétrico",
        "lorentzian_peak": "Pico Lorentziano",
        "logarithmic": "Logarítmico",
        "inverse": "Inverso",
        "sqrt_model": "Raiz Quadrada",
        "power_intercept": "Potência com Intercepto",
        "michaelis_menten": "Michaelis-Menten",
        "tanh_model": "Tangente Hiperbólica",
    }
    return names.get(model_name, model_name)


def calculate_metrics(y_true, y_pred, n_params):
    """
    Calcula R², AIC e BIC para um modelo
    
    Args:
        y_true: Valores observados
        y_pred: Valores preditos
        n_params: Número de parâmetros do modelo
    
    Returns:
        dict com r_squared, aic, bic
    """
    np = get_numpy()
    
    # Remove valores NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    n = len(y_true)
    
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # AIC e BIC
    mse = ss_res / n
    if mse > 0:
        log_likelihood = -n/2 * np.log(2 * np.pi * mse) - ss_res / (2 * mse)
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n) - 2 * log_likelihood
    else:
        aic = np.inf
        bic = np.inf
    
    return {
        "r_squared": float(r_squared),
        "aic": float(aic),
        "bic": float(bic)
    }


# Definição de todos os modelos
def polynomial_2(x, b0, b1, b2):
    return b0 + b1 * x + b2 * x**2


def polynomial_3(x, b0, b1, b2, b3):
    return b0 + b1 * x + b2 * x**2 + b3 * x**3


def exponential_2p(x, a, b):
    return a * get_numpy().exp(b * x)


def exponential_3p(x, a, b):
    return a * (1 - get_numpy().exp(-b * x))


def biexponential(x, a, b, c, d):
    return a * get_numpy().exp(b * x) + c * get_numpy().exp(d * x)


def mechanistic_growth(x, a, b):
    return a * (1 - get_numpy().exp(-b * x))


def cell_growth(x, a, b, c):
    np = get_numpy()
    return a * np.exp(b * x) / (1 + np.exp(c * x))


def weibull_growth(x, a, b, c):
    return a * (1 - get_numpy().exp(-b * x**c))


def gompertz(x, a, b, c):
    return a * get_numpy().exp(-b * get_numpy().exp(-c * x))


def weibull(x, a, b):
    return a * x**b


def logistic_2p(x, a, b, c):
    return a / (1 + get_numpy().exp(-b * (x - c)))


def logistic_4p(x, a, b, c, d):
    return d + (a - d) / (1 + get_numpy().exp(-b * (x - c)))


def probit_2p(x, a, b):
    from scipy.special import ndtr
    return ndtr(a + b * x)


def probit_4p(x, a, b, c, d):
    from scipy.special import ndtr
    return d + (a - d) * ndtr(b * (x - c))


def gaussian_peak(x, a, b, c):
    return a * get_numpy().exp(-((x - b)**2) / (2 * c**2))


def asymmetric_gaussian_peak(x, a, b, c, d):
    return a * get_numpy().exp(-((x - b)**2) / (2 * c**2)) + d * (x - b)


def lorentzian_peak(x, a, b, c):
    return a / (1 + ((x - b) / c)**2)


def logarithmic(x, a, b):
    return a + b * get_numpy().log(x)


def inverse(x, a, b):
    return a + b / x


def sqrt_model(x, a, b):
    return a + b * get_numpy().sqrt(x)


def power_intercept(x, a, b, c):
    return a + b * x**c


def michaelis_menten(x, vmax, km):
    return vmax * x / (km + x)


def tanh_model(x, A, B, C, D):
    return A + B * get_numpy().tanh(C * (x - D))


def fit_model(x, y, model_func, initial_params, param_names):
    """
    Ajusta um modelo não linear aos dados
    
    Args:
        x: Array de valores X
        y: Array de valores Y
        model_func: Função do modelo
        initial_params: Parâmetros iniciais para otimização
        param_names: Nomes dos parâmetros
    
    Returns:
        dict com coef (dicionário de parâmetros), metrics, y_pred ou None se falhar
    """
    from scipy.optimize import curve_fit
    np = get_numpy()
    
    try:
        # Remove NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < len(initial_params):
            return None
        
        # Ajusta o modelo
        popt, _ = curve_fit(model_func, x_clean, y_clean, p0=initial_params, maxfev=10000)
        
        # Predições
        y_pred = model_func(x_clean, *popt)
        
        # Cria dicionário de coeficientes
        coef = {name: float(val) for name, val in zip(param_names, popt)}
        
        # Calcula métricas
        metrics = calculate_metrics(y_clean, y_pred, len(initial_params))
        
        # Gera predições em uma grade fina para plotagem
        x_plot = np.linspace(x_clean.min(), x_clean.max(), 200)
        y_plot = model_func(x_plot, *popt)
        
        return {
            "coef": coef,
            "metrics": metrics,
            "x_pred": x_plot,
            "y_pred": y_plot
        }
    
    except Exception as e:
        return None


def calculate_nonlinear_regression(df, x_column, y_column):
    """
    Calcula regressão não linear para múltiplos modelos
    
    Args:
        df: DataFrame com os dados
        x_column: Nome da coluna X
        y_column: Nome da coluna Y
    
    Returns:
        dict com original (dados), predictions (dict de modelos), metrics (dict de modelos)
    """
    np = get_numpy()
    pd = get_pandas()
    
    # Extrai dados
    x = df[x_column].values.astype(float)
    y = df[y_column].values.astype(float)
    
    # Remove NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    
    # Calcula estatísticas para inicialização
    x_mean = np.mean(x)
    x_std = np.std(x)
    y_mean = np.mean(y)
    y_min = np.min(y)
    y_max = np.max(y)
    y_range = y_max - y_min
    
    # Especificações dos modelos
    model_specs = [
        ("polynomial_2", polynomial_2, [y_mean, 1, 0.1], ["Intercept", "linear", "quadratic"]),
        ("polynomial_3", polynomial_3, [y_mean, 1, 0.1, 0.01], ["Intercept", "linear", "quadratic", "cubic"]),
        ("exponential_2p", exponential_2p, [y[0] if len(y) > 0 else 1, 0.1], ["a", "b"]),
        ("exponential_3p", exponential_3p, [y_max, 0.1], ["a", "b"]),
        ("biexponential", biexponential, [y[0]/2 if len(y) > 0 else 1, 0.1, y[0]/2 if len(y) > 0 else 1, -0.1], ["a", "b", "c", "d"]),
        ("mechanistic_growth", mechanistic_growth, [y_max, 0.1], ["a", "b"]),
        ("cell_growth", cell_growth, [y_max, 0.1, 0.1], ["a", "b", "c"]),
        ("weibull_growth", weibull_growth, [y_max, 1, 1], ["a", "b", "c"]),
        ("gompertz", gompertz, [y_max, 1, 0.1], ["a", "b", "c"]),
        ("weibull", weibull, [1, 1], ["a", "b"]),
        ("logistic_2p", logistic_2p, [y_max, 1, x_mean], ["a", "b", "c"]),
        ("logistic_4p", logistic_4p, [y_max, 1, x_mean, y_min], ["a", "b", "c", "d"]),
        ("probit_2p", probit_2p, [0, 1], ["a", "b"]),
        ("probit_4p", probit_4p, [y_max, 1, x_mean, y_min], ["a", "b", "c", "d"]),
        ("gaussian_peak", gaussian_peak, [y_max, x_mean, x_std], ["a", "b", "c"]),
        ("asymmetric_gaussian_peak", asymmetric_gaussian_peak, [y_max, x_mean, x_std, 0.1], ["a", "b", "c", "d"]),
        ("lorentzian_peak", lorentzian_peak, [y_max, x_mean, x_std], ["a", "b", "c"]),
        ("logarithmic", logarithmic, [y_min, 1], ["a", "b"]),
        ("inverse", inverse, [y_mean, 1], ["a", "b"]),
        ("sqrt_model", sqrt_model, [y_min, 1], ["a", "b"]),
        ("power_intercept", power_intercept, [y_min, 1, 1], ["a", "b", "c"]),
        ("michaelis_menten", michaelis_menten, [y_max, x_mean], ["Vmax", "Km"]),
        ("tanh_model", tanh_model, [y_mean, y_range/2, 1, x_mean], ["A", "B", "C", "D"]),
    ]
    
    results = {}
    metrics = {}
    predictions = {}
    
    for model_name, model_func, initial_params, param_names in model_specs:
        result = fit_model(x, y, model_func, initial_params, param_names)
        
        if result is not None:
            results[model_name] = result
            metrics[model_name] = {
                "coef": result["coef"],
                "rSquared": result["metrics"]["r_squared"],
                "aic": result["metrics"]["aic"],
                "bic": result["metrics"]["bic"],
                "equation": get_model_equation(model_name)
            }
            predictions[model_name] = [
                {"x": float(x_val), "y": float(y_val)} 
                for x_val, y_val in zip(result["x_pred"], result["y_pred"])
            ]
    
    # Dados originais
    original = [{"x": float(x_val), "y": float(y_val)} for x_val, y_val in zip(x, y)]
    
    return {
        "original": original,
        "predictions": predictions,
        "metrics": metrics
    }
