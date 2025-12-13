"""
Hypothesis Test Utility Functions
Statistical hypothesis testing calculations
"""
from typing import List, Dict
from itertools import combinations
import string
from src.utils.lazy_imports import get_pandas, get_numpy, get_scipy_stats

# Lazy imports - carregados apenas quando usados
pd = None
np = None
stats = None
t = None

def _ensure_imports():
    """Garante que imports pesados estão carregados"""
    global pd, np, stats, t
    if pd is None:
        pd = get_pandas()
        np = get_numpy()
        stats = get_scipy_stats()
        t = stats.t
    return pd, np, stats, t


def remove_punctuation(df):
    """Remove pontuação dos valores do DataFrame"""
    _ensure_imports()
    return df.map(
        lambda x: (
            "".join([char for char in str(x) if char not in string.punctuation])
            if isinstance(x, str)
            else x
        )
    )


def convert_first_column_to_string(df):
    """Converte valores da primeira coluna para string"""
    _ensure_imports()
    first_col = df.columns[0]
    if not df[first_col].apply(lambda x: isinstance(x, str)).all():
        df[first_col] = first_col + df[first_col].astype(str)
    return df.copy()


def calculate_average_response(df) -> Dict[str, float]:
    """Calcula a média da resposta para cada grupo"""
    _ensure_imports()
    averages: Dict[str, float] = {}
    grouped = df.groupby(df.columns[0])
    for group_name, group_df in grouped:
        average = group_df.iloc[:, -1].mean()
        averages[group_name] = average
    return averages


def calculate_std_deviation(df) -> Dict[str, float]:
    """Calcula o desvio padrão da resposta para cada grupo"""
    _ensure_imports()
    std_deviation: Dict[str, float] = {}
    grouped = df.groupby(df.columns[0])
    
    for group_name, group_df in grouped:
        std = group_df.iloc[:, -1].std()
        std_deviation[group_name] = std
    return std_deviation


def generate_unique_combinations(df) -> List[str]:
    """Gera combinações únicas dos valores da primeira coluna"""
    _ensure_imports()
    first_column = df.iloc[:, 0]
    unique_rows = first_column.unique()
    combinations_list = list(combinations(unique_rows, 2))
    iterations = [f"{item[0]}-{item[1]}" for item in combinations_list]
    return iterations


def count_repeated_rows(df) -> int:
    """Conta o número de repetições do valor mais comum na primeira coluna"""
    _ensure_imports()
    first_column = df.iloc[:, 0]
    value_counts = first_column.value_counts()
    return value_counts.iloc[0]


def calculate_average_std_error_difference(response_values: Dict) -> float:
    """Calcula a média dos erros padrão das diferenças"""
    total_std_error_difference = sum(
        item["stdErrorDifference"] for item in response_values.values()
    )
    return total_std_error_difference / len(response_values) if len(response_values) != 0 else 0


def calculate_degrees_freedom(df) -> int:
    """Calcula graus de liberdade"""
    _ensure_imports()
    num_rows = df.shape[0]
    return num_rows - 1


def calculate_difference_std_error_difference(
    average: Dict[str, float],
    std_deviation: Dict[str, float],
    iterations: List[str],
    observation_number_group: int
) -> Dict:
    """Calcula diferenças entre médias e erros padrão das diferenças"""
    response_values = {}
    
    for iteration in iterations:
        items = iteration.split("-")
        if len(items) == 2:
            a, b = items
            if a in average and b in average:
                # Calcula diferença
                difference = average[a] - average[b]
                
                # Calcula Std Error Difference
                std_deviation_difference_a = (std_deviation[a] ** 2) / observation_number_group
                std_deviation_difference_b = (std_deviation[b] ** 2) / observation_number_group
                std_deviation_difference = (std_deviation_difference_a + std_deviation_difference_b) ** 0.5
                
                response_values[iteration] = {
                    "difference": abs(difference),
                    "stdErrorDifference": std_deviation_difference
                }
        else:
            response_values[iteration] = {"difference": 0, "stdErrorDifference": 0}
    
    return response_values


def calculate_fRatio_p_value(
    response_values: Dict,
    average_std_error_difference: float,
    degrees_freedom: int
) -> Dict:
    """Calcula F-Ratio e p-Value para as diferenças"""
    for key, values in response_values.items():
        difference = values["difference"]
        
        # Cálculo do F Ratio
        f_ratio = abs(difference) / average_std_error_difference if average_std_error_difference != 0 else 0
        response_values[key]["fRatio"] = f_ratio
        
        # Cálculo do p-Value
        p_value = 2 * t.sf(abs(f_ratio), degrees_freedom)
        response_values[key]["pValue"] = p_value
    
    return response_values


def calculate_ci(response_values: Dict, degrees_freedom: int) -> Dict:
    """Calcula intervalos de confiança inferior e superior"""
    critical_t = t.ppf(1 - 0.05 / 2, degrees_freedom)
    
    for key, values in response_values.items():
        difference = values["difference"]
        std_error_difference = values["stdErrorDifference"]
        
        # Calcula CI Inferior
        ci_inferior = difference - critical_t * std_error_difference
        
        # Calcula CI Superior
        ci_superior = difference + critical_t * std_error_difference
        
        response_values[key]["ciInferior"] = ci_inferior
        response_values[key]["ciSuperior"] = ci_superior
    
    return response_values


def calculate_mean_difference(df, response_columns: List[str]) -> Dict:
    """
    Calcula teste de diferença de média para múltiplas respostas
    
    Args:
        df: DataFrame com dados
        response_columns: Lista de colunas de resposta
    
    Returns:
        Dicionário com resultados para cada coluna de resposta
    """
    df = remove_punctuation(df)
    mse_response = {}
    
    # Split dataframes por coluna de resposta
    for response_column in response_columns:
        if response_column not in df.columns:
            continue
        
        # Seleciona colunas não-resposta + resposta atual
        other_cols = [col for col in df.columns if col not in response_columns]
        work_df = df[other_cols + [response_column]].copy()
        work_df = work_df.dropna()
        
        if len(work_df) < 2:
            continue
        
        # Converte primeira coluna para string
        work_df = convert_first_column_to_string(work_df)
        
        # Calcula média de tratamento
        averages = calculate_average_response(work_df)
        
        # Calcula desvio padrão
        std_deviation = calculate_std_deviation(work_df)
        
        # Gera iterações
        iterations = generate_unique_combinations(work_df)
        
        # Conta número de observações por grupo
        observation_number_group = count_repeated_rows(work_df)
        
        # Calcula difference e stdError difference
        response_values = calculate_difference_std_error_difference(
            averages, std_deviation, iterations, observation_number_group
        )
        
        # Calcula média do stdError difference
        average_std_error_difference = calculate_average_std_error_difference(response_values)
        
        # Calcula graus de liberdade
        degrees_freedom = calculate_degrees_freedom(work_df)
        
        # Calcula F-Ratio e P-Value
        response_values = calculate_fRatio_p_value(
            response_values, average_std_error_difference, degrees_freedom
        )
        
        # Calcula CI
        response_values = calculate_ci(response_values, degrees_freedom)
        
        mse_response[response_column] = {"meanDifference": response_values}
    
    return {"mse": mse_response}


def calculate_one_way_anova(df, response_columns: List[str]) -> Dict:
    """
    Calcula One-Way ANOVA para múltiplas respostas
    
    Args:
        df: DataFrame com dados
        response_columns: Lista de colunas de resposta
    
    Returns:
        Dicionário com resultados ANOVA para cada coluna de resposta
    """
    df = remove_punctuation(df)
    one_way_anova_response = {}
    
    for response_column in response_columns:
        if response_column not in df.columns:
            continue
        
        # Seleciona colunas não-resposta + resposta atual
        other_cols = [col for col in df.columns if col not in response_columns]
        work_df = df[other_cols + [response_column]].copy()
        work_df = work_df.dropna()
        
        # Validações mínimas
        if len(work_df) < 2 or len(other_cols) == 0:
            continue
        
        # Verifica se há pelo menos 2 grupos diferentes
        first_column = work_df.columns[0]
        num_groups = work_df[first_column].nunique()
        if num_groups < 2:
            print(f"ANOVA requires at least 2 groups for {response_column}, found {num_groups}")
            continue
        
        try:
            # Calcula média dos tratamentos
            first_column = work_df.columns[0]
            last_column = work_df.columns[-1]
            grouped_mean = work_df.groupby(first_column)[last_column].mean()
            average_dict = dict(zip(grouped_mean.index, grouped_mean.values))
            
            # Calcula média da coluna de resposta
            mean_column = work_df[last_column].mean()
            
            # Calcula soma dos erros
            sq_errors = []
            for _, row in work_df.iterrows():
                first_row_key = row[first_column]
                if first_row_key in average_dict:
                    average_value = average_dict[first_row_key]
                    valor_ultimo_coluna = row[last_column]
                    sq_errors.append((valor_ultimo_coluna - average_value) ** 2)
            
            sq_error_total = sum(sq_errors)
            
            # Calcula a soma dos modelos
            sq_models = {}
            for key, item in average_dict.items():
                sq_models[key] = (item - mean_column) ** 2
            
            # Calcula a soma total dos modelos ponderada pela contagem
            sq_models_total = 0
            for key, sq_model in sq_models.items():
                # Multiplica pelo número de observações deste grupo
                group_count = (work_df[first_column] == key).sum()
                sq_models_total += sq_model * group_count
            
            # Calcula o número de valores únicos
            tratment_number = work_df[first_column].nunique()
            
            # Número total de linhas
            total_rows = work_df.shape[0]
            
            # Calcula a tabela ANOVA
            degrees_freedom_total = total_rows - 1
            degrees_freedom_model = tratment_number - 1
            degrees_freedom_error = degrees_freedom_total - degrees_freedom_model
            
            # Evita divisão por zero
            if degrees_freedom_model <= 0 or degrees_freedom_error <= 0:
                print(f"Insufficient degrees of freedom for {response_column}")
                continue
            
            m_square_model = sq_models_total / degrees_freedom_model
            m_square_error = sq_error_total / degrees_freedom_error
            
            # Evita divisão por zero no F-ratio
            if m_square_error == 0:
                print(f"Error variance is zero for {response_column}")
                continue
                
            f_ratio = m_square_model / m_square_error
            
            # Calcula p-value
            from scipy.stats import f as f_dist
            prob_f = f_dist.sf(f_ratio, degrees_freedom_model, degrees_freedom_error)
            
            anova = {
                "grausLiberdade": {
                    "modelo": degrees_freedom_model,
                    "erro": degrees_freedom_error,
                    "total": degrees_freedom_total
                },
                "sQuadrados": {
                    "modelo": sq_models_total,
                    "erro": sq_error_total,
                    "total": sq_models_total + sq_error_total
                },
                "mQuadrados": {
                    "modelo": m_square_model,
                    "erro": m_square_error
                },
                "fRatio": f_ratio,
                "probF": prob_f
            }
            
            # Calcula Summary of Fit
            total_sum_squares = sq_error_total + sq_models_total
            
            # Evita divisão por zero
            if total_sum_squares == 0:
                r_square = 0
                r_square_adjust = 0
            else:
                r_square = sq_models_total / total_sum_squares
                
                # Calcula R² ajustado
                denominator = total_rows - degrees_freedom_model - 1
                if denominator <= 0:
                    r_square_adjust = r_square
                else:
                    r_square_adjust = 1 - ((1 - r_square) * (total_rows - 1)) / denominator
            
            summary_of_fit = {
                "rQuadrado": r_square,
                "rQuadradoAjustado": r_square_adjust,
                "rmse": m_square_error ** 0.5,
                "media": mean_column,
                "observacoes": total_rows
            }
            
            # Prepare dataframe for chart
            data_for_chart = []
            for _, row in work_df.iterrows():
                data_for_chart.append({
                    "x": str(row[first_column]),
                    "y": row[last_column]
                })
            
            one_way_anova_response[response_column] = {
                "anova": anova,
                "summaryOfFit": summary_of_fit,
                "dataFrame": data_for_chart
            }
            
        except Exception as e:
            print(f"Error calculating ANOVA for {response_column}: {e}")
            continue
    
    return {"oneWayAnova": one_way_anova_response}


def calculate_t_test_expected_mean(
    df,
    response_column: str,
    expected_mean: float
) -> Dict:
    """
    Calcula t-test comparando amostra com valor esperado
    
    Args:
        df: DataFrame com dados
        response_column: Coluna de resposta
        expected_mean: Média esperada para comparação
    
    Returns:
        Dicionário com resultados do t-test
    """
    _ensure_imports()
    
    if response_column not in df.columns:
        return {}
    
    data = df[response_column].dropna()
    
    if len(data) < 2:
        return {}
    
    # Calcula estatísticas da amostra
    sample_mean = data.mean()
    sample_std = data.std()
    n = len(data)
    
    # Calcula t-statistic
    t_statistic = (sample_mean - expected_mean) / (sample_std / np.sqrt(n))
    
    # Calcula p-value (two-tailed)
    degrees_freedom = n - 1
    p_value = 2 * t.sf(abs(t_statistic), degrees_freedom)
    
    # Calcula intervalo de confiança
    critical_t = t.ppf(1 - 0.05 / 2, degrees_freedom)
    margin_error = critical_t * (sample_std / np.sqrt(n))
    ci_lower = sample_mean - margin_error
    ci_upper = sample_mean + margin_error
    
    return {
        "sampleMean": sample_mean,
        "expectedMean": expected_mean,
        "sampleStd": sample_std,
        "n": n,
        "tStatistic": t_statistic,
        "pValue": p_value,
        "degreesOfFreedom": degrees_freedom,
        "ciLower": ci_lower,
        "ciUpper": ci_upper
    }


def calculate_t_test_sample(df, response_columns: List[str], expected_mean: float = 0) -> Dict:
    """
    Calcula teste t de amostra (one-sample ou paired-sample)
    
    Args:
        df: DataFrame com dados
        response_columns: Lista de colunas de resposta
        expected_mean: Média esperada para comparação (usado apenas em one-sample)
    
    Returns:
        Dicionário com resultados do teste t
    """
    _ensure_imports()
    
    df = remove_punctuation(df)
    results = {}
    
    # Se há apenas 1 coluna: one-sample t-test (não usado mais, mas mantido para compatibilidade)
    if len(response_columns) == 1:
        col = response_columns[0]
        if col not in df.columns:
            return {}
        
        data = df[col].dropna()
        
        if len(data) < 2:
            return {}
        
        mean = data.mean()
        std_dev = data.std(ddof=1)
        n = len(data)
        
        # Calcula t-statistic
        t_calculate = (mean - expected_mean) / (std_dev / np.sqrt(n))
        
        # Calcula p-value (two-tailed)
        degrees_freedom = n - 1
        p_value = 2 * (1 - t.cdf(abs(t_calculate), df=degrees_freedom))
        
        results["result"] = {
            "type": "one-sample",
            "column": col,
            "mean": mean,
            "std": std_dev,
            "tCalculate": t_calculate,
            "pValue": p_value,
            "yValues": {col: data.tolist()}
        }
        
    # Se há 2 ou mais colunas: paired-sample t-test
    else:
        col_pairs = list(combinations(response_columns, 2))
        pair_results = {}
        all_y_values = {}
        
        for col1, col2 in col_pairs:
            if col1 not in df.columns or col2 not in df.columns:
                continue
            
            # Pega apenas linhas sem NaN em ambas colunas
            work_df = df[[col1, col2]].dropna()
            
            if len(work_df) < 2:
                continue
            
            # Calcula diferença absoluta entre as colunas
            diff_series = abs(work_df[col1] - work_df[col2])
            mean_diff = diff_series.mean()
            std_dev = diff_series.std(ddof=1)
            n = len(diff_series)
            
            # Evita divisão por zero
            if std_dev == 0:
                t_calculate = 0.0
                p_value = 1.0
            else:
                t_calculate = mean_diff / (std_dev / np.sqrt(n))
                degrees_freedom = n - 1
                p_value = 2 * (1 - t.cdf(abs(t_calculate), df=degrees_freedom))
            
            pair_key = f"{col1} - {col2}"
            pair_results[pair_key] = {
                "pair": pair_key,
                "mean": mean_diff,
                "std": std_dev,
                "tCalculate": t_calculate,
                "pValue": p_value,
                "n": n
            }
            
            # Armazena os valores Y
            all_y_values[col1] = work_df[col1].tolist()
            all_y_values[col2] = work_df[col2].tolist()
        
        if pair_results:
            results["result"] = {
                "type": "paired-sample",
                "pairs": pair_results,
                "yValues": all_y_values
            }
    
    return results
