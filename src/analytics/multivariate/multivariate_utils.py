"""
Multivariate Analysis Utilities
Funções para análise multivariada com matriz de correlação
"""

from typing import Dict, List, Tuple
from src.utils.lazy_imports import get_pandas, get_numpy


def calculate_mean_column_values(df) -> Dict[str, float]:
    """
    Calcula a média de todas as colunas numéricas do DataFrame
    
    Args:
        df: DataFrame pandas
    
    Returns:
        Dict com nome da coluna e média
    """
    pd = get_pandas()
    return df.mean().to_dict()


def calculate_squared_diff(df, mean_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Calcula a soma das diferenças ao quadrado entre valores e médias
    
    Args:
        df: DataFrame pandas
        mean_dict: Dicionário com médias das colunas
    
    Returns:
        Dict com soma dos quadrados para cada coluna
    """
    np = get_numpy()
    result_dict = {}
    
    for column in df.columns:
        squared_diff_sum = np.sum((df[column] - mean_dict[column]) ** 2)
        result_dict[column] = squared_diff_sum
    
    return result_dict


def calculate_x_normalized_all_columns(df, mean_columns: Dict[str, float], 
                                      square: Dict[str, float]):
    """
    Normaliza todas as colunas usando (X - média) / desvio padrão
    
    Args:
        df: DataFrame pandas
        mean_columns: Médias das colunas
        square: Soma dos quadrados
    
    Returns:
        DataFrame normalizado
    """
    pd = get_pandas()
    normalized_df = pd.DataFrame(columns=df.columns)
    
    for column in df.columns:
        normalized_values = (df[column] - mean_columns[column]) / (square[column] ** 0.5)
        normalized_df[column] = normalized_values
    
    return normalized_df


def calculate_x_normalized_transpose(df):
    """
    Retorna a transposta do DataFrame normalizado
    
    Args:
        df: DataFrame normalizado
    
    Returns:
        DataFrame transposto
    """
    return df.transpose()


def calculate_correlation_matrix(transposed_matrix, df):
    """
    Calcula a matriz de correlação usando multiplicação de matrizes
    
    Args:
        transposed_matrix: Matriz transposta
        df: DataFrame original normalizado
    
    Returns:
        Matriz de correlação (numpy array)
    """
    np = get_numpy()
    np.set_printoptions(suppress=True)
    
    transposed_array = np.array(transposed_matrix)
    df_array = df.to_numpy()
    result_matrix = np.matmul(transposed_array, df_array)
    
    return np.round(result_matrix, 3)


def perform_multivariate_analysis(df) -> Tuple:
    """
    Executa análise multivariada completa
    
    Args:
        df: DataFrame com colunas numéricas
    
    Returns:
        Tuple: (correlation_matrix, column_names, normalized_df)
    """
    # Calcula médias
    mean_columns = calculate_mean_column_values(df)
    
    # Calcula soma dos quadrados
    square = calculate_squared_diff(df, mean_columns)
    
    # Normaliza dados
    x_normalized = calculate_x_normalized_all_columns(df, mean_columns, square)
    
    # Transposta
    x_normalized_transpose = calculate_x_normalized_transpose(x_normalized)
    
    # Matriz de correlação
    correlation_matrix = calculate_correlation_matrix(x_normalized_transpose, x_normalized)
    
    column_names = list(df.columns)
    
    return correlation_matrix, column_names, x_normalized


def validate_multivariate_data(df) -> Tuple[bool, str]:
    """
    Valida dados para análise multivariada
    
    Args:
        df: DataFrame
    
    Returns:
        Tuple: (is_valid, error_message)
    """
    pd = get_pandas()
    
    if df is None or df.empty:
        return False, "DataFrame está vazio"
    
    # Seleciona apenas colunas numéricas
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if len(numeric_cols) < 2:
        return False, "São necessárias pelo menos 2 colunas numéricas"
    
    if len(numeric_cols) > 20:
        return False, "Máximo de 20 colunas permitidas"
    
    # Verifica NaN
    if df[numeric_cols].isnull().any().any():
        return False, "Existem valores faltantes (NaN) nas colunas numéricas"
    
    return True, ""


def calculate_trendline(x_data: List[float], y_data: List[float]) -> List[float]:
    """
    Calcula linha de tendência linear
    
    Args:
        x_data: Dados do eixo X
        y_data: Dados do eixo Y
    
    Returns:
        Lista com valores da linha de tendência
    """
    np = get_numpy()
    
    n = len(x_data)
    sum_x = np.sum(x_data)
    sum_y = np.sum(y_data)
    sum_xy = np.sum(np.array(x_data) * np.array(y_data))
    sum_xx = np.sum(np.array(x_data) ** 2)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    intercept = (sum_y - slope * sum_x) / n
    
    return [slope * x + intercept for x in x_data]
