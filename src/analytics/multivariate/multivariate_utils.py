"""
Multivariate Analysis Utilities
Funções para análise multivariada com matriz de correlação
"""

from typing import Dict, List, Tuple
from src.utils.lazy_imports import get_pandas, get_numpy
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


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


def calculate_correlation_with_pvalues(df):
    """
    Calcula matriz de correlação com p-values usando pandas
    
    Args:
        df: DataFrame com dados numéricos
    
    Returns:
        Tuple: (correlation_matrix, pvalue_matrix)
    """
    pd = get_pandas()
    np = get_numpy()
    
    # Usar pandas corr() que é mais eficiente
    corr_matrix = df.corr().values
    
    # Calcular p-values
    n = len(df)
    pvalue_matrix = np.zeros_like(corr_matrix)
    
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            if i != j:
                r = corr_matrix[i, j]
                # t-statistic para correlação de Pearson
                t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r**2) if abs(r) < 1 else 0
                # p-value bilateral
                pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                pvalue_matrix[i, j] = pvalue
            else:
                pvalue_matrix[i, j] = 0  # Diagonal sempre 0
    
    return corr_matrix, pvalue_matrix


def calculate_vif(df):
    """
    Calcula VIF (Variance Inflation Factor) para detectar multicolinearidade
    
    Args:
        df: DataFrame com dados numéricos
    
    Returns:
        Dict com VIF para cada variável
    """
    pd = get_pandas()
    np = get_numpy()
    from sklearn.linear_model import LinearRegression
    
    vif_data = {}
    columns = df.columns.tolist()
    
    for i, col in enumerate(columns):
        # Usar outras colunas como features
        X = df.drop(columns=[col])
        y = df[col]
        
        # Regressão linear
        model = LinearRegression()
        model.fit(X, y)
        
        # R²
        r_squared = model.score(X, y)
        
        # VIF = 1 / (1 - R²)
        if r_squared < 0.9999:  # Evitar divisão por zero
            vif = 1 / (1 - r_squared)
        else:
            vif = np.inf
        
        vif_data[col] = vif
    
    return vif_data


def calculate_hierarchical_clustering(corr_matrix, method='average'):
    """
    Calcula clustering hierárquico para agrupar variáveis correlacionadas
    
    Args:
        corr_matrix: Matriz de correlação
        method: Método de linkage ('average', 'complete', 'single')
    
    Returns:
        Linkage matrix para dendrograma
    """
    np = get_numpy()
    
    # Converter correlação para distância: dist = 1 - |corr|
    distance_matrix = 1 - np.abs(corr_matrix)
    
    # Converter para forma condensada
    condensed_dist = squareform(distance_matrix, checks=False)
    
    # Clustering hierárquico
    linkage_matrix = hierarchy.linkage(condensed_dist, method=method)
    
    return linkage_matrix


def interpret_correlation(corr_value, pvalue):
    """
    Interpreta força e significância da correlação
    
    Args:
        corr_value: Valor da correlação
        pvalue: P-value da correlação
    
    Returns:
        Dict com interpretação
    """
    # Força da correlação (regra de Cohen)
    abs_corr = abs(corr_value)
    
    if abs_corr >= 0.9:
        strength = 'Muito Forte'
        color = 'darkgreen' if pvalue < 0.05 else 'green'
    elif abs_corr >= 0.7:
        strength = 'Forte'
        color = 'green' if pvalue < 0.05 else 'lightgreen'
    elif abs_corr >= 0.5:
        strength = 'Moderada'
        color = 'orange' if pvalue < 0.05 else 'yellow'
    elif abs_corr >= 0.3:
        strength = 'Fraca'
        color = 'orange'
    else:
        strength = 'Muito Fraca'
        color = 'lightgray'
    
    # Significância
    if pvalue < 0.001:
        significance = '***'
        sig_text = 'Altamente significativo (p < 0.001)'
    elif pvalue < 0.01:
        significance = '**'
        sig_text = 'Muito significativo (p < 0.01)'
    elif pvalue < 0.05:
        significance = '*'
        sig_text = 'Significativo (p < 0.05)'
    else:
        significance = 'ns'
        sig_text = 'Não significativo (p ≥ 0.05)'
    
    # Direção
    direction = 'Positiva' if corr_value > 0 else 'Negativa' if corr_value < 0 else 'Nula'
    
    return {
        'strength': strength,
        'significance': significance,
        'sig_text': sig_text,
        'direction': direction,
        'color': color,
        'pvalue': pvalue
    }


def interpret_vif(vif_value):
    """
    Interpreta valor de VIF
    
    Args:
        vif_value: Valor do VIF
    
    Returns:
        Dict com interpretação
    """
    np = get_numpy()
    
    if np.isinf(vif_value):
        return {
            'status': 'Perfeita Colinearidade',
            'color': 'red',
            'message': 'VIF = ∞ - Colinearidade perfeita, remova esta variável'
        }
    elif vif_value > 10:
        return {
            'status': 'Alta Colinearidade',
            'color': 'red',
            'message': f'VIF = {vif_value:.2f} - Alta multicolinearidade, considere remover'
        }
    elif vif_value > 5:
        return {
            'status': 'Colinearidade Moderada',
            'color': 'yellow',
            'message': f'VIF = {vif_value:.2f} - Multicolinearidade moderada'
        }
    else:
        return {
            'status': 'Sem Colinearidade',
            'color': 'green',
            'message': f'VIF = {vif_value:.2f} - Sem problemas de multicolinearidade'
        }
