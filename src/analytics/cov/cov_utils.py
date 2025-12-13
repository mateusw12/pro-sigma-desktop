"""
COV EMS Utility Functions
Component of Variance - Expected Mean Squares Analysis
"""
import statsmodels.api as sm
import statsmodels.formula.api as smf
from typing import List, Dict, Tuple


# Tabela de constantes para subgrupos
TABELA_SUBGROUP = {
    2: {'A2': 1.880, 'd2': 1.128, 'D3': 0, 'D4': 3.267},
    3: {'A2': 1.023, 'd2': 1.693, 'D3': 0, 'D4': 2.574},
    4: {'A2': 0.729, 'd2': 2.059, 'D3': 0, 'D4': 2.282},
    5: {'A2': 0.577, 'd2': 2.326, 'D3': 0, 'D4': 2.114},
    6: {'A2': 0.483, 'd2': 2.534, 'D3': 0, 'D4': 2.004},
    7: {'A2': 0.419, 'd2': 2.704, 'D3': 0.076, 'D4': 1.924},
    8: {'A2': 0.373, 'd2': 2.847, 'D3': 0.136, 'D4': 1.864},
    9: {'A2': 0.337, 'd2': 2.970, 'D3': 0.184, 'D4': 1.816},
    10: {'A2': 0.308, 'd2': 3.078, 'D3': 0.223, 'D4': 1.777},
    11: {'A2': 0.285, 'd2': 3.173, 'D3': 0.256, 'D4': 1.744},
    12: {'A2': 0.266, 'd2': 3.258, 'D3': 0.283, 'D4': 1.717},
    13: {'A2': 0.249, 'd2': 3.336, 'D3': 0.307, 'D4': 1.693},
    14: {'A2': 0.235, 'd2': 3.407, 'D3': 0.328, 'D4': 1.672},
    15: {'A2': 0.223, 'd2': 3.472, 'D3': 0.347, 'D4': 1.653},
    16: {'A2': 0.212, 'd2': 3.532, 'D3': 0.363, 'D4': 1.637},
    17: {'A2': 0.203, 'd2': 3.588, 'D3': 0.378, 'D4': 1.622},
    18: {'A2': 0.194, 'd2': 3.640, 'D3': 0.391, 'D4': 1.608},
    19: {'A2': 0.187, 'd2': 3.689, 'D3': 0.403, 'D4': 1.597},
    20: {'A2': 0.180, 'd2': 3.735, 'D3': 0.415, 'D4': 1.585},
    21: {'A2': 0.173, 'd2': 3.778, 'D3': 0.425, 'D4': 1.575},
    22: {'A2': 0.167, 'd2': 3.819, 'D3': 0.434, 'D4': 1.566},
    23: {'A2': 0.162, 'd2': 3.858, 'D3': 0.443, 'D4': 1.557},
    24: {'A2': 0.157, 'd2': 3.895, 'D3': 0.451, 'D4': 1.548},
    25: {'A2': 0.153, 'd2': 3.931, 'D3': 0.459, 'D4': 1.541}
}


def remove_punctuation(df: pd.DataFrame) -> pd.DataFrame:
    """Remove pontuação dos valores das colunas"""
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace('[^\w\s]', '', regex=True)
    return df


def replace_data_frame(itens: List[str], df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas onde todos os valores são idênticos"""
    itens_to_remove = []
    for col in itens:
        if col in df.columns and np.all(df[col] == df[col].iloc[0]):
            itens_to_remove.append(col)
    
    for col in itens_to_remove:
        itens.remove(col)
        df.drop(col, inplace=True, axis=1)
    
    return df


def fit_linear_regression(df: pd.DataFrame, formula: str):
    """Ajusta modelo de regressão linear"""
    try:
        model = smf.ols(formula, data=df).fit()
        return model
    except:
        return None


def calculate_anova_table(model):
    """Calcula tabela ANOVA"""
    table_to_read = sm.stats.anova_lm(model, typ=2)
    table_to_read = table_to_read.fillna('')
    return table_to_read


def construct_main_effects_formula(items: List[str]) -> str:
    """Constrói fórmula de efeitos principais"""
    formula = ''
    for index, item in enumerate(items):
        if index < len(items) - 2:
            formula += 'C(' + item + ') + '
        elif index == len(items) - 2:
            formula += 'C(' + item + ')'
    return formula


def construct_interaction_effects_formula(items: List[str]) -> str:
    """Constrói fórmula de efeitos de interação"""
    formula = ''
    for x in range(len(items) - 1):
        for y in range(len(items) - 1):
            if items[x] != items[y]:
                partial_string_r = 'C(' + items[x] + '):C(' + items[y] + ')'
                partial_string_l = 'C(' + items[y] + '):C(' + items[x] + ')'
                if partial_string_r not in formula and partial_string_l not in formula:
                    if y < len(items) - 2:
                        formula += partial_string_r + '+'
    return formula


def combine_strings(string_splited: List[str], separator: str = ' + ') -> str:
    """Combina strings separadas"""
    combined_string = ""
    for x in range(len(string_splited) - 1):
        if x < len(string_splited) - 2:
            combined_string += string_splited[x] + separator
        elif x == len(string_splited) - 2:
            combined_string += string_splited[x]
    return combined_string


def calculate_mean_and_amplitude(df: pd.DataFrame, columns: List[str], 
                                 columns_x: List[str], x) -> Dict:
    """Calcula média e amplitude para análise nested"""
    grouped_items_first = {}
    grouped_items_last = {}
    r_bar = {}
    x_arr_col_rev_to_cut = columns_x.copy()
    del x_arr_col_rev_to_cut[-1]

    for v in range(len(x_arr_col_rev_to_cut)):
        first_group = df.groupby(x_arr_col_rev_to_cut).first().reset_index()
        last_group = df.groupby(x_arr_col_rev_to_cut).last().reset_index()
        grouped_items_first[v] = first_group
        grouped_items_last[v] = last_group
        amp = []
        mean = []

        for x in range(len(first_group)):
            lower_value = None
            greater_value = None
            data_to_mean = []
            len_t = 0
            
            if v == 0:
                for y in range(first_group['line'].iloc[x], last_group['line'].iloc[x] + 1):
                    data_to_mean.append(df[columns[-1]].iloc[y])
                    if greater_value is None or greater_value < df[columns[-1]].iloc[y]:
                        greater_value = df[columns[-1]].iloc[y]
                    if lower_value is None or lower_value > df[columns[-1]].iloc[y]:
                        lower_value = df[columns[-1]].iloc[y]
                amp.append(greater_value - lower_value)
                mean.append(sum(data_to_mean) / len(data_to_mean))
                len_t = len(df) // len(amp)
            else:
                len_t = len(grouped_items_first[v - 1]) // len(grouped_items_first[v])
                for z in range(x * len_t, ((x + 1) * len_t)):
                    if greater_value is None or greater_value < grouped_items_first[v - 1]['mean'].iloc[z]:
                        greater_value = grouped_items_first[v - 1]['mean'].iloc[z]
                    if lower_value is None or lower_value > grouped_items_first[v - 1]['mean'].iloc[z]:
                        lower_value = grouped_items_first[v - 1]['mean'].iloc[z]
                for y in range(first_group['line'].iloc[x], last_group['line'].iloc[x] + 1):
                    data_to_mean.append(df[columns[-1]].iloc[y])
                amp.append(greater_value - lower_value)
                mean.append(sum(data_to_mean) / len(data_to_mean))

        grouped_items_first[v]['mean'] = mean
        grouped_items_first[v]['amp'] = amp
        r_bar[v] = {'rBar': sum(amp) / len(amp), 'size': len_t}
        del x_arr_col_rev_to_cut[-1]

    lower_value = None
    greater_value = None
    for x in grouped_items_first[len(columns_x) - 2]['mean']:
        if greater_value is None or greater_value < x:
            greater_value = x
        if lower_value is None or lower_value > x:
            lower_value = x
    r_bar[len(columns_x) - 1] = {'rBar': greater_value - lower_value, 'size': 2}

    return r_bar


def calculate_variation_table(r_bar: Dict, x_arr_col: List[str]) -> Dict:
    """Calcula tabela de variação para análise nested"""
    result = {}
    total_variance = 0

    for x in r_bar.keys():
        if x == 0:
            r_bar_size = r_bar[x]['size']
            if r_bar_size < 2:
                r_bar_size = 2
            d2 = TABELA_SUBGROUP[r_bar_size]['d2']
            r_bar_c = r_bar[x]['rBar']
            variance = (r_bar_c / d2) ** 2
            desvpad = variance ** 0.5
            result[x] = {'variance': variance, 'desvpad': desvpad, 'percentage': 0.0}
        else:
            variance_before = 0.0
            size_c = 1
            for y in range(x, 0, -1):
                size_c = size_c * r_bar[y - 1]['size']
                variance_before += result[y - 1]['variance'] / size_c
            d2 = TABELA_SUBGROUP[r_bar[x]['size']]['d2']
            r_bar_c = r_bar[x]['rBar']
            variance = (r_bar_c / d2) ** 2 - variance_before
            desvpad = variance ** 0.5 if variance > 0 else 0
            result[x] = {'variance': variance, 'desvpad': desvpad, 'percentage': 0.0}

        if variance > 0:
            total_variance += variance

    x_arr_col_rev = x_arr_col.copy()
    x_arr_col_rev.reverse()

    for x in range(len(x_arr_col_rev)):
        if result[x]['variance'] > 0:
            result[x]['percentage'] = (result[x]['variance'] / total_variance) * 100
        result[x_arr_col_rev[x]] = result.pop(x)

    result['total'] = total_variance
    return result


def check_balanced(df: pd.DataFrame, analysis_type: str) -> bool:
    """Verifica se os dados estão balanceados"""
    if analysis_type == "crossed":
        df = df.iloc[:, :-1]
    elif analysis_type == "nested":
        df = df.iloc[:, :-2]
    
    unique_value_counts = df.apply(lambda col: col.nunique())

    if unique_value_counts.nunique() == 1:
        return True
    else:
        return False


def calculate_mean_square(sum_sq, df_values: pd.Series) -> List[float]:
    """Calcula quadrado médio"""
    return [sum_sq[x] / df_values[x] if df_values[x] != 0 else 0 for x in range(len(sum_sq))]


def calculate_percent_total(sum_sq, total_sum_sq: float) -> List[float]:
    """Calcula porcentagem do total"""
    values = [round(sum_sq[x] * 100 / total_sum_sq, 3) for x in range(len(sum_sq) - 1)]
    return values


def get_replace_label_crossed(label: str) -> str:
    """Formata labels para análise crossed"""
    return label.replace("C(", "").replace(")", "").replace(":", " * ")
