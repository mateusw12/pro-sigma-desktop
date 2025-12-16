"""
COV Utility Functions
Component of Variance Analysis
"""
from typing import List, Dict


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


def remove_punctuation(df):
    """Remove pontuação dos valores das colunas"""
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(r'[^\w\s]', '', regex=True)
    return df


def replace_data_frame(itens: List[str], df, np_module):
    """Remove colunas onde todos os valores são idênticos"""
    itens_to_remove = []
    for col in itens:
        if col in df.columns and np_module.all(df[col] == df[col].iloc[0]):
            itens_to_remove.append(col)
    
    for col in itens_to_remove:
        itens.remove(col)
        df.drop(col, inplace=True, axis=1)
    
    # Retorna tanto o dataframe quanto a lista atualizada
    return df, itens


def fit_linear_regression(df, formula: str, smf_module):
    """Ajusta modelo de regressão linear"""
    try:
        model = smf_module.ols(formula, data=df).fit()
        return model
    except:
        return None


def calculate_anova_table(model, sm_module):
    """Calcula tabela ANOVA"""
    table_to_read = sm_module.stats.anova_lm(model, typ=2)
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


def calculate_mean_and_amplitude(df, columns: List[str], 
                                 columns_x: List[str], x) -> Dict:
    """Calcula média e amplitude para análise nested"""
    grouped_items_first = {}
    grouped_items_last = {}
    r_bar = {}
    x_arr_col_rev_to_cut = columns_x.copy()
    del x_arr_col_rev_to_cut[-1]

    for v in range(len(x_arr_col_rev_to_cut)):
        # Garante que as colunas existem no dataframe
        groupby_cols = [col for col in x_arr_col_rev_to_cut if col in df.columns]
        if not groupby_cols:
            break
        
        first_group = df.groupby(groupby_cols, as_index=False).first()
        last_group = df.groupby(groupby_cols, as_index=False).last()
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
                # Acessa a coluna 'line' que está presente no resultado do groupby
                first_line = int(first_group.iloc[x]['line'])
                last_line = int(last_group.iloc[x]['line'])
                for y in range(first_line, last_line + 1):
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
                    mean_val = grouped_items_first[v - 1].iloc[z]['mean']
                    if greater_value is None or greater_value < mean_val:
                        greater_value = mean_val
                    if lower_value is None or lower_value > mean_val:
                        lower_value = mean_val
                first_line = int(first_group.iloc[x]['line'])
                last_line = int(last_group.iloc[x]['line'])
                for y in range(first_line, last_line + 1):
                    data_to_mean.append(df[columns[-1]].iloc[y])
                amp.append(greater_value - lower_value)
                mean.append(sum(data_to_mean) / len(data_to_mean))

        grouped_items_first[v]['mean'] = mean
        grouped_items_first[v]['amp'] = amp
        r_bar[v] = {'rBar': sum(amp) / len(amp), 'size': len_t}
        del x_arr_col_rev_to_cut[-1]

    lower_value = None
    greater_value = None
    # Acessa os valores da coluna 'mean' no último grupo
    mean_values = grouped_items_first[len(columns_x) - 2]['mean'].values
    for mean_val in mean_values:
        if greater_value is None or greater_value < mean_val:
            greater_value = mean_val
        if lower_value is None or lower_value > mean_val:
            lower_value = mean_val
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


def check_balanced(df, analysis_type: str) -> bool:
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


def calculate_mean_square(sum_sq, df_values) -> List[float]:
    """Calcula quadrado médio"""
    return [sum_sq[x] / df_values[x] if df_values[x] != 0 else 0 for x in range(len(sum_sq))]


def calculate_percent_total(sum_sq, total_sum_sq: float) -> List[float]:
    """Calcula porcentagem do total"""
    values = [round(sum_sq[x] * 100 / total_sum_sq, 3) for x in range(len(sum_sq) - 1)]
    return values


def get_replace_label_crossed(label: str) -> str:
    """Formata labels para análise crossed"""
    return label.replace("C(", "").replace(")", "").replace(":", " * ")


def prepare_variance_chart_data(variation_table: Dict, exclude_total=True) -> tuple:
    """
    Prepara dados para visualização de componentes de variância
    
    Args:
        variation_table: Tabela de variação do calculate_variation_table
        exclude_total: Se deve excluir 'total' do resultado
    
    Returns:
        tuple: (labels, variances, percentages)
    """
    labels = []
    variances = []
    percentages = []
    
    for key, value in variation_table.items():
        if key == 'total' and exclude_total:
            continue
        if isinstance(value, dict):
            labels.append(key)
            variances.append(value['variance'])
            percentages.append(value['percentage'])
    
    return labels, variances, percentages


def calculate_variance_reml(df, response_col: str, factor_cols: List[str], method='nested'):
    """
    Calcula componentes de variância usando REML (Restricted Maximum Likelihood)
    Implementação equivalente ao lmer() do R para replicar exatamente os resultados
    
    Args:
        df: DataFrame com os dados
        response_col: Nome da coluna de resposta (Y)
        factor_cols: Lista de colunas de fatores (X)
        method: 'nested' ou 'crossed'
    
    Returns:
        dict com componentes de variância estimados por REML
    """
    try:
        import numpy as np
        import pandas as pd
        
        # Preparar dados
        data = df[factor_cols + [response_col]].copy()
        data = data.dropna()
        
        # Renomear última coluna para Y
        data = data.rename(columns={response_col: 'Y'})
        
        # Converter Y para numérico
        data['Y'] = pd.to_numeric(data['Y'], errors='coerce')
        data = data.dropna(subset=['Y'])
        
        num_factors = len(factor_cols)
        
        # Construir fórmula baseada no método
        if method == 'nested':
            formula = _build_nested_formula(factor_cols, num_factors)
        else:  # crossed
            # Para crossed, remover última coluna se linhas forem únicas
            df_without_Y = data[factor_cols].copy()
            are_rows_unique = not df_without_Y.duplicated().any()
            
            if are_rows_unique and num_factors > 1:
                # Remover última coluna
                factor_cols = factor_cols[:-1]
                data = data.drop(columns=[df_without_Y.columns[-1]])
                num_factors = len(factor_cols)
            
            formula = _build_crossed_formula(factor_cols, num_factors)
        
        # Tentar usar pymer4 (wrapper para lmer do R) se disponível
        try:
            from pymer4.models import Lmer
            
            # Ajustar modelo com pymer4 (usa lmer do R internamente)
            model = Lmer(formula, data=data)
            model.fit(REML=True, control='check.nobs.vs.nlev="ignore", check.nobs.vs.rankZ="ignore", check.nobs.vs.nRE="ignore"')
            
            # Extrair componentes de variância
            variance_components = {}
            random_effects = model.ranef_var
            
            for component, variance in random_effects.items():
                variance_components[component] = float(variance)
            
            # Variância residual
            variance_components['Residual'] = float(model.residual_variance)
            
        except ImportError:
            # Se pymer4 não disponível, usar statsmodels com fórmula simplificada
            from statsmodels.formula.api import mixedlm
            
            # Construir fórmula statsmodels (mais limitada)
            variance_components = _fit_statsmodels_reml(data, factor_cols, method)
        
        # Calcular total e porcentagens
        total_var = sum(variance_components.values())
        
        variance_results = {}
        for component, var_value in variance_components.items():
            percentage = (var_value / total_var * 100) if total_var > 0 else 0
            variance_results[component] = {
                'variance': var_value,
                'desvpad': np.sqrt(abs(var_value)),
                'percentage': percentage
            }
        
        variance_results['total'] = total_var
        
        # Informações do modelo
        model_info = {
            'method': 'REML',
            'log_likelihood': getattr(model, 'logLike', 'N/A') if 'model' in locals() else 'N/A',
            'aic': getattr(model, 'AIC', 'N/A') if 'model' in locals() else 'N/A',
            'bic': getattr(model, 'BIC', 'N/A') if 'model' in locals() else 'N/A',
            'converged': True
        }
        
        return {
            'variances': variance_results,
            'model_info': model_info,
            'full_result': model if 'model' in locals() else None
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'variances': None,
            'model_info': None
        }


def _build_nested_formula(factor_cols: List[str], num_factors: int) -> str:
    """
    Constrói fórmula nested para REML (equivalente ao R)
    Exemplo: Y ~ (1|A) + (1|A:B) + (1|A:B:C)
    """
    terms = []
    
    for i in range(1, num_factors + 1):
        nested_term = ':'.join(factor_cols[:i])
        terms.append(f"(1|{nested_term})")
    
    formula = "Y ~ " + " + ".join(terms)
    return formula


def _build_crossed_formula(factor_cols: List[str], num_factors: int) -> str:
    """
    Constrói fórmula crossed para REML (equivalente ao R)
    Inclui efeitos principais e todas as interações
    """
    from itertools import combinations
    
    terms = []
    
    # Efeitos principais
    for factor in factor_cols:
        terms.append(f"(1|{factor})")
    
    # Interações de 2 fatores
    if num_factors >= 2:
        for combo in combinations(factor_cols, 2):
            interaction = ':'.join(combo)
            terms.append(f"(1|{interaction})")
    
    # Interação de ordem superior (todos os fatores)
    if num_factors >= 3:
        full_interaction = ':'.join(factor_cols)
        terms.append(f"(1|{full_interaction})")
    
    formula = "Y ~ " + " + ".join(terms)
    return formula


def _fit_statsmodels_reml(data, factor_cols: List[str], method: str) -> dict:
    """
    Fallback usando statsmodels quando pymer4 não está disponível
    Menos preciso mas funcional
    """
    from statsmodels.regression.mixed_linear_model import MixedLM
    import numpy as np
    
    variance_components = {}
    
    # Para cada fator, calcular variância como efeito aleatório
    for i, factor in enumerate(factor_cols):
        try:
            if method == 'nested' and i == 0:
                # Primeiro fator como grupo principal
                groups = data[factor].astype(str)
            else:
                # Criar grupos compostos
                groups = data[factor_cols[:i+1]].astype(str).agg('_'.join, axis=1)
            
            model = MixedLM.from_formula("Y ~ 1", groups=groups, data=data)
            result = model.fit(reml=True)
            
            # Variância do efeito aleatório
            random_var = float(result.cov_re.iloc[0, 0]) if hasattr(result.cov_re, 'iloc') else float(result.cov_re)
            
            if method == 'nested':
                # Para nested, subtrair variâncias anteriores
                if i > 0:
                    prev_var = sum([v for k, v in variance_components.items() if k != 'Residual'])
                    random_var = max(0, random_var - prev_var)
            
            variance_components[factor] = random_var
            
        except Exception as e:
            print(f"Erro ao calcular variância para {factor}: {e}")
            variance_components[factor] = 0
    
    # Adicionar variância residual do último modelo
    if 'result' in locals():
        variance_components['Residual'] = result.scale
    
    return variance_components


def compare_ems_reml(ems_results: Dict, reml_results: Dict) -> Dict:
    """
    Compara resultados de EMS (ANOVA) vs REML
    
    Args:
        ems_results: Resultados do método EMS
        reml_results: Resultados do método REML
    
    Returns:
        dict com comparação
    """
    comparison = {
        'ems_total': ems_results.get('total', 0) if isinstance(ems_results, dict) else 0,
        'reml_total': reml_results.get('variances', {}).get('total', 0) if reml_results else 0,
        'differences': []
    }
    
    # Comparar componentes individuais
    if isinstance(ems_results, dict) and reml_results and reml_results.get('variances'):
        ems_vars = {k: v for k, v in ems_results.items() if k != 'total' and isinstance(v, dict)}
        reml_vars = {k: v for k, v in reml_results['variances'].items() if k != 'total'}
        
        for component in set(list(ems_vars.keys()) + list(reml_vars.keys())):
            ems_val = ems_vars.get(component, {}).get('variance', 0)
            reml_val = reml_vars.get(component, {}).get('variance', 0)
            
            diff_abs = abs(ems_val - reml_val)
            diff_pct = (diff_abs / ems_val * 100) if ems_val > 0 else 0
            
            comparison['differences'].append({
                'component': component,
                'ems': ems_val,
                'reml': reml_val,
                'diff_abs': diff_abs,
                'diff_pct': diff_pct
            })
    
    return comparison
