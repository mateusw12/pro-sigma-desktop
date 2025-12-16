"""
Logistic Regression Utilities
Binary Logistic Regression with GLM
"""
import numpy as np
import pandas as pd
from typing import Dict, List


def calculate_logistic_regression(df: pd.DataFrame, response_col: str, 
                                  predictor_cols: List[str], 
                                  categorical_cols: List[str] = None) -> Dict:
    """
    Calcula regressão logística binária com link logit
    Equivalente ao glm(family=binomial(link="logit")) do R
    
    Args:
        df: DataFrame com os dados
        response_col: Nome da coluna resposta (binária ou categórica com 2 níveis)
        predictor_cols: Lista de colunas preditoras
        categorical_cols: Lista de colunas categóricas (para dummy encoding)
    
    Returns:
        Dict com todos os resultados da análise
    """
    import statsmodels.api as sm
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    
    # Preparar dados
    df_copy = df.copy()
    
    # Converter variável resposta para binária (0/1)
    y_original = df_copy[response_col].copy()
    unique_values = y_original.dropna().unique()
    
    if len(unique_values) != 2:
        raise ValueError(f"A variável resposta deve ter exatamente 2 valores únicos. Encontrados: {unique_values}")
    
    # Mapear para 0 e 1 (alfabeticamente: primeiro = 0, segundo = 1)
    sorted_values = sorted(unique_values, key=str)
    value_mapping = {sorted_values[0]: 0, sorted_values[1]: 1}
    y = y_original.map(value_mapping).astype(int)
    
    # Guardar mapeamento para referência
    reverse_mapping = {0: str(sorted_values[0]), 1: str(sorted_values[1])}
    
    # Identificar colunas categóricas
    if categorical_cols is None:
        categorical_cols = []
    
    # Criar dummies para variáveis categóricas
    X_data = df_copy[predictor_cols].copy()
    
    # Converter categóricas para dummies (excluindo primeira categoria como referência)
    categorical_dict = {}
    for col in categorical_cols:
        if col in X_data.columns:
            # Converter para string primeiro (para lidar com valores numéricos)
            X_data[col] = X_data[col].astype(str)
            # Agora converter para category
            X_data[col] = X_data[col].astype('category')
            # Criar dummies
            dummies = pd.get_dummies(X_data[col], prefix=col, drop_first=False)
            # Garantir que dummies são numéricas (0 e 1)
            dummies = dummies.astype(int)
            categorical_dict[col] = list(dummies.columns)
            X_data = X_data.drop(col, axis=1)
            X_data = pd.concat([X_data, dummies], axis=1)
    
    # Garantir que todas as colunas são numéricas
    X_data = X_data.apply(pd.to_numeric, errors='coerce')
    
    # Remover linhas com NaN (que podem ter sido criadas pela conversão)
    valid_indices = ~X_data.isna().any(axis=1)
    X_data = X_data[valid_indices]
    y = y[valid_indices]
    
    if len(X_data) < 10:
        raise ValueError("Dados insuficientes após conversão de variáveis categóricas")
    
    # Adicionar constante (intercepto)
    X = sm.add_constant(X_data)
    
    # Ajustar modelo logístico
    model = sm.Logit(y, X).fit(method='bfgs', maxiter=1000, disp=0)
    
    # Substituir coeficientes NA por zero
    coefficients = model.params.copy()
    coefficients = coefficients.fillna(0)
    
    # Previsões
    predictions_prob = model.predict(X)
    predictions_class = (predictions_prob > 0.5).astype(int)
    
    # Matriz de confusão
    conf_matrix = confusion_matrix(y, predictions_class)
    
    # Métricas de avaliação
    accuracy = accuracy_score(y, predictions_class)
    precision = precision_score(y, predictions_class, zero_division=0)
    recall = recall_score(y, predictions_class, zero_division=0)
    f1 = f1_score(y, predictions_class, zero_division=0)
    
    # Parameter Estimates
    param_estimates = pd.DataFrame({
        'term': model.params.index,
        'estimate': model.params.values,
        'stdError': model.bse.values,
        'zValue': model.tvalues.values,
        'prob': model.pvalues.values
    })
    param_estimates['logWorth'] = -np.log10(param_estimates['prob'].clip(lower=1e-300))
    
    # Adicionar níveis faltantes para categóricas (nível de referência)
    if categorical_cols:
        param_estimates = _add_missing_categorical_levels(
            param_estimates, categorical_dict, df_copy, categorical_cols
        )
    
    # Logit (linear predictor)
    logit_values = model.predict(X, linear=True)
    
    # Whole Model Test
    # Modelo reduzido (apenas intercepto)
    X_reduced = sm.add_constant(np.ones(len(y)))
    model_reduced = sm.Logit(y, X_reduced).fit(disp=0)
    
    # Likelihood Ratio Test
    ll_full = model.llf
    ll_reduced = model_reduced.llf
    ll_diff = ll_full - ll_reduced
    chi_square = 2 * ll_diff
    df_diff = len(model.params) - 1  # Diferença de graus de liberdade
    
    # Chi-square p-value
    from scipy.stats import chi2
    prob_chisq = 1 - chi2.cdf(chi_square, df_diff)
    
    # McFadden's R² (pseudo R²)
    r_square = 1 - (model.llf / model_reduced.llf)
    
    # AIC e BIC
    aic = model.aic
    num_params = len(model.params) - 1
    n = len(y)
    aic_c = aic + (2 * num_params * (num_params + 1)) / (n - num_params - 1)
    bic = model.bic
    
    # Métricas do modelo
    metrics = {
        'logLikelihoodDiff': float(ll_diff),
        'logLikelihoodReduced': float(ll_reduced),
        'logLikelihoodFull': float(ll_full),
        'chiSquare': float(chi_square),
        'probChisq': float(prob_chisq),
        'rsquare': float(r_square),
        'aic': float(aic_c),
        'bic': float(bic),
        'observations': int(n),
        'df': int(num_params),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    
    # Curva sigmoide
    logit_min = logit_values.min()
    logit_max = logit_values.max()
    logit_range = np.linspace(logit_min - 1, logit_max + 1, 200)
    sigmoid_y = 1 / (1 + np.exp(-logit_range))
    
    sigmoid_data = {
        'x': logit_range.tolist(),
        'y': sigmoid_y.tolist()
    }
    
    # Equações
    equations = _generate_equations(param_estimates, categorical_cols)
    
    # ANOVA (Likelihood Ratio Test por termo)
    anova_results = _calculate_anova(model, X, y)
    
    return {
        'parameterEstimates': param_estimates.to_dict('records'),
        'logit': logit_values.tolist(),
        'confusionMatrix': conf_matrix.tolist(),
        'metrics': metrics,
        'sigmoid': sigmoid_data,
        'equations': equations,
        'anovaTable': anova_results,
        'predictions': predictions_prob.tolist(),
        'predictedClass': predictions_class.tolist(),
        'categoricalDict': categorical_dict,
        'responseMapping': value_mapping,  # Mapeamento da resposta
        'reverseMappingResponse': reverse_mapping,  # Mapeamento reverso
        'model': model  # Para uso posterior se necessário
    }


def _add_missing_categorical_levels(param_estimates: pd.DataFrame, 
                                    categorical_dict: Dict,
                                    df: pd.DataFrame,
                                    categorical_cols: List[str]) -> pd.DataFrame:
    """
    Adiciona níveis de referência faltantes para variáveis categóricas
    (O nível omitido tem coeficiente = -soma dos outros)
    """
    missing_rows = []
    
    for col in categorical_cols:
        if col not in categorical_dict:
            continue
        
        # Níveis presentes no modelo
        dummy_cols = categorical_dict[col]
        present_levels = [c.replace(f"{col}_", "") for c in dummy_cols]
        
        # Todos os níveis nos dados
        all_levels = df[col].astype('category').cat.categories.tolist()
        
        # Nível faltante (referência)
        missing_levels = set(all_levels) - set(present_levels)
        
        for missing_level in missing_levels:
            # Coeficientes dos níveis presentes
            mask = param_estimates['term'].str.startswith(f"{col}_")
            estimates_present = param_estimates.loc[mask, 'estimate'].values
            
            # Coeficiente do nível faltante = -soma dos presentes
            estimate_missing = -np.sum(estimates_present)
            
            term_missing = f"{col}_{missing_level}"
            
            missing_rows.append({
                'term': term_missing,
                'estimate': estimate_missing,
                'stdError': np.nan,
                'zValue': np.nan,
                'prob': np.nan,
                'logWorth': np.nan
            })
    
    if missing_rows:
        missing_df = pd.DataFrame(missing_rows)
        param_estimates = pd.concat([param_estimates, missing_df], ignore_index=True)
        param_estimates = param_estimates.sort_values('term').reset_index(drop=True)
    
    return param_estimates


def _generate_equations(param_estimates: pd.DataFrame, 
                       categorical_cols: List[str]) -> Dict:
    """
    Gera equações de probabilidade
    """
    # Separar intercepto e termos
    intercept_row = param_estimates[param_estimates['term'] == 'const']
    intercept = intercept_row['estimate'].values[0] if len(intercept_row) > 0 else 0
    
    terms_df = param_estimates[param_estimates['term'] != 'const'].copy()
    
    # Equação principal
    terms_list = []
    for _, row in terms_df.iterrows():
        coef = row['estimate']
        var = row['term']
        
        # Verificar se é categórica
        is_categorical = any(var.startswith(f"{cat}_") for cat in categorical_cols)
        
        if is_categorical:
            # Para categórica, apenas o nome da variável base
            var_base = var.split('_')[0]
            if var_base not in [t.split('*')[0].strip() for t in terms_list]:
                terms_list.append(f"{var_base}")
        else:
            terms_list.append(f"({coef:.8f}) * {var}")
    
    equation_main = f"P(Y=1) = 1 / (1 + exp(-1 * ({intercept:.8f} + {' + '.join(terms_list)})))"
    
    # Equação de cálculo (logit)
    terms_calc = [f"({row['estimate']:.8f} * {row['term']})" 
                  for _, row in terms_df.iterrows()]
    equation_calc = f"({intercept:.8f} + {' + '.join(terms_calc)})"
    
    # Equações individuais por variável
    equations_individual = {}
    for _, row in terms_df.iterrows():
        var = row['term']
        coef = row['estimate']
        equations_individual[var] = f"P(Y=1|{var}) = 1 / (1 + exp(-1 * ({coef:.8f} * {var})))"
    
    return {
        'equationMain': equation_main,
        'equationCalc': equation_calc,
        'equations': equations_individual
    }


def _calculate_anova(model, X, y):
    """
    Calcula ANOVA (Likelihood Ratio Test) para cada termo
    """
    import statsmodels.api as sm
    from scipy.stats import chi2
    
    anova_results = []
    
    # Modelo completo
    ll_full = model.llf
    
    # Para cada preditor (exceto constante)
    predictors = [col for col in X.columns if col != 'const']
    
    for pred in predictors:
        # Modelo sem este preditor
        X_reduced = X.drop(pred, axis=1)
        try:
            model_reduced = sm.Logit(y, X_reduced).fit(disp=0)
            ll_reduced = model_reduced.llf
            
            # Likelihood ratio test
            chi_sq = 2 * (ll_full - ll_reduced)
            df = 1
            p_value = 1 - chi2.cdf(chi_sq, df)
            
            anova_results.append({
                'term': pred,
                'df': df,
                'deviance': float(chi_sq),
                'prob': float(p_value)
            })
        except:
            # Se o modelo não convergir, pular
            continue
    
    return anova_results


def interpret_logistic_results(metrics: Dict, param_estimates: List[Dict]) -> Dict:
    """
    Gera interpretações dos resultados
    """
    interpretations = []
    
    # Interpretação do modelo geral
    if metrics['probChisq'] < 0.001:
        interpretations.append("O modelo é altamente significativo (p < 0.001)")
    elif metrics['probChisq'] < 0.05:
        interpretations.append(f"O modelo é significativo (p = {metrics['probChisq']:.4f})")
    else:
        interpretations.append(f"O modelo não é significativo (p = {metrics['probChisq']:.4f})")
    
    # Interpretação do R²
    r2 = metrics['rsquare']
    if r2 > 0.5:
        interpretations.append(f"Excelente ajuste do modelo (R² = {r2:.4f})")
    elif r2 > 0.3:
        interpretations.append(f"Bom ajuste do modelo (R² = {r2:.4f})")
    elif r2 > 0.1:
        interpretations.append(f"Ajuste moderado do modelo (R² = {r2:.4f})")
    else:
        interpretations.append(f"Ajuste fraco do modelo (R² = {r2:.4f})")
    
    # Interpretação da acurácia
    acc = metrics['accuracy']
    if acc > 0.9:
        interpretations.append(f"Excelente acurácia de classificação ({acc*100:.1f}%)")
    elif acc > 0.8:
        interpretations.append(f"Boa acurácia de classificação ({acc*100:.1f}%)")
    elif acc > 0.7:
        interpretations.append(f"Acurácia moderada de classificação ({acc*100:.1f}%)")
    else:
        interpretations.append(f"Acurácia baixa de classificação ({acc*100:.1f}%)")
    
    # Preditores significativos
    significant_predictors = [p['term'] for p in param_estimates 
                            if p['term'] != 'const' and p.get('prob', 1) < 0.05]
    
    if significant_predictors:
        interpretations.append(f"Preditores significativos (p < 0.05): {', '.join(significant_predictors)}")
    else:
        interpretations.append("Nenhum preditor significativo encontrado")
    
    return {
        'interpretations': interpretations,
        'significantPredictors': significant_predictors
    }
