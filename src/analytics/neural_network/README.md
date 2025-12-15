# Módulo de Redes Neurais - MLP

## 📋 Descrição

Módulo de análise usando **Multi-Layer Perceptron (MLP)** para problemas de **Classificação** e **Regressão**. Implementado com **scikit-learn** e interface gráfica usando **customtkinter**.

## 🎯 Funcionalidades

### ✅ Tipos de Análise
- **Classificação**: Problemas com variável alvo categórica
- **Regressão**: Problemas com variável alvo contínua

### ✅ Métodos de Validação
1. **Holdout**: Divisão simples em treino/teste
2. **K-Fold Cross-Validation**: Validação cruzada com k partições

### ✅ Funcionalidades de Treinamento
- **GridSearchCV**: Otimização automática de hiperparâmetros
- **Múltiplas arquiteturas**: 8 configurações de camadas ocultas
- **Funções de ativação**: relu, tanh, logistic, identity
- **Solvers**: adam (otimizador adaptativo), sgd (gradiente estocástico)
- **Regularização**: Ajuste automático de alpha (L2)

### ✅ Métricas e Visualizações

#### Classificação
- Acurácia
- Precisão (Precision)
- Recall
- F1-Score
- ROC-AUC
- Matriz de Confusão
- Curva ROC

#### Regressão
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² (Coeficiente de Determinação)
- Média e Desvio Padrão

### ✅ Importância de Features
- **Permutation Importance**: Mede o impacto de cada variável nas predições

## 🏗️ Arquitetura do Código

```
src/analytics/neural_network/
├── __init__.py                  # Inicialização do módulo
├── neural_network_utils.py      # Backend: treinamento e métricas
└── neural_network_window.py     # Interface gráfica
```

### Arquivo: `neural_network_utils.py` (493 linhas)

#### Lazy Imports (9 funções)
```python
get_sklearn_neural_network()      # MLPClassifier, MLPRegressor
get_sklearn_preprocessing()       # OneHotEncoder, LabelEncoder
get_sklearn_model_selection()    # train_test_split, GridSearchCV, KFold
get_sklearn_metrics()             # accuracy, precision, recall, F1, etc.
get_sklearn_compose()             # ColumnTransformer
get_sklearn_inspection()          # permutation_importance
```

#### Funções Principais

**1. `is_categorical_target(y)`**
- Detecta se Y é categórico (classificação) ou contínuo (regressão)
- Critério: dtype object ou ≤10 valores únicos

**2. `encode_categorical_columns(df, categorical_cols)`**
- Converte colunas categóricas usando `LabelEncoder`

**3. `transform_features(X, categorical_cols)`**
- Aplica `OneHotEncoder` em variáveis categóricas via `ColumnTransformer`

**4. `calculate_metrics_classification(y_true, y_pred, y_pred_proba)`**
- Calcula todas as métricas de classificação
- Retorna: accuracy, precision, recall, F1, confusion_matrix, ROC-AUC, ROC curve

**5. `calculate_metrics_regression(y_true, y_pred)`**
- Calcula métricas de regressão
- Retorna: MSE, RMSE, R², mean, std

**6. `calculate_feature_importance(model, X, y)`**
- Calcula `permutation_importance` com n_repeats=10
- Retorna OrderedDict ordenado por importância

**7. `train_neural_network_holdout(...)`**
- Treinamento com divisão Holdout
- Parâmetros:
  - `df`: DataFrame com dados
  - `x_columns`: Colunas X
  - `y_column`: Coluna Y
  - `categorical_cols`: Colunas categóricas
  - `activation`: Função de ativação
  - `test_size`: Proporção de teste (0-1)
  - `max_iter`: Máximo de iterações
- GridSearchCV com 24 combinações:
  - `hidden_layer_sizes`: 8 arquiteturas [(5,), (10,), (15,), (5,3), (10,5), (15,10), (10,5,3), (15,10,5)]
  - `solver`: ['adam', 'sgd']
  - `learning_rate`: ['constant', 'adaptive']
  - `alpha`: [0.0001, 0.001, 0.01]
- Retorna: model, is_classification, metrics_train, metrics_test, feature_importance, model_info, predictions

**8. `train_neural_network_kfold(...)`**
- Treinamento com K-Fold Cross-Validation
- Similar ao Holdout mas com múltiplas partições
- Usa `StratifiedKFold` para classificação (preserva proporção de classes)
- Retorna métricas médias e desvio padrão

### Arquivo: `neural_network_window.py` (730 linhas)

#### Classe: `NeuralNetworkWindow(CTkToplevel)`

**Interface Gráfica:**
1. **Seleção de Variáveis**
   - Checkboxes para X (múltiplas seleções)
   - Radio buttons para Y (seleção única)
   
2. **Configurações**
   - Método: Holdout ou K-Fold
   - Função de ativação: relu, tanh, logistic, identity
   - Test Size (%): 10-90% (Holdout)
   - N Folds: 2-10 (K-Fold)
   - Max Iterações: 100-2000

3. **Resultados**
   - Informações do Modelo (arquitetura, iterações, loss)
   - Tabela de Métricas (treino vs teste ou média ± std)
   - Tabela de Importância de Features
   - Gráfico: Real vs Predito (linha)
   - Gráfico: Importância de Features (barras)
   - Gráfico: Matriz de Confusão (classificação)

## 🚀 Como Usar

### 1. Via Interface Gráfica

1. **Importe dados**: Excel ou CSV na página inicial
2. **Selecione "Redes Neurais"** no menu de ferramentas
3. **Configure**:
   - Marque variáveis X
   - Selecione variável Y
   - Escolha método (Holdout ou K-Fold)
   - Ajuste função de ativação
   - Configure parâmetros
4. **Clique em "🚀 Treinar Rede Neural"**
5. **Analise resultados**: métricas, gráficos, importância

### 2. Via Código Python

```python
from src.analytics.neural_network.neural_network_utils import (
    train_neural_network_holdout,
    train_neural_network_kfold
)
import pandas as pd

# Carrega dados
df = pd.read_excel('dados.xlsx')

# ===== HOLDOUT =====
results_holdout = train_neural_network_holdout(
    df=df,
    x_columns=['Feature1', 'Feature2', 'Feature3'],
    y_column='Target',
    categorical_cols=[],  # Se houver categóricas: ['Feature1']
    activation='relu',
    test_size=0.3,
    max_iter=500
)

print(f"R² Teste: {results_holdout['metrics_test']['r2']:.4f}")
print(f"Arquitetura: {results_holdout['model_info']['hidden_layers']}")

# ===== K-FOLD =====
results_kfold = train_neural_network_kfold(
    df=df,
    x_columns=['Feature1', 'Feature2', 'Feature3'],
    y_column='Target',
    categorical_cols=[],
    activation='relu',
    n_folds=5,
    max_iter=500
)

print(f"Acurácia: {results_kfold['metrics']['accuracy']:.4f} ± {results_kfold['metrics']['accuracy_std']:.4f}")
```

## 📊 Exemplo de Output

### Holdout - Regressão
```python
{
    'model': MLPRegressor(...),
    'is_classification': False,
    'metrics_train': {
        'mse': 0.1234,
        'rmse': 0.3512,
        'r2': 0.9567,
        'mean': 10.5,
        'std': 2.3
    },
    'metrics_test': {
        'mse': 0.1456,
        'rmse': 0.3815,
        'r2': 0.9432,
        'mean': 10.4,
        'std': 2.4
    },
    'feature_importance': {
        'Feature1': 0.234,
        'Feature2': 0.189,
        'Feature3': 0.067
    },
    'model_info': {
        'hidden_layers': (15, 10),
        'n_layers': 2,
        'n_iter': 145,
        'loss': 0.001234,
        'best_params': {
            'hidden_layer_sizes': (15, 10),
            'activation': 'relu',
            'solver': 'adam',
            'learning_rate': 'adaptive',
            'alpha': 0.001
        }
    },
    'y_test': [...],
    'y_pred_test': [...]
}
```

### K-Fold - Classificação
```python
{
    'model': MLPClassifier(...),
    'is_classification': True,
    'metrics': {
        'accuracy': 0.8756,
        'accuracy_std': 0.0345,
        'precision': 0.8623,
        'precision_std': 0.0412,
        'recall': 0.8901,
        'recall_std': 0.0298,
        'f1_score': 0.8759,
        'f1_score_std': 0.0356,
        'roc_auc': 0.9234,
        'roc_auc_std': 0.0267,
        'confusion_matrix': [[45, 5], [3, 47]]
    },
    'feature_importance': {...},
    'model_info': {
        'n_folds': 5,
        'hidden_layers': (10, 5),
        ...
    }
}
```

## 🔧 Hiperparâmetros Otimizados

### GridSearchCV - Espaço de Busca

| Parâmetro | Valores Testados |
|-----------|------------------|
| `hidden_layer_sizes` | (5,), (10,), (15,), (5,3), (10,5), (15,10), (10,5,3), (15,10,5) |
| `activation` | Definido pelo usuário (relu, tanh, logistic, identity) |
| `solver` | adam, sgd |
| `learning_rate` | constant, adaptive |
| `alpha` | 0.0001, 0.001, 0.01 |

**Total de combinações**: 8 × 1 × 2 × 2 × 3 = **96 modelos** testados

## ⚙️ Requisitos

### Bibliotecas Python
```txt
scikit-learn >= 1.0.0
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
customtkinter >= 5.0.0
```

### Instalação
```bash
pip install scikit-learn pandas numpy matplotlib customtkinter
```

## 🧪 Testes

Execute o script de teste:
```bash
python test_neural_network.py
```

Testes incluídos:
1. ✅ Importação dos utils
2. ✅ Importação da janela
3. ✅ Detecção de tipo (classificação vs regressão)
4. ✅ Treinamento Holdout (regressão)
5. ✅ Treinamento K-Fold (classificação)
6. ✅ Importância de features

## 📝 Notas Importantes

### Performance
- **GridSearchCV usa paralelização**: `n_jobs=-1` (todos os cores disponíveis)
- **Tempo de execução**: 10-60 segundos dependendo do dataset e hiperparâmetros
- **Recomendação**: Use K-Fold apenas com datasets médios/grandes (>100 amostras)

### Boas Práticas
1. **Normalização**: MLP é sensível à escala - considere normalizar features numéricas
2. **Variáveis categóricas**: Devem ser identificadas para encoding correto
3. **Max iterações**: Aumente se o modelo não convergir (padrão: 500)
4. **Função de ativação**:
   - `relu`: Padrão, funciona bem na maioria dos casos
   - `tanh`: Alternativa clássica
   - `logistic`: Para problemas suaves
   - `identity`: Modelo linear (baseline)

### Limitações
- **Dados pequenos**: MLP precisa de amostras suficientes (recomendado: >50 por variável)
- **Overfitting**: Use regularização (alpha) e validação cruzada
- **Interpretabilidade**: Use feature importance para insights

## 🎯 Integração

O módulo está integrado em:
- ✅ `home_page.py`: Menu "Redes Neurais" (Plano Pro)
- ✅ `license_manager.py`: Feature `'neural_networks'` no Plano Pro
- ✅ `lazy_imports.py`: Imports lazy de sklearn (otimização de memória)

## 📚 Referências

- [Scikit-learn MLP](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
- [GridSearchCV](https://scikit-learn.org/stable/modules/grid_search.html)
- [Permutation Importance](https://scikit-learn.org/stable/modules/permutation_importance.html)

## 👨‍💻 Autor

**ProSigma Development Team**
- Implementação: Assistente de IA (GitHub Copilot)
- Data: Maio 2025
- Versão: 1.0.0

---

**Status**: ✅ **IMPLEMENTADO E FUNCIONAL**
