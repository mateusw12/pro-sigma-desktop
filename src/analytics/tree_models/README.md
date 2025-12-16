# Modelos de Árvore (Tree Models)

Módulo de análise com três modelos baseados em árvore para classificação e regressão.

## Modelos Disponíveis

### 1. Decision Tree (Árvore de Decisão)
- **Tipo**: Árvore de decisão simples
- **Parâmetros**:
  - Profundidade máxima
  - Mínimo de amostras para split
  - Mínimo de amostras em folha
- **Grid Search**: Otimização automática de hiperparâmetros

### 2. Random Forest (Floresta Aleatória)
- **Tipo**: Ensemble de árvores de decisão
- **Parâmetros**:
  - Número de árvores (n_estimators)
  - Profundidade máxima
  - Mínimo de amostras em folha
- **Características**: Reduz overfitting através de bootstrap aggregating

### 3. Gradient Boosting
- **Tipo**: Ensemble sequencial de árvores
- **Parâmetros**:
  - Número de árvores (n_estimators)
  - Taxa de aprendizado (learning_rate)
  - Subsample
  - Profundidade máxima
- **Grid Search**: Otimização completa de hiperparâmetros
- **Características**: Alto desempenho através de boosting

## Funcionalidades

### Treinamento
- ✅ Múltiplas variáveis independentes (X)
- ✅ Uma variável dependente (Y)
- ✅ Suporte para classificação e regressão
- ✅ Encoding automático de variáveis categóricas
- ✅ Seleção manual de colunas categóricas
- ✅ Divisão treino/teste configurável

### Métricas

**Classificação:**
- Acurácia
- Precisão
- Recall
- F1-Score
- Matriz de Confusão
- ROC AUC (para classificação binária)

**Regressão:**
- R² Score
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)

### Análise
- 📊 Importância de Features
- 📈 Métricas de desempenho em treino e teste
- 💾 Salvar modelo treinado (.pkl + .json)
- 📂 Carregar modelo salvo
- 🔄 Predições com modelos carregados

## Arquitetura

### Arquivos
- `__init__.py`: Inicialização do módulo
- `tree_models_utils.py`: Backend com funções de treinamento (522 linhas)
- `tree_models_window.py`: Interface gráfica (755 linhas)

### Lazy Imports
Todos os imports pesados (pandas, numpy, sklearn) são feitos sob demanda para melhorar performance de inicialização.

## Uso

### Interface
1. Selecione variáveis X (múltiplas)
2. Selecione variável Y (uma)
3. Marque colunas categóricas
4. Escolha tipo de modelo
5. Configure parâmetros específicos
6. Clique em "Treinar Modelo"

### Salvamento de Modelo
- Gera arquivo `.pkl` com modelo completo
- Gera arquivo `.json` com metadados legíveis
- Inclui encoders e preprocessadores
- Timestamp e versão do ProSigma

### Carregamento de Modelo
- Valida compatibilidade de versão
- Verifica existência de colunas necessárias
- Aplica encoding automaticamente
- Calcula métricas se Y disponível

## Integração

### home_page.py
```python
'tree_models': {
    'title': 'Modelos de Árvore',
    'description': 'Decision Tree, Random Forest e Gradient Boosting',
    'plan': 'pro',
    'in_development': False
}
```

### license_manager.py
Feature disponível no plano **Pro**.

## Dependências

- scikit-learn: Modelos e métricas
- pandas: Manipulação de dados
- numpy: Operações numéricas
- customtkinter: Interface gráfica
- pickle: Serialização de modelos
- json: Metadados

## Versão
1.0.0 - Implementação inicial completa
