# Análise de Regressão Linear Simples

## Descrição

Ferramenta para análise de regressão linear simples (uma variável independente X e uma variável dependente Y).

## Funcionalidades

### 1. Tabela de Resumo do Modelo
- **R²**: Coeficiente de determinação (proporção da variância explicada)
- **R² Ajustado**: R² ajustado pelo número de preditores
- **RMSE**: Raiz do erro quadrático médio
- **MAE**: Erro absoluto médio
- **Correlação**: Correlação de Pearson entre X e Y
- **Número de Observações**: Total de dados válidos utilizados
- **Média de X**: Média da variável independente
- **Média de Y**: Média da variável dependente

### 2. Tabela ANOVA
Análise de Variância da Regressão:
- **Fonte**: Regressão, Residual, Total
- **DF (Graus de Liberdade)**: Para regressão (1), residual (n-2), total (n-1)
- **SS (Soma dos Quadrados)**: Variação total, explicada e não explicada
- **MS (Quadrado Médio)**: SS/DF
- **F-Statistic**: Estatística F para teste de significância do modelo
- **P-Value**: Probabilidade associada ao teste F

### 3. Tabela de Coeficientes
Parâmetros da Regressão:
- **Termo**: Intercepto e coeficiente da variável X
- **Estimativa**: Valor estimado dos coeficientes (β₀ e β₁)
- **Erro Padrão**: Desvio padrão dos coeficientes
- **t-Ratio**: Estatística t para teste de significância
- **P-Value**: Probabilidade associada ao teste t
- **IC 95% Inferior**: Limite inferior do intervalo de confiança
- **IC 95% Superior**: Limite superior do intervalo de confiança
- **VIF**: Variance Inflation Factor (sempre 1.0 para regressão simples)

### 4. Gráfico de Regressão
- Scatter plot dos dados originais (X vs Y)
- Linha de regressão ajustada
- Bandas de intervalo de confiança de 95%
- Equação da regressão no título
- R² no título

### 5. Diagnóstico de Resíduos (4 painéis)
1. **Resíduos vs Valores Ajustados**: Verifica homocedasticidade
2. **Q-Q Plot**: Verifica normalidade dos resíduos
3. **Scale-Location**: Verifica homocedasticidade (resíduos padronizados)
4. **Resíduos vs X**: Detecta padrões não lineares

### 6. Histograma de Resíduos
- Distribuição dos resíduos
- Curva normal sobreposta
- Verifica normalidade visual dos resíduos

## Equação da Regressão

Y = β₀ + β₁ × X + ε

Onde:
- Y: Variável dependente (resposta)
- X: Variável independente (preditor)
- β₀: Intercepto (valor de Y quando X = 0)
- β₁: Coeficiente angular (mudança em Y por unidade de X)
- ε: Erro aleatório

## Método de Cálculo

### Mínimos Quadrados Ordinários (OLS)

Os coeficientes são estimados minimizando a soma dos quadrados dos resíduos:

**Fórmula Matricial:**
```
β = (X'X)⁻¹ X'y
```

**Fórmulas Diretas:**
```
β₁ = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / Σ[(xᵢ - x̄)²]
β₀ = ȳ - β₁ × x̄
```

### Erro Padrão dos Coeficientes
```
SE(β₁) = √[MSE / Σ(xᵢ - x̄)²]
SE(β₀) = √[MSE × (1/n + x̄² / Σ(xᵢ - x̄)²)]
```

Onde MSE = SS_residual / (n - 2)

### Estatística t
```
t = β / SE(β)
```

### R² (Coeficiente de Determinação)
```
R² = SS_regression / SS_total = 1 - (SS_residual / SS_total)
```

### R² Ajustado
```
R²_adj = 1 - [(1 - R²) × (n - 1) / (n - p - 1)]
```
Onde p = 1 (número de preditores)

### Estatística F
```
F = MS_regression / MS_residual
```

## Pressupostos da Regressão Linear

1. **Linearidade**: Relação linear entre X e Y
2. **Independência**: Observações independentes
3. **Homocedasticidade**: Variância constante dos resíduos
4. **Normalidade**: Resíduos seguem distribuição normal
5. **Ausência de outliers influentes**

## Interpretação

### Coeficiente Angular (β₁)
- **Positivo**: X e Y aumentam juntos
- **Negativo**: Quando X aumenta, Y diminui
- **Magnitude**: Mudança média em Y para cada unidade de X

### Intercepto (β₀)
- Valor previsto de Y quando X = 0
- Pode não ter interpretação prática se X = 0 não fizer sentido no contexto

### R²
- **0.0 - 0.3**: Correlação fraca
- **0.3 - 0.7**: Correlação moderada
- **0.7 - 1.0**: Correlação forte

### P-Value
- **< 0.001**: Altamente significativo (***)
- **< 0.01**: Muito significativo (**)
- **< 0.05**: Significativo (*)
- **≥ 0.05**: Não significativo

## Uso

1. Selecione a variável independente (X)
2. Selecione a variável dependente (Y)
3. Escolha opções de visualização (opcional)
4. Clique em "Executar Análise de Regressão"

## Requisitos de Dados

- Mínimo de 3 observações válidas (sem NaN)
- Ambas variáveis devem ser numéricas
- Recomendado: n > 30 para resultados mais confiáveis

## Exportação

- As tabelas podem ser copiadas diretamente
- Os gráficos podem ser salvos através do menu de contexto do matplotlib

## Referências

- Montgomery, D. C., Peck, E. A., & Vining, G. G. (2012). Introduction to Linear Regression Analysis.
- Kutner, M. H., Nachtsheim, C. J., Neter, J., & Li, W. (2005). Applied Linear Statistical Models.
