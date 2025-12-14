# Análise de Regressão Linear Múltipla

## Descrição

Ferramenta avançada para análise de regressão linear múltipla com suporte para:
- Múltiplas variáveis independentes (X)
- Múltiplas variáveis dependentes (Y) - análise separada para cada
- Termos de interação entre variáveis X
- Seleção de modelo (completo ou reduzido)

## Funcionalidades

### 1. Seleção de Variáveis

#### Variáveis Independentes (X)
- Selecione múltiplas variáveis X via checkboxes
- Mínimo: 1 variável X
- Máximo: limitado apenas pelos dados disponíveis

#### Termos de Interação
- Sistema automático de geração de interações
- Quando 2+ variáveis X são selecionadas, todas as combinações possíveis aparecem
- Exemplo: X₁, X₂, X₃ → pode incluir X₁×X₂, X₁×X₃, X₂×X₃
- Selecione quais interações incluir no modelo

#### Variáveis Dependentes (Y)
- Selecione múltiplas variáveis Y
- **Uma análise completa é gerada para cada Y**
- Útil para analisar múltiplas respostas com os mesmos preditores

### 2. Tipos de Modelo

#### Modelo Completo
- Inclui todas as variáveis X selecionadas
- Inclui todos os termos de interação selecionados
- Nenhuma variável é removida automaticamente

#### Modelo Reduzido (Backward Elimination)
- Começa com todas as variáveis
- Remove iterativamente variáveis não significativas (p > 0.05)
- Para quando todas as variáveis remanescentes são significativas
- Mostra quais variáveis foram removidas

### 3. Tabelas de Resultados

#### Tabela de Resumo do Modelo
- **R²**: Proporção da variância explicada
- **R² Ajustado**: R² ajustado pelo número de preditores
- **RMSE**: Raiz do erro quadrático médio
- **MAE**: Erro absoluto médio
- **AIC**: Critério de Informação de Akaike (menor é melhor)
- **BIC**: Critério de Informação Bayesiano (menor é melhor)
- **Nº Observações**: Total de dados válidos
- **Nº Preditores**: Quantidade de variáveis X (excluindo intercepto)
- **Média de Y**: Valor médio da variável resposta

#### Tabela ANOVA
- **Fonte**: Regressão, Residual, Total
- **DF**: Graus de liberdade
- **SS**: Soma dos quadrados
- **MS**: Quadrado médio (SS/DF)
- **F-Statistic**: Teste de significância global do modelo
- **P-Value**: Significância do modelo (vermelho se < 0.05)

#### Tabela de Coeficientes
- **Termo**: Nome da variável ou interação
- **Estimativa**: Valor do coeficiente (β)
- **Erro Padrão**: Desvio padrão do coeficiente
- **t-Ratio**: Estatística t para teste de significância
- **P-Value**: Significância individual (vermelho se < 0.05)
- **IC 95% Inferior/Superior**: Intervalo de confiança
- **VIF**: Variance Inflation Factor (diagnóstico de multicolinearidade)

### 4. Gráficos

#### Valores Preditos vs Reais
- Scatter plot: valores reais (eixo X) vs preditos (eixo Y)
- Linha vermelha tracejada: predição perfeita (Y = X)
- Quanto mais próximos da linha, melhor o modelo
- R² mostrado no título

#### Diagnóstico de Resíduos (4 painéis)
1. **Resíduos vs Valores Ajustados**
   - Verifica homocedasticidade (variância constante)
   - Deve parecer aleatório em torno de zero
   - Padrões indicam problemas

2. **Q-Q Plot Normal**
   - Verifica normalidade dos resíduos
   - Pontos devem seguir linha diagonal
   - Desvios nas pontas indicam caudas pesadas/leves

3. **Scale-Location**
   - Verifica homocedasticidade (outra forma)
   - √|resíduos padronizados| vs valores ajustados
   - Linha horizontal indica variância constante

4. **Resíduos vs Ordem**
   - Detecta padrões temporais ou de coleta
   - Deve parecer aleatório
   - Tendências indicam autocorrelação

#### Histograma de Resíduos
- Distribuição dos resíduos
- Curva normal sobreposta
- Deve aproximar-se de uma distribuição normal

## Equação da Regressão Múltipla

### Sem Interações
Y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ + ε

### Com Interações
Y = β₀ + β₁X₁ + β₂X₂ + β₁₂(X₁×X₂) + ... + ε

Onde:
- Y: Variável dependente
- X₁, X₂, ..., Xₚ: Variáveis independentes
- β₀: Intercepto
- β₁, β₂, ..., βₚ: Coeficientes das variáveis
- β₁₂: Coeficiente do termo de interação
- ε: Erro aleatório

## Interpretação

### Coeficientes (βᵢ)
- **Sem interações**: Mudança em Y quando Xᵢ aumenta 1 unidade, mantendo outras variáveis constantes
- **Com interações**: Interpretação depende dos valores de outras variáveis

### VIF (Variance Inflation Factor)
- **VIF < 5**: ✓ Multicolinearidade baixa (ok)
- **VIF 5-10**: ⚠ Multicolinearidade moderada (atenção)
- **VIF > 10**: ✗ Multicolinearidade alta (considere remover variável)

Alto VIF significa que a variável é altamente correlacionada com outras, tornando difícil isolar seu efeito.

### R² vs R² Ajustado
- **R²**: Sempre aumenta ao adicionar variáveis (pode superestimar)
- **R² Ajustado**: Penaliza modelos com muitas variáveis (mais confiável)
- Use R² ajustado para comparar modelos com diferentes números de preditores

### AIC e BIC
- Critérios para seleção de modelo
- **Menor é melhor**
- BIC penaliza mais modelos complexos que AIC
- Use para comparar modelo completo vs reduzido

### P-Value do Modelo (ANOVA)
- Testa H₀: todos os coeficientes = 0 (modelo inútil)
- p < 0.05: pelo menos uma variável X é significativa

### P-Value dos Coeficientes
- Testa H₀: coeficiente específico = 0 (variável não tem efeito)
- p < 0.05: variável é significativa (destacada em vermelho)

## Pressupostos da Regressão Múltipla

1. **Linearidade**: Y é combinação linear das variáveis X
2. **Independência**: Observações são independentes
3. **Homocedasticidade**: Variância dos resíduos é constante
4. **Normalidade**: Resíduos seguem distribuição normal
5. **Sem multicolinearidade**: Variáveis X não são altamente correlacionadas
6. **Sem outliers influentes**: Pontos extremos não distorcem o modelo

## Workflow Recomendado

1. **Selecione variáveis X** relevantes baseado na teoria/conhecimento do domínio
2. **Selecione variáveis Y** que deseja modelar
3. **Adicione interações** se houver razão para crer que X's interagem
4. **Execute modelo completo** primeiro
5. **Verifique VIF** para detectar multicolinearidade
   - Se VIF > 10, considere remover variável ou usar modelo reduzido
6. **Analise p-values** dos coeficientes
   - Variáveis não significativas podem ser removidas
7. **Execute modelo reduzido** para comparação
8. **Compare AIC/BIC** entre modelos
9. **Verifique pressupostos** através dos gráficos de resíduos
10. **Interprete coeficientes** significativos no contexto do problema

## Exemplo de Uso

### Cenário: Prever Consumo de Combustível

**Variáveis X:**
- Peso do veículo (kg)
- Potência do motor (hp)
- Velocidade máxima (km/h)

**Interações:**
- Peso × Potência (veículos pesados com motores potentes)

**Variável Y:**
- Consumo (km/L)

**Modelo Completo:**
```
Consumo = β₀ + β₁(Peso) + β₂(Potência) + β₃(Velocidade) + β₄(Peso×Potência)
```

**Interpretação:**
- β₁ < 0: Veículos mais pesados consomem mais
- β₂ < 0: Motores mais potentes consomem mais
- β₄: Efeito conjunto de peso e potência

## Requisitos de Dados

- **Mínimo**: 1 variável X, 1 variável Y
- **Observações**: n > p + 2 (onde p = número de preditores)
- **Recomendado**: n > 10p para resultados confiáveis
- Todas as variáveis devem ser numéricas
- Dados com NaN são automaticamente removidos

## Limitações

- Apenas relações lineares são modeladas
- Não detecta relações não lineares automaticamente
- Sensível a outliers
- Requer pressupostos serem atendidos para inferência válida
- Alto número de preditores pode levar a overfitting

## Diferenças: Regressão Simples vs Múltipla

| Aspecto | Simples | Múltipla |
|---------|---------|----------|
| Nº Variáveis X | 1 | ≥1 |
| VIF | Sempre 1 | Calculado |
| Interações | Não aplicável | Disponível |
| Modelo Reduzido | Não aplicável | Backward elimination |
| Múltiplos Y | Não | Sim |
| Interpretação | Direta | Controle de confundidores |

## Referências

- Montgomery, D. C., Peck, E. A., & Vining, G. G. (2012). Introduction to Linear Regression Analysis.
- Kutner, M. H., et al. (2005). Applied Linear Statistical Models.
- James, G., et al. (2013). An Introduction to Statistical Learning.
