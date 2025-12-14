# Análise de Regressão Linear Múltipla

## Descrição

Ferramenta avançada para análise de regressão linear múltipla com suporte para:
- Múltiplas variáveis independentes (X)
- Múltiplas variáveis dependentes (Y) - análise separada para cada
- Termos de interação entre variáveis X
- Seleção de modelo (completo ou reduzido)

## Funcionalidades

### 1. Seleção de Variáveis

#### Variáveis Independentes Numéricas (X)
- Selecione múltiplas variáveis X numéricas via checkboxes
- Mínimo: 1 variável X (numérica ou categórica)
- Máximo: limitado apenas pelos dados disponíveis

#### Variáveis Independentes Categóricas (X)
- **Suporte completo para variáveis categóricas!**
- Selecione variáveis categóricas (texto, categorias) como preditores
- Sistema automático de **codificação dummy (one-hot encoding)**
- Para k categorias, cria k-1 variáveis dummy
- Primeira categoria ordenada alfabeticamente = **categoria de referência**
- Exemplo: Gênero [Masculino, Feminino] → 1 dummy: Gênero[Masculino]
  - Referência: Feminino (valor 0 em todas dummies)
  - Dummy: Gênero[Masculino] = 1 se Masculino, 0 se Feminino
- Interface mostra número de categorias entre parênteses
- Informações de codificação são exibidas nos resultados

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

### Sem Interações (Apenas Numéricas)
Y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ + ε

### Com Interações
Y = β₀ + β₁X₁ + β₂X₂ + β₁₂(X₁×X₂) + ... + ε

### Com Variáveis Categóricas (Dummy Variables)
Y = β₀ + β₁X₁ + β₂D₁ + β₃D₂ + ... + ε

Onde:
- Y: Variável dependente (numérica)
- X₁, X₂, ..., Xₚ: Variáveis independentes numéricas
- D₁, D₂, ...: Variáveis dummy (0 ou 1) representando categorias
- β₀: Intercepto (valor médio quando todas X=0 e categoria = referência)
- β₁, β₂, ..., βₚ: Coeficientes das variáveis numéricas
- β para dummies: Diferença em relação à categoria de referência
- β₁₂: Coeficiente do termo de interação
- ε: Erro aleatório

## Codificação de Variáveis Categóricas (Dummy Encoding)

### Conceito
Variáveis categóricas não podem ser usadas diretamente em regressão linear. São convertidas em variáveis dummy (0 ou 1).

### Método: k-1 Encoding
Para uma variável com k categorias, criamos k-1 variáveis dummy:
- **1 categoria = referência** (baseline): todas dummies = 0
- **k-1 categorias restantes**: cada uma com sua dummy

### Exemplo Prático

**Variável Original: Turno (Manhã, Tarde, Noite)**

Codificação:
```
Turno[Manhã]  = 0  (referência - não precisa de dummy)
Turno[Tarde]  = variável dummy 1
Turno[Noite]  = variável dummy 2
```

Tabela de conversão:
| Turno Original | Turno[Tarde] | Turno[Noite] |
|---------------|--------------|--------------|
| Manhã         | 0            | 0            |
| Tarde         | 1            | 0            |
| Noite         | 0            | 1            |

**Na Equação:**
```
Y = β₀ + β₁(Turno[Tarde]) + β₂(Turno[Noite]) + ...
```

**Interpretação dos Coeficientes:**
- β₀: Valor médio de Y quando Turno = Manhã (referência)
- β₁: Diferença média de Y entre Tarde e Manhã
- β₂: Diferença média de Y entre Noite e Manhã

**Exemplo Numérico:**
```
Produtividade = 80 + 5(Turno[Tarde]) - 3(Turno[Noite])
```

- Manhã: 80 + 5(0) - 3(0) = 80 unidades
- Tarde: 80 + 5(1) - 3(0) = 85 unidades (+5 vs Manhã)
- Noite: 80 + 5(0) - 3(1) = 77 unidades (-3 vs Manhã)

### Por Que k-1 Dummies?

Usar k dummies causa **multicolinearidade perfeita**:
- Se soubermos valores de k-1 dummies, podemos deduzir a k-ésima
- Matriz X'X se torna singular (não inversível)
- Regressão falha

Exemplo ruim (k=3, usar 3 dummies):
```
Se Turno[Manhã]=0, Turno[Tarde]=0 → Turno[Noite]=1 (sempre!)
```

Uma dummy é redundante, por isso usamos k-1.

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

### Cenário 1: Prever Consumo de Combustível (Variáveis Numéricas)

**Variáveis X Numéricas:**
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

### Cenário 2: Prever Salário (Com Variáveis Categóricas)

**Variáveis X Numéricas:**
- Anos de experiência
- Horas trabalhadas/semana

**Variáveis X Categóricas:**
- Departamento [Vendas, TI, RH, Operações]
- Nível [Júnior, Pleno, Sênior]

**Variável Y:**
- Salário (R$)

**Codificação Automática:**
```
Departamento → 3 dummies (referência: Operações)
  - Departamento[RH] = 1 se RH, 0 caso contrário
  - Departamento[TI] = 1 se TI, 0 caso contrário
  - Departamento[Vendas] = 1 se Vendas, 0 caso contrário

Nível → 2 dummies (referência: Júnior)
  - Nível[Pleno] = 1 se Pleno, 0 caso contrário
  - Nível[Sênior] = 1 se Sênior, 0 caso contrário
```

**Modelo Expandido:**
```
Salário = β₀ + β₁(Experiência) + β₂(Horas) 
         + β₃(Dept[RH]) + β₄(Dept[TI]) + β₅(Dept[Vendas])
         + β₆(Nível[Pleno]) + β₇(Nível[Sênior])
```

**Interpretação:**
- β₁: Aumento de salário por ano adicional de experiência
- β₂: Aumento de salário por hora adicional trabalhada
- β₃: Diferença salarial entre RH e Operações (referência)
- β₄: Diferença salarial entre TI e Operações
- β₅: Diferença salarial entre Vendas e Operações
- β₆: Diferença salarial entre Pleno e Júnior (referência)
- β₇: Diferença salarial entre Sênior e Júnior

## Requisitos de Dados

- **Mínimo**: 1 variável X (numérica ou categórica), 1 variável Y (numérica)
- **Observações**: n > p + 2 (onde p = número de preditores totais após encoding)
- **Recomendado**: n > 10p para resultados confiáveis
- **Variáveis Y**: Devem ser numéricas
- **Variáveis X Numéricas**: Quantidades, medições contínuas
- **Variáveis X Categóricas**: Texto, categorias (automaticamente codificadas)
- Dados com NaN são automaticamente removidos
- Categorias com apenas 1 valor único são ignoradas

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
