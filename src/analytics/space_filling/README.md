# Space Filling Design

## Visão Geral

A ferramenta de Space Filling Design permite criar experimentos eficientes usando técnicas de preenchimento de espaço e analisar os resultados através de regressão polinomial.

## Características

### Geração de Experimentos

**Métodos Disponíveis:**

1. **Latin Hypercube Sampling (LHS)**
   - Amostragem uniforme no espaço de design
   - Garante cobertura equilibrada de cada dimensão
   - Ideal para análises de sensibilidade

2. **LHS Minimin**
   - Minimiza a distância mínima entre pontos
   - Melhora a distribuição espacial
   - 5 iterações para otimização

3. **LHS Maximin**
   - Maximiza a distância média entre pontos
   - Melhor cobertura do espaço
   - 5 iterações para otimização

4. **Sphere Packing**
   - Distribui pontos como esferas em um espaço
   - Evita aglomeração de pontos
   - Ótimo para superfícies de resposta

### Análise de Resultados

- **Regressão Múltipla**: Ajusta modelo polinomial aos dados
- **ANOVA**: Testa significância estatística do modelo
- **Parameter Estimates**: Identifica fatores mais importantes
- **Gráficos de Diagnóstico**: Visualiza qualidade do ajuste
- **Múltiplas Respostas**: Análise separada para cada variável Y

## Como Usar

### Gerar Experimento

1. Clique em **"📋 Gerar Experimento"**
2. Configure os parâmetros:
   - **Tipo de Design**: LHS, LHS Min, LHS Max ou Sphere Packing
   - **Número de Fatores**: Quantas variáveis X (1-26)
   - **Número de Rodadas**: Quantidade de experimentos (recomendado: 10 × fatores)
   - **Colunas Y**: Número de variáveis de resposta

3. **(Opcional)** Marque "Gerar valores aleatórios para Y"
   - Útil para simular dados
   - Configure Min, Máx e Intervalo

4. Configure cada fator:
   - Selecione o fator (A, B, C...)
   - Nome da coluna (ex: "Temperatura")
   - Nível Mínimo (ex: -1 ou 50)
   - Nível Máximo (ex: 1 ou 100)
   - Clique **"➕ Adicionar"**

5. Clique **"📊 Gerar Experimento"**
6. Salve o arquivo Excel gerado

### Analisar Dados

1. **Importe** o arquivo Excel com dados experimentais
2. Selecione:
   - **Variáveis X**: Fatores do experimento (podem ser múltiplos)
   - **Variáveis Y**: Respostas medidas (podem ser múltiplas)
   - **(Opcional)** Adicione **Interações** (ex: X1*X2)

3. Marque **"Modelo Reduzido"** se quiser incluir termos de interação/quadráticos

4. Clique **"🔍 Calcular Análise"**

5. Visualize resultados em **tabs** (uma para cada Y):
   - Equação do modelo
   - Tabela ANOVA
   - Resumo do ajuste (R², RMSE)
   - Estimativas dos parâmetros
   - Gráficos (Overlay, Importância dos parâmetros)

## Interpretação de Resultados

### Equação do Modelo

```
Y = β₀ + β₁*(X₁ - X̄₁) + β₂*(X₂ - X̄₂) + ...
```

- **β₀**: Intercepto (valor médio de Y quando X está na média)
- **β₁, β₂...**: Coeficientes (impacto de cada fator)
- **Codificado**: Os valores X são centrados na média

### Tabela ANOVA

| Fonte | GL | SQ | MQ | F | Prob > F |
|-------|----|----|----|----|----------|
| Modelo | k | SS_model | MS_model | F | p-value |
| Erro | n-k-1 | SS_error | MS_error | - | - |
| Total | n-1 | SS_total | - | - | - |

- **Prob > F < 0.05**: Modelo é estatisticamente significativo ✅
- **Prob > F > 0.05**: Modelo não é significativo ⚠️

### Resumo do Ajuste

- **R²**: Proporção da variação explicada (0-1)
  - **> 0.90**: Excelente ajuste 🟢
  - **0.70-0.90**: Bom ajuste 🟡
  - **< 0.70**: Ajuste fraco 🔴

- **R² Ajustado**: R² penalizado pelo número de parâmetros
  - Prefira modelos com R² Ajustado maior

- **RMSE**: Erro médio das previsões (menor é melhor)
  - Compare com a faixa de Y para avaliar magnitude

### Estimativas dos Parâmetros

| Termo | Estimativa | Erro Padrão | t Ratio | Prob > \|t\| |
|-------|------------|-------------|---------|--------------|
| Intercept | 50.0 | 2.1 | 23.8 | < 0.0001 |
| X₁ | 12.5 | 1.8 | 6.9 | < 0.0001 |
| X₂ | -3.2 | 1.9 | -1.7 | 0.0945 |

- **Prob > |t| < 0.05**: Parâmetro é significativo ✅
- **Prob > |t| > 0.05**: Parâmetro não é significativo ⚠️
- **Estimativa positiva**: X aumenta, Y aumenta
- **Estimativa negativa**: X aumenta, Y diminui

### Gráficos

#### Overlay Plot (Y vs Y Predito)
- Mostra qualidade do ajuste
- Linhas próximas = bom ajuste
- Linhas distantes = ajuste ruim

#### Importância dos Parâmetros
- Barras horizontais com |Estimativa|
- Maiores barras = fatores mais importantes
- Ajuda na seleção de variáveis

## Casos de Uso

### 1. Otimização de Processo
- **Objetivo**: Encontrar configuração ótima de parâmetros
- **Exemplo**: Temperatura, Pressão, Tempo vs. Qualidade
- **Uso**: Gere LHS Maximin com 10-20 rodadas/fator

### 2. Análise de Sensibilidade
- **Objetivo**: Identificar fatores mais influentes
- **Exemplo**: Componentes de um produto vs. Custo
- **Uso**: Gere LHS básico, analise Parameter Estimates

### 3. Superfície de Resposta
- **Objetivo**: Mapear comportamento da resposta
- **Exemplo**: 2-3 fatores vs. Performance
- **Uso**: Gere Sphere Packing, adicione termos de interação

### 4. Screening de Variáveis
- **Objetivo**: Reduzir número de fatores em estudos futuros
- **Exemplo**: 10+ variáveis vs. Resultado
- **Uso**: LHS com menos rodadas, identifique não-significativos

## Dicas e Boas Práticas

### Planejamento do Experimento

✅ **Recomendações:**
- Use **10-20 rodadas por fator** (ex: 5 fatores = 50-100 rodadas)
- Para **screening inicial**: 5-10 rodadas por fator
- Para **otimização final**: 15-25 rodadas por fator
- **Sphere Packing** para superfícies complexas
- **LHS Maximin** para cobertura uniforme

⚠️ **Evite:**
- Menos de 5 rodadas por fator (modelo subajustado)
- Mais de 30 rodadas por fator (desperdício de recursos)
- Níveis min/max muito próximos (pouco efeito)

### Análise de Dados

✅ **Checklist:**
- [ ] Verificar se R² > 0.70
- [ ] Confirmar Prob > F < 0.05 (ANOVA)
- [ ] Remover parâmetros não-significativos (Prob > 0.10)
- [ ] Verificar overlay plot (boa aderência)
- [ ] Interpretar sinal dos coeficientes (físico/lógico)

### Interações

- **X1*X2**: Efeito de X1 depende do valor de X2
- **X1²**: Efeito quadrático (curvatura)
- Adicione interações se:
  - R² baixo no modelo linear
  - Conhecimento do processo sugere interação
  - Overlay plot mostra padrões não-lineares

## Limitações

### Técnicas

- ⚠️ **Apenas modelos lineares/polinomiais**
  - Não detecta relações não-lineares complexas
  - Considere redes neurais para relações complexas

- ⚠️ **Assume normalidade dos resíduos**
  - Verificar com testes de normalidade
  - Transformar Y se necessário (log, sqrt)

- ⚠️ **Multicolinearidade**
  - Fatores muito correlacionados causam problemas
  - Use análise multivariada para detectar

### Requisitos de Dados

- **Mínimo**: n > k + 1 (rodadas > fatores + 1)
- **Recomendado**: n > 5k (rodadas > 5 × fatores)
- Todas as variáveis devem ser numéricas
- Sem valores faltantes (NaN)

## Equações e Métodos

### Latin Hypercube Sampling

```
X_ij ~ Uniform[0, 1] com estratificação em k intervalos
```

Cada dimensão é dividida em n intervalos iguais, garantindo uma amostra por intervalo.

### Sphere Packing

```
d(x_i, x_j) ≥ r ∀ i ≠ j
```

Pontos são posicionados como esferas que não se sobrepõem, maximizando distância mínima.

### Regressão Linear Múltipla

```
Y = Xβ + ε

β = (X'X)⁻¹X'Y

R² = SS_model / SS_total

RMSE = √(SS_error / (n - k - 1))
```

## Tecnologias Utilizadas

- **Interface**: customtkinter
- **Cálculos**: numpy, scipy
- **DOE**: pyDOE2
- **Visualizações**: matplotlib
- **Análise**: pandas (lazy loading)

## Integração

- **Categoria**: Ferramentas Avançadas (Pro)
- **Ícone**: ⬜ (quadrado)
- **Requer**: Importação de dados (Excel/CSV) para análise
- **Geração**: Não requer dados (gera Excel)
- **Nível**: Pro

## Diferenças vs Outras Ferramentas

| Característica | Space Filling | Regressão Múltipla | DOE Fatorial |
|----------------|---------------|---------------------|--------------|
| Planejamento | Automático | Manual | Fatorial completo |
| Tipo de pontos | Contínuos | Quaisquer | Níveis discretos |
| Rodadas | n = 10k | Qualquer | 2^k ou 3^k |
| Objetivo | Exploração | Predição | Efeitos principais |
| Interações | Opcional | Sim | Automático |

## Próximas Melhorias

- [ ] Análise de resíduos (normalidade, homocedasticidade)
- [ ] Gráficos de superfície de resposta 3D
- [ ] Otimização numérica (encontrar máximo/mínimo)
- [ ] Validação cruzada (k-fold)
- [ ] Exportar equação para Excel
- [ ] Space Filling adaptativo (adicionar pontos)
- [ ] Comparison de diferentes designs

## Referências

- McKay, M. D., Beckman, R. J., & Conover, W. J. (1979). "A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code"
- Latin Hypercube Sampling (LHS)
- Design of Experiments (DOE)
- Response Surface Methodology (RSM)
- pyDOE2 Documentation
