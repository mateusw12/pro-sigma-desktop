# Análise Multivariada

## Visão Geral

A ferramenta de Análise Multivariada permite examinar relações entre múltiplas variáveis simultaneamente através de matriz de correlação e visualizações avançadas.

## Características

### Cálculos Realizados

1. **Normalização de Dados**
   - Padronização: (X - média) / desvio padrão
   - Cálculo de médias por coluna
   - Soma dos quadrados das diferenças

2. **Matriz de Correlação**
   - Correlações entre todas as variáveis
   - Valores entre -1 e 1
   - Identificação de correlações fortes (|r| > 0.75)

3. **Análise Visual**
   - Heatmap de correlação com escala de cores
   - Scatter Plot Matrix (gráfico de dispersão matricial)
   - Linhas de tendência em cada scatter plot

### Funcionalidades

- **Seleção Automática**: Detecta e usa apenas colunas numéricas
- **Validação de Dados**: Verifica valores faltantes e quantidade de variáveis
- **Visualizações Interativas**: Alterna entre heatmap e scatter matrix
- **Cores Significativas**: Destaque para correlações fortes

## Como Usar

### Passo 1: Preparar Dados
1. Importe um arquivo Excel/CSV com múltiplas colunas numéricas
2. Mínimo: 2 variáveis numéricas
3. Máximo: 20 variáveis (para visualização clara)

### Passo 2: Abrir Ferramenta
1. No menu principal, clique em "Análise Multivariada" 📊
2. Selecione os dados importados
3. A ferramenta processa automaticamente

### Passo 3: Interpretar Resultados

#### Matriz de Correlação (Tabela)
- **Diagonal**: Sempre 1.0 (correlação de uma variável consigo mesma)
- **Valores positivos**: Relação direta (quando X aumenta, Y aumenta)
- **Valores negativos**: Relação inversa (quando X aumenta, Y diminui)
- **Cores**:
  - 🔴 **Vermelho**: Correlação negativa forte (≤ -0.75)
  - 🔵 **Azul**: Correlação positiva forte (≥ 0.75)

#### Heatmap de Correlação
- **Escala de Cores**: Do azul (negativo) ao vermelho (positivo)
- **Intensidade**: Quanto mais forte a cor, maior a correlação
- **Centro branco**: Correlação próxima de zero (sem relação)

#### Scatter Plot Matrix
- **Diagonal**: Histogramas mostrando distribuição de cada variável
- **Fora da diagonal**: Gráficos de dispersão entre pares de variáveis
- **Linha vermelha**: Linha de tendência (regressão linear)
- **Padrões**:
  - Pontos alinhados: Correlação forte
  - Pontos dispersos: Correlação fraca
  - Inclinação positiva: Correlação positiva
  - Inclinação negativa: Correlação negativa

## Interpretação de Correlações

### Força da Correlação (valores absolutos)

| Valor |r| | Interpretação |
|---------|---------------|
| 0.00 - 0.19 | Correlação muito fraca |
| 0.20 - 0.39 | Correlação fraca |
| 0.40 - 0.59 | Correlação moderada |
| 0.60 - 0.79 | Correlação forte |
| 0.80 - 1.00 | Correlação muito forte |

### Direção da Correlação

- **r > 0**: Correlação positiva (variáveis crescem juntas)
- **r < 0**: Correlação negativa (uma cresce, outra decresce)
- **r ≈ 0**: Sem correlação linear

## Exemplo de Uso

### Dados de Entrada
```
Temperatura | Vendas_Sorvete | Vendas_Café | Umidade
------------|----------------|-------------|--------
25          | 150            | 80          | 60
30          | 180            | 70          | 55
20          | 120            | 95          | 70
35          | 200            | 60          | 50
```

### Resultados Esperados
- **Temperatura x Vendas_Sorvete**: Correlação positiva forte (~0.95)
- **Temperatura x Vendas_Café**: Correlação negativa forte (~-0.92)
- **Temperatura x Umidade**: Correlação negativa moderada (~-0.65)

## Casos de Uso

### 1. Análise de Processo Industrial
- Identificar quais parâmetros de processo afetam a qualidade
- Exemplo: Temperatura, pressão, velocidade vs. defeitos

### 2. Análise de Mercado
- Relacionar variáveis de vendas, marketing e sazonalidade
- Exemplo: Investimento em marketing vs. receita

### 3. Controle de Qualidade
- Verificar correlações entre medições de diferentes características
- Exemplo: Dimensões de uma peça mecânica

### 4. Estudos de Confiabilidade
- Analisar fatores que afetam falhas ou vida útil
- Exemplo: Temperatura, uso, manutenção vs. tempo até falha

## Limitações

### Dados Requeridos
- **Mínimo**: 2 variáveis numéricas
- **Máximo**: 20 variáveis (para visualização legível)
- **Observações**: Recomendado mínimo de 30 linhas para correlações confiáveis

### Considerações Estatísticas
- ⚠️ **Correlação ≠ Causalidade**: Alta correlação não implica que uma variável causa a outra
- ⚠️ **Apenas Linear**: Detecta apenas relações lineares
- ⚠️ **Outliers**: Valores extremos podem distorcer resultados
- ⚠️ **Multicolinearidade**: Correlações muito altas (>0.95) entre X's podem causar problemas em regressões

## Método de Cálculo

### 1. Normalização
```
X_normalizado = (X - média(X)) / sqrt(Σ(X - média)²)
```

### 2. Matriz de Correlação
```
R = X'ᵀ × X'
```
Onde X' é a matriz de dados normalizados

### 3. Coeficiente de Correlação de Pearson
```
r = Σ((X - X̄)(Y - Ȳ)) / sqrt(Σ(X - X̄)² × Σ(Y - Ȳ)²)
```

## Tecnologias Utilizadas

- **Interface**: customtkinter
- **Cálculos**: numpy, pandas (lazy loading)
- **Visualizações**: 
  - matplotlib para gráficos
  - seaborn para heatmap
  - FigureCanvasTkAgg para integração Tkinter

## Integração

- **Categoria**: Ferramentas Avançadas (Pro)
- **Ícone**: 📊 (gráfico de barras)
- **Requer**: Importação de dados (Excel/CSV)
- **Nível**: Pro

## Validações Automáticas

✅ Verifica se há pelo menos 2 colunas numéricas
✅ Detecta e reporta valores faltantes (NaN)
✅ Limita a 20 variáveis para performance
✅ Seleciona automaticamente apenas colunas numéricas

## Próximas Melhorias

- [ ] Exportação da matriz de correlação para Excel
- [ ] Teste de significância estatística das correlações
- [ ] Análise de Componentes Principais (PCA)
- [ ] Análise de Cluster
- [ ] Análise Fatorial
- [ ] Correlação de Spearman (não-paramétrica)
- [ ] Detecção automática de outliers
- [ ] Correlações parciais

## Diferenças vs Regressão Múltipla

| Característica | Multivariada | Regressão Múltipla |
|----------------|--------------|-------------------|
| Objetivo | Explorar relações | Prever Y |
| Variável Y | Não requerida | Obrigatória |
| Variáveis X | Múltiplas | Múltiplas |
| Resultado | Matriz correlação | Equação preditiva |
| Uso | Análise exploratória | Modelagem preditiva |

## Referências

- Análise Multivariada de Dados
- Coeficiente de Correlação de Pearson
- Scatter Plot Matrix (SPLOM)
- Matriz de Correlação
- Análise Exploratória de Dados (EDA)
