# Gage R&R - Measurement System Analysis

## 📊 Visão Geral

O Gage R&R (Repeatability and Reproducibility) é uma ferramenta estatística fundamental para Six Sigma que avalia a qualidade de um sistema de medição. Determina quanto da variação observada é devido ao sistema de medição versus a variação real das peças.

## 🎯 Objetivo

Avaliar se o sistema de medição é capaz de:
- **Repetibilidade**: Variação quando o mesmo operador mede a mesma peça múltiplas vezes (Equipment Variation - EV)
- **Reprodutibilidade**: Variação entre diferentes operadores medindo a mesma peça (Appraiser Variation - AV)
- Distinguir entre peças diferentes (discriminação)

## 📐 Método ANOVA

Esta implementação utiliza o método ANOVA (Analysis of Variance), que é mais preciso que o método Range e considera:
- Variação das peças
- Variação dos operadores
- Interação entre peça e operador
- Repetibilidade (variação do equipamento)

## 🔢 Componentes de Variância

### 1. Equipment Variation (EV) - Repetibilidade
- Variação devido ao equipamento de medição
- Representa a consistência do instrumento

### 2. Appraiser Variation (AV) - Reprodutibilidade
- Variação devido aos operadores
- Inclui:
  - Variação entre operadores
  - Interação Operador × Peça

### 3. Gage R&R
- **GRR = EV + AV**
- Total da variação do sistema de medição

### 4. Part Variation (PV)
- Variação real entre as peças
- O que realmente queremos medir

## 📊 Métricas de Avaliação

### % Study Variation (%SV)
Percentual de cada componente em relação à variação total:

```
%GRR = (6σ_GRR / 6σ_Total) × 100
```

**Interpretação:**
- **< 10%**: 🟢 Excelente - Sistema aceitável
- **10-30%**: 🟡 Marginal - Pode ser aceitável dependendo da aplicação
- **> 30%**: 🔴 Inaceitável - Sistema precisa melhorar

### % Tolerance (%Tol)
Percentual em relação à tolerância especificada (USL - LSL):

```
%Tol = (6σ_GRR / Tolerância) × 100
```

### Number of Distinct Categories (ndc)
Capacidade do sistema de discriminar entre peças:

```
ndc = √(2 × (Var_Parts / Var_GRR))
```

**Interpretação:**
- **≥ 5**: 🟢 Excelente - Sistema discrimina bem
- **2-4**: 🟡 Marginal - Discriminação limitada
- **< 2**: 🔴 Inaceitável - Sistema não discrimina adequadamente

## 📈 Gráficos de Controle

### 1. Range Chart por Operador
- Monitora a variação (range) de cada operador por peça
- Identifica operadores com alta variabilidade

### 2. Average Chart por Operador
- Compara médias entre operadores
- Identifica viés sistemático

### 3. X-bar Chart
- Gráfico de controle das médias por peça
- UCL/LCL baseados em A2 × R̄

### 4. R Chart
- Gráfico de controle dos ranges
- UCL/LCL baseados em D3 e D4

### 5. Components of Variation
- Visualização dos componentes: EV, AV, PV
- Comparação percentual

### 6. By Part Chart
- Distribuição de todas as medições por peça
- Scatter plot para visualizar variação

## 🔄 Estrutura dos Dados

### Formato Requerido

O arquivo deve conter:
- **Coluna de Peças**: Identificador único de cada peça (Part)
- **Coluna de Operadores**: Identificador de cada operador (Operator/Appraiser)
- **Colunas de Medições**: Múltiplas tentativas (Trial1, Trial2, Trial3...)

### Exemplo de Dados

```csv
Part,Operator,Trial1,Trial2,Trial3
A,Op1,10.2,10.3,10.1
A,Op2,10.1,10.2,10.3
A,Op3,10.0,10.1,10.2
B,Op1,15.5,15.6,15.4
B,Op2,15.4,15.5,15.6
B,Op3,15.3,15.4,15.5
...
```

### Requisitos Mínimos

- **Peças**: Mínimo 5-10 peças (idealmente 10)
- **Operadores**: Mínimo 2-3 operadores
- **Tentativas**: Mínimo 2-3 repetições por combinação Peça×Operador

## 📋 Tabela ANOVA

A análise gera uma tabela ANOVA completa:

| Source | DF | SS | MS | Var |
|--------|----|----|----|----|
| Parts | p-1 | SS_Parts | MS_Parts | σ²_Parts |
| Operators | o-1 | SS_Operators | MS_Operators | σ²_Operators |
| Part×Operator | (p-1)(o-1) | SS_Interaction | MS_Interaction | σ²_Interaction |
| Repeatability | po(r-1) | SS_Equipment | MS_Equipment | σ²_Equipment |
| Total | por-1 | SS_Total | - | σ²_Total |

Onde:
- p = número de peças
- o = número de operadores
- r = número de repetições

## 🎯 Como Usar

1. **Carregar Dados**
   - Arquivo CSV ou Excel com estrutura adequada

2. **Configurar**
   - Selecionar coluna de Peças
   - Selecionar coluna de Operadores
   - Adicionar colunas de medições (trials)
   - (Opcional) Informar tolerância

3. **Analisar**
   - Clique em "🔬 Analisar Gage R&R"
   - Aguarde processamento

4. **Interpretar Resultados**
   - Verifique %GRR (deve ser < 30%)
   - Analise NDC (deve ser ≥ 2)
   - Observe gráficos de controle
   - Identifique fontes de variação

5. **Exportar**
   - Salve relatório em JSON
   - Compartilhe resultados

## 🔍 Interpretação dos Resultados

### Gage R&R Aceitável (%GRR < 10%)
✅ Sistema de medição é adequado
✅ Pode ser usado para controle de processo
✅ Discrimina bem entre peças

### Gage R&R Marginal (10% < %GRR < 30%)
⚠️ Sistema pode ser aceitável para algumas aplicações
⚠️ Considere:
   - Criticidade da característica
   - Custo de melhoria
   - Alternativas disponíveis

### Gage R&R Inaceitável (%GRR > 30%)
❌ Sistema NÃO é adequado
❌ Ações necessárias:
   - Calibrar equipamento
   - Treinar operadores
   - Melhorar procedimento de medição
   - Substituir instrumento

## 🛠️ Melhorando o Sistema de Medição

### Alta Repetibilidade (EV)
- Calibrar equipamento
- Manutenção do instrumento
- Verificar fixação da peça
- Avaliar condições ambientais

### Alta Reprodutibilidade (AV)
- Treinar operadores
- Padronizar procedimento
- Melhorar instruções de trabalho
- Reduzir subjetividade

### Alta Interação Operador×Peça
- Revisar técnica de medição
- Simplificar procedimento
- Verificar ergonomia

## 📚 Referências

- AIAG MSA Manual (4th Edition)
- Montgomery, D.C. - Statistical Quality Control
- Wheeler, D.J. - EMP III: Evaluating the Measurement Process

## 💡 Dicas

1. **Seleção de Peças**: Escolha peças que representem toda a faixa de variação esperada
2. **Ordem Aleatória**: Randomize a ordem de medição para evitar viés
3. **Condições Controladas**: Mantenha condições ambientais estáveis
4. **Frequência**: Reavalie o sistema periodicamente
5. **Documentação**: Registre todas as condições do estudo

## 🎓 Aplicações Six Sigma

- **DMAIC - Measure**: Validar sistema de medição antes de coletar dados
- **Capability Studies**: Garantir que dados sejam confiáveis
- **Process Control**: Assegurar que variação detectada é real
- **Continuous Improvement**: Identificar oportunidades de melhoria no sistema de medição
