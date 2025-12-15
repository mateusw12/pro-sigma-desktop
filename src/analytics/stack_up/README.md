# Stack-Up Analysis (Empilhamento de Tolerâncias)

## Descrição

A ferramenta de **Stack-Up** (Empilhamento de Tolerâncias) permite analisar como as tolerâncias individuais de diferentes características se acumulam em um sistema, calculando a distribuição estatística do resultado final.

## Funcionalidades

### Entrada de Dados

1. **Configuração de Características**
   - Defina o número de características a analisar (1-50)
   - Para cada característica, configure:
     - **Nome**: Identificação da característica
     - **Valor Mínimo**: Limite inferior da tolerância
     - **Valor Máximo**: Limite superior da tolerância
     - **Sensibilidade**: Fator de influência na resposta final
     - **Quota**: Tipo de controle aplicado
       - Standard (1): Controle padrão
       - CTS (1.33): Critical to Schedule (crítico para cronograma)
       - CTQ (2): Critical to Quality (crítico para qualidade)

2. **Simulação Monte Carlo**
   - Defina o número de rodadas (100-250.000)
   - Maior número = maior precisão, mas mais tempo de processamento
   - Recomendado: 5.000 rodadas para análises gerais

3. **Importação/Exportação**
   - Baixe um modelo padrão Excel
   - Importe dados de arquivos Excel (.xlsx) ou CSV
   - Exporte resultados calculados

### Cálculos Realizados

A ferramenta calcula automaticamente:

1. **Média de cada característica**: (Máx + Mín) / 2
2. **Desvio padrão**: (Máx - Mín) / (6 × Quota)
3. **Distribuições normais** para cada característica
4. **Equação final**: Y = Σ(Característica × Sensibilidade)
5. **Distribuição do resultado** (Y) considerando todas as características

### Resultados

A ferramenta apresenta:

1. **Tabela de Resumo**
   - Média calculada para cada característica
   - Desvio padrão de cada característica

2. **Equação Final**
   - Fórmula completa mostrando como as características se combinam
   - Formato: Y = Char1 × Sens1 + Char2 × Sens2 + ...

3. **Tabela de Dados Completa**
   - Todas as rodadas simuladas
   - Valores gerados para cada característica
   - Resultado final (Y) para cada rodada

## Casos de Uso

### Exemplo 1: Montagem Mecânica

Análise de empilhamento de tolerâncias em uma montagem com 5 peças:

```
Característica A: Comprimento da Peça 1
- Mín: 99.8 mm, Máx: 100.2 mm
- Sensibilidade: 1
- Quota: Standard

Característica B: Comprimento da Peça 2
- Mín: 49.9 mm, Máx: 50.1 mm
- Sensibilidade: 1
- Quota: CTQ (crítica)

Resultado: Analisa o comprimento total da montagem
```

### Exemplo 2: Processo de Manufatura

Análise de múltiplas operações em série:

```
Operação 1: Corte (Sensibilidade: +1)
Operação 2: Usinagem (Sensibilidade: -0.5)
Operação 3: Acabamento (Sensibilidade: +0.3)

Resultado: Dimensão final da peça
```

## Interpretação dos Resultados

### Médias e Desvios Padrão

- **Média alta**: A característica contribui significativamente para o resultado
- **Desvio padrão alto**: Grande variabilidade na característica
- **Quota CTQ (2)**: Reduz o desvio padrão calculado, indicando controle mais rigoroso

### Equação Final

A equação mostra como cada característica influencia o resultado:
- **Sensibilidade positiva**: Aumenta o resultado
- **Sensibilidade negativa**: Diminui o resultado
- **Magnitude da sensibilidade**: Grau de influência

### Análise da Distribuição Y

Com os dados exportados, você pode:
1. Calcular a capacidade do processo (Cp, Cpk)
2. Estimar a probabilidade de defeitos
3. Identificar características críticas
4. Otimizar tolerâncias para reduzir custos

## Dicas de Uso

### Boas Práticas

1. **Comece com 5.000 rodadas** para análises iniciais
2. **Use 50.000+ rodadas** para análises finais críticas
3. **Valide a sensibilidade** de cada característica experimentalmente
4. **Documente as premissas** de cada análise

### Troubleshooting

**Problema**: Resultados não esperados
- ✓ Verifique se Min < Max para todas as características
- ✓ Confirme os valores de sensibilidade
- ✓ Valide a quota aplicada

**Problema**: Tempo de processamento longo
- ✓ Reduza o número de rodadas
- ✓ Diminua o número de características se possível

## Fundamentação Teórica

### Simulação Monte Carlo

O método Monte Carlo gera amostras aleatórias seguindo distribuições normais para cada característica, permitindo:
- Análise estatística robusta
- Consideração de interações complexas
- Previsão de comportamento do sistema

### Distribuição Normal

Cada característica é modelada como uma distribuição normal:
- μ (média) = (Max + Min) / 2
- σ (desvio padrão) = (Max - Min) / (6 × Quota)

A premissa é que a característica está dentro de ±3σ (99.73% dos valores).

### Propagação de Incertezas

A equação final propaga as incertezas individuais:
- **Soma de variáveis**: σ²_total = Σ(Sensibilidade² × σ²_característica)
- **Resultado**: Distribuição normal do sistema completo

## Plano Necessário

- **Intermediate** ou superior

## Referências

- ISO 1101: Tolerâncias geométricas
- ASME Y14.5: Dimensionamento e tolerância
- Montgomery, D.C. (2009). Introduction to Statistical Quality Control
