# Análise de Capacidade de Processo - Guia de Uso

## 📊 Visão Geral

A ferramenta de **Process Capability** permite avaliar a capacidade de um processo em atender especificações definidas, calculando índices como Cp, Cpk, Pp e Ppk.

## 🚀 Como Usar

### 1. Importar Dados
- Na página inicial, clique em **"Importar Excel"** ou **"Importar CSV"**
- Selecione um arquivo com os dados do processo
- Os dados devem conter pelo menos uma coluna numérica com as medições

### 2. Abrir a Ferramenta
- Após importar os dados, localize o card **"Process Capability"** (ícone 📈)
- Clique no botão **"Abrir →"**

### 3. Configurar a Análise

#### Seleção de Colunas:
- **Coluna X (Fase/Grupo)**: Opcional. Use para dividir a análise por grupos (ex: Máquina A, Máquina B)
- **Coluna Y (Resposta)**: Obrigatório. A coluna com os dados numéricos a serem analisados

#### Tipo de Análise:
- **Distribuição dos Dados**:
  - ✅ **Normal**: Para dados que seguem distribuição normal (Gaussiana)
  - ✅ **Não Normal**: Para dados que não seguem distribuição normal (usa percentis)

- **Tipo de Tolerância**:
  - 📏 **Bilateral**: Possui limite superior (LSE) e inferior (LIE)
  - ⬆️ **Superior Unilateral**: Apenas limite superior (LSE)
  - ⬇️ **Inferior Unilateral**: Apenas limite inferior (LIE)

#### Limites de Especificação:
- **LSE** (Limite Superior de Especificação): Valor máximo aceitável
- **LIE** (Limite Inferior de Especificação): Valor mínimo aceitável

### 4. Calcular e Visualizar
- Clique em **"🔍 Calcular Capacidade"**
- Aguarde o processamento
- Os resultados serão exibidos com:
  - 📊 **Tabelas de Índices**: Cp, Cpk, Pp, Ppk com intervalos de confiança
  - 📈 **Histograma**: Distribuição dos dados com limites e curva normal
  - 📊 **Gráfico de Barras**: Comparação visual dos índices

## 📖 Interpretação dos Resultados

### Índices de Capacidade

| Índice | Descrição | Interpretação |
|--------|-----------|---------------|
| **Cp** | Capacidade Potencial | Variação do processo vs. tolerância (ignora centralização) |
| **Cpk** | Capacidade Real | Considera variação E centralização do processo |
| **Pp** | Performance Potencial | Baseado em sigma total (longo prazo) |
| **Ppk** | Performance Real | Performance real considerando centralização |

### Classificação de Qualidade

| Valor | Classificação | Significado |
|-------|---------------|-------------|
| **≥ 1.33** | 🟢 Excelente | Processo capaz, baixíssima taxa de defeitos |
| **1.0 - 1.33** | 🟡 Aceitável | Processo marginalmente capaz |
| **< 1.0** | 🔴 Inadequado | Processo incapaz, alta taxa de defeitos |

### Métricas do Processo

- **Média**: Valor médio das medições
- **Sigma Within**: Desvio padrão de curto prazo (variação natural)
- **Sigma Overall**: Desvio padrão de longo prazo (variação total)
- **Estabilidade**: Razão Overall/Within (ideal: próximo de 1.0)
- **PPM**: Partes por milhão com defeito (apenas para não-normal)

## 📁 Exemplos de Dados

### Exemplo 1: Análise Simples (sem fase)
```csv
Medida
99.5
100.2
99.8
100.1
...
```
- LSE: 102.0
- LIE: 98.0
- Distribuição: Normal
- Tolerância: Bilateral

### Exemplo 2: Análise com Fase
```csv
Maquina,Diametro
A,99.5
A,100.2
B,100.5
B,101.2
...
```
- Coluna X: Maquina
- Coluna Y: Diametro
- Análise separada para cada máquina

## ⚠️ Dicas Importantes

1. **Dados Suficientes**: Tenha pelo menos 30 observações para análise confiável
2. **Verificar Normalidade**: Use testes de normalidade antes de escolher "Normal"
3. **Processo Estável**: A análise pressupõe processo estatisticamente estável
4. **Outliers**: Remova outliers extremos que não representam o processo normal
5. **Limites Corretos**: Verifique se LSE e LIE estão corretos conforme especificação

## 🎯 Casos de Uso

### Manufatura
- Avaliar se máquinas produzem peças dentro das tolerâncias
- Comparar capacidade entre diferentes equipamentos/operadores
- Monitorar degradação de capacidade ao longo do tempo

### Qualidade
- Validar processos novos ou modificados
- Demonstrar conformidade para certificações
- Priorizar melhorias baseadas em índices baixos

### Six Sigma
- Calcular nível sigma do processo
- Estabelecer baseline antes de projetos de melhoria
- Validar ganhos após implementação de melhorias

## 🔧 Solução de Problemas

| Problema | Solução |
|----------|---------|
| "Dados insuficientes" | Importe arquivo com mais dados (mínimo 2 valores) |
| Índices muito baixos | Verifique se limites LSE/LIE estão corretos |
| Gráfico distorcido | Remova outliers extremos dos dados |
| "Selecione coluna Y válida" | Certifique-se de que a coluna existe e contém números |

## 📚 Referências

- AIAG SPC Manual (Automotive Industry Action Group)
- ISO 22514-1:2014 - Statistical methods in process management
- Montgomery, D.C. (2012). Introduction to Statistical Quality Control

---

💡 **Dica**: Use a funcionalidade de histórico para acessar rapidamente análises anteriores!
