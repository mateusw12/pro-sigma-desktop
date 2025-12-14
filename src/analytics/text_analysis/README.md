# Análise de Texto - Text Analysis

## Visão Geral
Ferramenta completa de mineração de texto e processamento de linguagem natural (NLP) similar ao Text Analysis do JMP.

## Funcionalidades

### 1. **Seleção de Dados**
- Escolha a coluna de texto para análise
- Suporte para qualquer coluna com dados textuais do DataFrame

### 2. **Configuração de Idioma**
- **Português**: Stopwords automáticas em português brasileiro
- **English**: Stopwords em inglês
- **Español**: Stopwords em espanhol

### 3. **Busca de Palavras/Frases**
- Busca por palavras-chave ou frases específicas
- Opção de busca case-sensitive
- Contagem de ocorrências
- Exibição de contextos onde a palavra aparece

### 4. **Personalização de Análise**
- **Palavras customizadas a ignorar**: Adicione suas próprias stopwords
- **Caracteres a ignorar**: Remove caracteres especiais específicos
- **Remover stopwords automáticas**: Toggle para usar stopwords padrão do idioma

### 5. **Análise de Palavras**
- Contagem de frequência de palavras
- Top 50 palavras mais frequentes
- Percentual de uso de cada palavra
- Ranking de palavras

### 6. **Análise de Frases (N-grams)**
- Bigramas (2 palavras): Frases com 2 palavras
- Trigramas (3 palavras): Frases com 3 palavras
- Quadrigramas (4 palavras): Frases com 4 palavras
- Top 30 frases mais frequentes

### 7. **Visualizações**

#### Nuvem de Palavras (Word Cloud)
- Visualização interativa das palavras mais frequentes
- Tamanho proporcional à frequência
- Até 100 palavras na nuvem
- Esquema de cores personalizável

#### Gráfico de Frequência de Palavras
- Gráfico de barras horizontal
- Top 20 palavras mais frequentes
- Labels com valores exatos

#### Gráfico de Frequência de Frases
- Gráfico de barras das frases mais comuns
- Top 15 frases
- Ideal para identificar padrões de linguagem

### 8. **Estatísticas Gerais**
- Total de documentos
- Documentos válidos (não vazios)
- Total de caracteres
- Média de caracteres por documento
- Total de palavras
- Média de palavras por documento
- Número de palavras únicas

### 9. **Exportação**
- Tabelas exportáveis em formato CSV
- Gráficos exportáveis em PNG/PDF

## Stopwords Incluídas

### Português
Inclui palavras comuns como: a, o, as, os, de, da, do, em, para, com, sem, e, ou, mas, se, que, ele, ela, eles, elas, não, sim, ser, estar, ter, etc.

### English
Inclui: a, an, the, and, or, but, if, then, i, you, he, she, it, we, they, yes, no, is, are, was, were, etc.

### Español
Inclui: el, la, los, las, de, en, por, para, con, y, o, pero, si, no, yo, tú, él, ella, ser, estar, etc.

## Pré-processamento de Texto

O sistema automaticamente:
1. Remove URLs e emails
2. Converte para minúsculas (opcional)
3. Remove espaços extras
4. Remove pontuação
5. Remove stopwords
6. Filtra palavras muito curtas (< 2 caracteres)
7. Mantém apenas palavras alfabéticas

## Casos de Uso

### 1. Análise de Feedback de Clientes
- Identifique palavras e temas mais mencionados
- Encontre padrões em reclamações ou elogios
- Busque termos específicos como "defeito", "qualidade", "entrega"

### 2. Análise de Relatórios de Não Conformidade
- Palavras mais comuns em NCs
- Frases recorrentes indicando problemas sistêmicos
- Busca por componentes ou processos específicos

### 3. Análise de Ações Corretivas
- Padrões em descrições de ações
- Efetividade de soluções similares
- Termos técnicos mais utilizados

### 4. Pesquisas de Satisfação
- Análise de respostas abertas
- Temas emergentes
- Sentimentos expressos

### 5. Análise de Documentação Técnica
- Termos técnicos mais usados
- Consistência de terminologia
- Gaps de documentação

## Exemplo de Uso

1. Importe seus dados com coluna de texto
2. Selecione a ferramenta "Análise de Texto"
3. Escolha a coluna com os textos
4. Selecione o idioma
5. (Opcional) Configure palavras customizadas a ignorar
6. (Opcional) Digite palavra-chave para buscar
7. Clique em "Analisar Texto"
8. Explore as tabelas e visualizações geradas

## Tecnologias Utilizadas

- **pandas**: Manipulação de dados
- **matplotlib**: Visualizações
- **wordcloud**: Geração de nuvens de palavras
- **collections.Counter**: Contagem eficiente
- **re**: Expressões regulares para processamento

## Performance

- Otimizado para análise de até 10.000 documentos
- Processamento rápido com regex otimizadas
- Lazy loading de bibliotecas pesadas
- Cache inteligente de resultados

## Limitações

- Não realiza análise de sentimentos (pode ser adicionado)
- Não identifica entidades nomeadas (NER)
- Não faz stemming ou lematização automática
- Funciona melhor com textos em um único idioma

## Melhorias Futuras

- [ ] Análise de sentimentos com machine learning
- [ ] Identificação de entidades (NER)
- [ ] Stemming e lematização
- [ ] Análise de tópicos (LDA)
- [ ] Clustering de documentos similares
- [ ] Comparação entre grupos de textos
- [ ] Suporte para mais idiomas
- [ ] Exportação de relatório completo
