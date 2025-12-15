# Diagrama de Ishikawa (Espinha de Peixe)

## Visão Geral

O Diagrama de Ishikawa, também conhecido como Diagrama de Espinha de Peixe ou Diagrama de Causa e Efeito, é uma ferramenta de qualidade usada para identificar, explorar e exibir graficamente as possíveis causas de um problema específico ou condição (efeito).

## Características

### Funcionalidades Principais

- **Interface Intuitiva**: Interface gráfica amigável com customtkinter
- **Categorias Personalizáveis**: Adicione até 8 categorias de causas
- **Múltiplas Causas**: Até 5 causas por categoria
- **Visualização em Tempo Real**: Diagrama atualizado instantaneamente
- **Exportação PNG**: Salve o diagrama em alta resolução (300 DPI)
- **Categorias Pré-definidas**: Sugestões baseadas nos 6M's

### Categorias Clássicas (6M's)

1. **Método** - Processos e procedimentos
2. **Material** - Matérias-primas e insumos
3. **Mão de Obra** - Pessoas e habilidades
4. **Máquina** - Equipamentos e ferramentas
5. **Medição** - Instrumentos e calibração
6. **Meio Ambiente** - Condições ambientais

## Como Usar

### Passo 1: Definir o Efeito
1. No campo "Efeito (Problema)", digite o problema ou efeito que deseja analisar
2. Opcionalmente, adicione um título descritivo para o diagrama

### Passo 2: Adicionar Categorias
1. Clique em "➕ Adicionar Categoria"
2. Digite o nome da categoria (ex: "Método", "Material")
3. A ferramenta sugere automaticamente os 6M's tradicionais

### Passo 3: Adicionar Causas
1. Para cada categoria, preencha até 5 causas
2. Deixe campos vazios se não houver causas suficientes
3. Campos vazios são ignorados automaticamente

### Passo 4: Visualizar
1. Clique em "🔄 Atualizar Diagrama"
2. O diagrama será gerado e exibido no painel direito
3. Categorias alternam entre posições superiores e inferiores

### Passo 5: Exportar
1. Clique em "💾 Exportar PNG"
2. Escolha o local e nome do arquivo
3. O diagrama será salvo em alta resolução (300 DPI)

## Exemplo de Uso

### Problema: "Defeito no Produto"

**Categorias e Causas:**

1. **Método**
   - Processo inadequado
   - Falta de padronização

2. **Material**
   - Matéria-prima com defeito
   - Armazenamento incorreto

3. **Mão de Obra**
   - Falta de treinamento
   - Fadiga

4. **Máquina**
   - Equipamento desregulado
   - Falta de manutenção

5. **Medição**
   - Instrumento descalibrado
   - Erro de leitura

6. **Meio Ambiente**
   - Temperatura inadequada
   - Umidade elevada

## Limitações

- **Máximo de 8 categorias**: Para manter a legibilidade do diagrama
- **Máximo de 5 causas por categoria**: Evita sobrecarga visual
- **Efeito obrigatório**: Deve ser preenchido para criar o diagrama
- **Pelo menos 1 categoria**: Mínimo de uma categoria com causas

## Benefícios

### Análise de Problemas
- Identifica causas raiz de problemas
- Organiza ideias de forma estruturada
- Facilita brainstorming em equipe

### Comunicação Visual
- Apresentação clara de relações causa-efeito
- Fácil entendimento por stakeholders
- Documentação visual de análises

### Melhoria Contínua
- Base para planos de ação
- Priorização de causas
- Acompanhamento de melhorias

## Dicas de Uso

1. **Brainstorming em Equipe**: Use em sessões colaborativas
2. **5 Porquês**: Combine com a técnica dos "5 Porquês" para causas raiz
3. **Priorização**: Após criar o diagrama, priorize as causas mais impactantes
4. **Revisão Periódica**: Atualize o diagrama conforme novas causas são identificadas
5. **Documentação**: Exporte e anexe em relatórios e apresentações

## Casos de Uso Comuns

- **Manufatura**: Análise de defeitos de produção
- **Serviços**: Investigação de problemas de qualidade
- **Processos**: Identificação de gargalos
- **Projetos**: Análise de riscos e problemas
- **Manutenção**: Diagnóstico de falhas em equipamentos

## Formato de Exportação

### PNG de Alta Qualidade
- Resolução: 300 DPI
- Fundo: Branco
- Formato: PNG com transparência
- Adequado para: Impressão, apresentações, relatórios

## Integração

A ferramenta está integrada ao menu principal do ProSigma:
- **Categoria**: Ferramentas Básicas
- **Ícone**: 🐟 (peixe, referência ao "fishbone diagram")
- **Acesso**: Não requer importação de dados
- **Nível**: Basic (disponível em todos os planos)

## Tecnologias Utilizadas

- **Interface**: customtkinter
- **Visualização**: matplotlib
- **Exportação**: matplotlib.savefig
- **Canvas**: FigureCanvasTkAgg para integração Tkinter

## Validação de Dados

A ferramenta valida automaticamente:
- ✅ Efeito não vazio
- ✅ Pelo menos 1 categoria
- ✅ Máximo de 8 categorias
- ✅ Máximo de 5 causas por categoria
- ✅ Nomes de categoria não vazios

## Próximas Melhorias Possíveis

- [ ] Exportação para PDF
- [ ] Temas de cores personalizáveis
- [ ] Importação/exportação de dados em JSON
- [ ] Templates pré-configurados
- [ ] Anotações e comentários no diagrama
- [ ] Suporte a sub-causas (níveis hierárquicos)

## Referências

- Kaoru Ishikawa - Criador da ferramenta
- Metodologia Six Sigma
- Gestão da Qualidade Total (TQM)
- Ferramentas da Qualidade (7 Ferramentas)
