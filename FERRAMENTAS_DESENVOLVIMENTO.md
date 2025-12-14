# Sistema de Ferramentas em Desenvolvimento

## Visão Geral
O sistema permite marcar ferramentas como "em desenvolvimento" para desabilitá-las visualmente na interface até que estejam prontas para uso.

## Como Usar

### 1. Marcar uma Ferramenta como "Em Desenvolvimento"

No arquivo `src/ui/home_page.py`, localize a definição da ferramenta no dicionário `tools_definition` e adicione o campo `'in_development': True`:

```python
'nome_da_ferramenta': {
    'title': 'Nome da Ferramenta',
    'description': 'Descrição da ferramenta',
    'plan': 'basic',  # ou 'intermediate' ou 'pro'
    'in_development': True  # Marca como em desenvolvimento
},
```

### 2. Habilitar uma Ferramenta Pronta

Quando a ferramenta estiver pronta, basta alterar para `False` ou remover o campo:

```python
'nome_da_ferramenta': {
    'title': 'Nome da Ferramenta',
    'description': 'Descrição da ferramenta',
    'plan': 'basic',
    'in_development': False  # Ferramenta habilitada
},
```

## Comportamento Visual

### Ferramenta Habilitada (in_development: False ou ausente)
- ✅ Cor de fundo: `gray20`
- ✅ Hover: `#2E86DE` (azul)
- ✅ Texto: branco
- ✅ Borda: `gray30`
- ✅ Clique: Abre a ferramenta normalmente

### Ferramenta Desabilitada (in_development: True)
- 🚧 Cor de fundo: `gray15` (mais escuro)
- 🚧 Hover: `gray15` (sem mudança)
- 🚧 Texto: `gray50` (acinzentado)
- 🚧 Borda: `gray25` (mais escura)
- 🚧 Label adicional: "🚧 Em Desenvolvimento"
- 🚧 Tooltip: Mensagem de aviso adicional
- 🚧 Clique: Mostra mensagem informativa

## Exemplo de Mensagem

Quando um usuário clica em uma ferramenta em desenvolvimento:

```
🚧 Nome da Ferramenta

Esta ferramenta ainda está em desenvolvimento e será 
disponibilizada em uma próxima versão do Pro Sigma.

Agradecemos sua compreensão!
```

## Estado Atual das Ferramentas

### ✅ Habilitadas
- Process Capability
- Testes de Hipótese
- Teste de Distribuição
- COV EMS
- Descriptive Statistics
- Testes de Normalidade
- Cartas de Controle
- Monte Carlo
- Análise de Variabilidade
- Análise de Texto

### 🚧 Em Desenvolvimento
- Analytics
- Dashboard
- Regressão Simples
- Regressão Múltipla
- Análise Multivariada
- StackUp
- DOE
- Space Filling
- Custos de Garantia
- Redes Neurais
- Árvore de Decisão

## Vantagens do Sistema

1. **Transparência**: Usuários veem todas as ferramentas planejadas
2. **Expectativa**: Sabem o que está por vir
3. **Feedback**: Podem expressar interesse em ferramentas específicas
4. **Desenvolvimento Gradual**: Ferramentas podem ser ativadas individualmente
5. **Manutenção Fácil**: Apenas uma flag para habilitar/desabilitar
6. **Visual Claro**: Diferenciação visual imediata
7. **Não Intrusivo**: Não interfere com ferramentas funcionais

## Código Responsável

### Criação do Card (home_page.py - método _create_tool_card)
```python
# Verificar se está em desenvolvimento
is_in_development = tool_info.get('in_development', False)

# Configuração visual baseada no status
if is_in_development:
    # Ferramenta desabilitada
    fg_color = "gray15"
    hover_color = "gray15"
    text_color = "gray50"
    border_color = "gray25"
    button_text = f"{icon}\n\n{tool_info['title']}\n\n🚧 Em Desenvolvimento"
    command = lambda: self._show_in_development_message(tool_info['title'])
else:
    # Ferramenta ativa
    fg_color = "gray20"
    hover_color = "#2E86DE"
    text_color = "white"
    border_color = "gray30"
    button_text = f"{icon}\n\n{tool_info['title']}"
    command = lambda: self.open_tool(feature_id)
```

### Mensagem de Desenvolvimento (home_page.py)
```python
def _show_in_development_message(self, tool_name):
    """Mostra mensagem quando ferramenta em desenvolvimento é clicada"""
    messagebox.showinfo(
        "Ferramenta em Desenvolvimento",
        f"🚧 {tool_name}\n\n"
        "Esta ferramenta ainda está em desenvolvimento e será "
        "disponibilizada em uma próxima versão do Pro Sigma.\n\n"
        "Agradecemos sua compreensão!"
    )
```

## Checklist para Nova Ferramenta

Ao adicionar uma nova ferramenta:

- [ ] Adicionar entrada no `tools_definition`
- [ ] Definir `title`, `description`, `plan`
- [ ] Definir `in_development: True` inicialmente
- [ ] Adicionar ícone correspondente no `icon_map`
- [ ] Implementar a funcionalidade da ferramenta
- [ ] Adicionar no método `open_tool`
- [ ] Testar a ferramenta
- [ ] Alterar para `in_development: False`
- [ ] Atualizar documentação

## Futuras Melhorias

- [ ] Badge com data estimada de lançamento
- [ ] Contador de usuários interessados
- [ ] Newsletter automática quando ferramenta for liberada
- [ ] Beta testers para ferramentas em desenvolvimento
- [ ] Preview da interface da ferramenta
- [ ] Roadmap visual de desenvolvimento
