# Stack-Up Analysis - Implementação Concluída

## Resumo da Implementação

A ferramenta de **Stack-Up (Empilhamento de Tolerâncias)** foi implementada com sucesso no ProSigma Desktop.

## Arquivos Criados

### Módulo Principal
```
src/analytics/stack_up/
├── __init__.py                 # Inicialização do módulo
├── stack_up_utils.py          # Funções de cálculo
├── stack_up_window.py         # Interface gráfica (PyQt6)
├── README.md                  # Documentação completa
├── test_stack_up.py           # Testes unitários
└── example_usage.py           # Exemplos de uso
```

## Funcionalidades Implementadas

### ✅ Entrada de Dados
- [x] Configuração de múltiplas características (1-50)
- [x] Campos para Min, Max, Sensibilidade e Quota
- [x] Validação de dados
- [x] Interface responsiva com scroll

### ✅ Importação/Exportação
- [x] Download de template Excel
- [x] Importação de arquivos Excel/CSV
- [x] Exportação de resultados para Excel

### ✅ Cálculos Estatísticos
- [x] Cálculo de médias: (Max + Min) / 2
- [x] Cálculo de desvios padrão: (Max - Min) / (6 × Quota)
- [x] Simulação Monte Carlo configurável (100-250.000 rodadas)
- [x] Geração de distribuições normais
- [x] Cálculo da equação resultante

### ✅ Visualização de Resultados
- [x] Tabela de resumo com médias e desvios padrão
- [x] Exibição da equação final
- [x] Interface limpa e organizada

### ✅ Integração no Sistema
- [x] Adicionado ao menu principal (home_page.py)
- [x] Ícone configurado (📏)
- [x] Status: implementado (in_development: False)
- [x] Plano: Intermediate

### ✅ Documentação
- [x] README detalhado com exemplos
- [x] Testes unitários completos
- [x] Exemplos de uso programático
- [x] Fundamentação teórica incluída

## Tipos de Quota Suportados

| Quota | Valor | Descrição |
|-------|-------|-----------|
| **Standard** | 1 | Controle padrão |
| **CTS** | 1.33 | Critical to Schedule (crítico para cronograma) |
| **CTQ** | 2 | Critical to Quality (crítico para qualidade) |

## Casos de Uso

1. **Montagem Mecânica**: Análise de empilhamento de tolerâncias em montagens
2. **Processos de Manufatura**: Avaliação de múltiplas operações em série
3. **Controle de Qualidade**: Cálculo de capacidade de processo
4. **Otimização de Tolerâncias**: Identificação de características críticas

## Tecnologias Utilizadas

- **Python 3.x**
- **PyQt6**: Interface gráfica
- **NumPy**: Cálculos numéricos e distribuições
- **Pandas**: Manipulação de dados
- **OpenPyXL**: Exportação para Excel

## Como Usar

### Interface Gráfica

1. Acesse o ProSigma Desktop
2. No menu principal, clique em "StackUp" (📏)
3. Configure:
   - Número de características
   - Número de rodadas (recomendado: 5000)
4. Clique em "Gerar Características"
5. Preencha os dados de cada característica
6. Clique em "Calcular"
7. Visualize os resultados e exporte se necessário

### Uso Programático

```python
from src.analytics.stack_up.stack_up_utils import calculate_stack_up

factors = {
    'factor_1': {
        'name': 'Peça A',
        'min': 99.8,
        'max': 100.2,
        'sensitivity': 1.0,
        'quota': '1'
    },
    'factor_2': {
        'name': 'Peça B',
        'min': 49.9,
        'max': 50.1,
        'sensitivity': 1.0,
        'quota': '2'
    }
}

resultado = calculate_stack_up(rounds=5000, factors=factors)

print(f"Equação: {resultado['equation']}")
print(f"Médias: {resultado['means']}")
print(f"Desvios: {resultado['stds']}")

# Exportar dados
df = resultado['dataframe']
df.to_excel('stack_up_results.xlsx', index=False)
```

## Testes

Execute os testes com:

```bash
python -m pytest src/analytics/stack_up/test_stack_up.py -v
```

Ou use unittest:

```bash
python src/analytics/stack_up/test_stack_up.py
```

## Exemplos

Execute os exemplos com:

```bash
python src/analytics/stack_up/example_usage.py
```

Os exemplos demonstram:
- Análise básica
- Processo de manufatura
- Sensibilidades diferentes
- Comparação de quotas
- Exportação de dados

## Validações Implementadas

- ✅ Mínimo deve ser menor que máximo
- ✅ Sensibilidade não pode ser zero
- ✅ Pelo menos um fator deve ser fornecido
- ✅ Valores numéricos válidos
- ✅ Nomes de características únicos

## Performance

- **Rodadas recomendadas**: 5.000
- **Tempo estimado (5.000 rodadas, 5 características)**: < 1 segundo
- **Rodadas máximas**: 250.000
- **Características máximas**: 50

## Próximas Melhorias Possíveis

### Futuro (Opcional)

- [ ] Gráficos de distribuição (histograma do Y)
- [ ] Análise de sensibilidade visual
- [ ] Gráfico de Pareto das contribuições
- [ ] Cálculo automático de Cp/Cpk
- [ ] Análise de Monte Carlo 3D
- [ ] Otimização de tolerâncias
- [ ] Relatórios em PDF
- [ ] Integração com banco de dados de características

## Notas Importantes

1. **Simulação Monte Carlo**: Os resultados são probabilísticos e podem variar ligeiramente entre execuções
2. **Premissa**: As características seguem distribuição normal
3. **Quota**: Valores maiores indicam controle mais rigoroso
4. **Sensibilidade**: Pode ser positiva ou negativa

## Referências

- ISO 1101: Tolerâncias geométricas
- ASME Y14.5: Dimensionamento e tolerância
- Montgomery, D.C. (2009). Introduction to Statistical Quality Control
- Law, A.M., & Kelton, W.D. (2000). Simulation Modeling and Analysis

## Status do Projeto

✅ **IMPLEMENTAÇÃO CONCLUÍDA** - Stack-Up Analysis está pronto para uso!

---

**Versão**: 1.0.0  
**Data**: 15/12/2025  
**Desenvolvedor**: ProSigma Team  
**Licença**: Requer plano Intermediate ou superior
