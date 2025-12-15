"""
Exemplo de uso da ferramenta Stack-Up
Demonstra como usar programaticamente o módulo de Stack-Up
"""

from src.utils.lazy_imports import get_pandas
from src.analytics.stack_up.stack_up_utils import calculate_stack_up


def exemplo_basico():
    """Exemplo básico de análise de stack-up"""
    print("=" * 60)
    print("EXEMPLO BÁSICO: Montagem de 2 peças")
    print("=" * 60)
    
    # Define as características
    factors = {
        'factor_1': {
            'name': 'Peça 1',
            'min': 99.8,
            'max': 100.2,
            'sensitivity': 1.0,
            'quota': '1'
        },
        'factor_2': {
            'name': 'Peça 2',
            'min': 49.9,
            'max': 50.1,
            'sensitivity': 1.0,
            'quota': '1'
        }
    }
    
    # Calcula o stack-up
    resultado = calculate_stack_up(rounds=5000, factors=factors)
    
    # Exibe resultados
    print("\nMÉDIAS:")
    for nome, media in resultado['means'].items():
        print(f"  {nome}: {media:.4f}")
    
    print("\nDESVIOS PADRÃO:")
    for nome, std in resultado['stds'].items():
        print(f"  {nome}: {std:.6f}")
    
    print(f"\nEQUAÇÃO:")
    print(f"  {resultado['equation']}")
    
    # Estatísticas do resultado
    df = resultado['dataframe']
    print(f"\nESTATÍSTICAS DO RESULTADO (Y):")
    print(f"  Média: {df['Y'].mean():.4f}")
    print(f"  Desvio Padrão: {df['Y'].std():.6f}")
    print(f"  Mínimo: {df['Y'].min():.4f}")
    print(f"  Máximo: {df['Y'].max():.4f}")
    
    return resultado


def exemplo_processo_manufatura():
    """Exemplo de processo de manufatura com múltiplas operações"""
    print("\n" + "=" * 60)
    print("EXEMPLO: Processo de Manufatura")
    print("=" * 60)
    
    # Define as operações
    factors = {
        'corte': {
            'name': 'Corte',
            'min': 99.5,
            'max': 100.5,
            'sensitivity': 1.0,
            'quota': '1'
        },
        'usinagem': {
            'name': 'Usinagem',
            'min': -0.3,
            'max': -0.1,
            'sensitivity': 1.0,  # Remove material
            'quota': '2'  # CTQ - operação crítica
        },
        'acabamento': {
            'name': 'Acabamento',
            'min': -0.05,
            'max': 0.05,
            'sensitivity': 1.0,
            'quota': '2'  # CTQ - operação crítica
        }
    }
    
    # Calcula o stack-up
    resultado = calculate_stack_up(rounds=10000, factors=factors)
    
    # Exibe resultados
    print("\nRESUMO DAS OPERAÇÕES:")
    for nome in resultado['means'].keys():
        media = resultado['means'][nome]
        std = resultado['stds'][nome]
        print(f"\n{nome}:")
        print(f"  Média: {media:.4f} mm")
        print(f"  Desvio Padrão: {std:.6f} mm")
    
    print(f"\nEQUAÇÃO DO PROCESSO:")
    print(f"  {resultado['equation']}")
    
    # Análise da dimensão final
    df = resultado['dataframe']
    print(f"\nDIMENSÃO FINAL:")
    print(f"  Média: {df['Y'].mean():.4f} mm")
    print(f"  Desvio Padrão: {df['Y'].std():.6f} mm")
    print(f"  Mínimo esperado: {df['Y'].min():.4f} mm")
    print(f"  Máximo esperado: {df['Y'].max():.4f} mm")
    
    # Capacidade do processo (assumindo LSL=99.0 e USL=100.0)
    LSL = 99.0
    USL = 100.0
    mean = df['Y'].mean()
    std = df['Y'].std()
    
    Cp = (USL - LSL) / (6 * std)
    Cpk = min((USL - mean) / (3 * std), (mean - LSL) / (3 * std))
    
    print(f"\nCAPACIDADE DO PROCESSO (LSL={LSL}, USL={USL}):")
    print(f"  Cp: {Cp:.3f}")
    print(f"  Cpk: {Cpk:.3f}")
    
    if Cpk >= 1.33:
        print("  Status: ✓ Processo capaz (Cpk ≥ 1.33)")
    elif Cpk >= 1.0:
        print("  Status: ⚠ Processo aceitável (1.0 ≤ Cpk < 1.33)")
    else:
        print("  Status: ✗ Processo não capaz (Cpk < 1.0)")
    
    return resultado


def exemplo_sensibilidades_diferentes():
    """Exemplo com diferentes sensibilidades"""
    print("\n" + "=" * 60)
    print("EXEMPLO: Sensibilidades Diferentes")
    print("=" * 60)
    
    factors = {
        'fator_a': {
            'name': 'Fator A',
            'min': 10.0,
            'max': 11.0,
            'sensitivity': 2.0,  # Alta influência
            'quota': '1'
        },
        'fator_b': {
            'name': 'Fator B',
            'min': 5.0,
            'max': 6.0,
            'sensitivity': 0.5,  # Baixa influência
            'quota': '1'
        },
        'fator_c': {
            'name': 'Fator C',
            'min': 3.0,
            'max': 4.0,
            'sensitivity': -1.5,  # Influência negativa
            'quota': '1'
        }
    }
    
    resultado = calculate_stack_up(rounds=5000, factors=factors)
    
    print("\nANÁLISE DE SENSIBILIDADE:")
    
    # Calcula a contribuição de cada fator para a variância
    df = resultado['dataframe']
    total_var = df['Y'].var()
    
    for nome in resultado['means'].keys():
        std = resultado['stds'][nome]
        sensitivity = None
        for key, factor in factors.items():
            if factor['name'] == nome:
                sensitivity = factor['sensitivity']
                break
        
        if sensitivity:
            contribution = (sensitivity * std) ** 2
            percent = (contribution / total_var) * 100 if total_var > 0 else 0
            
            print(f"\n{nome}:")
            print(f"  Sensibilidade: {sensitivity}")
            print(f"  Desvio Padrão: {std:.6f}")
            print(f"  Contribuição para variância: {percent:.2f}%")
    
    print(f"\nEQUAÇÃO:")
    print(f"  {resultado['equation']}")
    
    print(f"\nRESULTADO FINAL:")
    print(f"  Média: {df['Y'].mean():.4f}")
    print(f"  Desvio Padrão: {df['Y'].std():.6f}")
    
    return resultado


def exemplo_exportar_dados():
    """Exemplo de exportação de dados"""
    print("\n" + "=" * 60)
    print("EXEMPLO: Exportação de Dados")
    print("=" * 60)
    
    factors = {
        'factor_1': {
            'name': 'Característica 1',
            'min': 100.0,
            'max': 101.0,
            'sensitivity': 1.0,
            'quota': '1'
        },
        'factor_2': {
            'name': 'Característica 2',
            'min': 50.0,
            'max': 51.0,
            'sensitivity': 1.0,
            'quota': '1'
        }
    }
    
    resultado = calculate_stack_up(rounds=1000, factors=factors)
    
    # Exporta para Excel
    df = resultado['dataframe']
    output_file = 'stack_up_exemplo.xlsx'
    
    try:
        df.to_excel(output_file, index=False)
        print(f"\n✓ Dados exportados para: {output_file}")
        print(f"  Total de linhas: {len(df)}")
        print(f"  Colunas: {', '.join(df.columns)}")
    except Exception as e:
        print(f"\n✗ Erro ao exportar: {e}")
    
    return resultado


def exemplo_comparacao_quotas():
    """Compara o impacto de diferentes quotas"""
    print("\n" + "=" * 60)
    print("EXEMPLO: Comparação de Quotas")
    print("=" * 60)
    
    quotas = {
        'Standard': '1',
        'CTS': '1.33',
        'CTQ': '2'
    }
    
    resultados = {}
    
    for nome_quota, valor_quota in quotas.items():
        factors = {
            'factor_1': {
                'name': 'Característica',
                'min': 99.0,
                'max': 101.0,
                'sensitivity': 1.0,
                'quota': valor_quota
            }
        }
        
        resultado = calculate_stack_up(rounds=5000, factors=factors)
        resultados[nome_quota] = resultado
        
        df = resultado['dataframe']
        print(f"\n{nome_quota} (Quota = {valor_quota}):")
        print(f"  Desvio Padrão da característica: {resultado['stds']['Característica']:.6f}")
        print(f"  Desvio Padrão do resultado Y: {df['Y'].std():.6f}")
    
    print("\nCONCLUSÃO:")
    print("  Quotas maiores (CTQ) resultam em desvios padrão menores,")
    print("  indicando controle mais rigoroso da característica.")
    
    return resultados


def main():
    """Executa todos os exemplos"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "EXEMPLOS DE USO - STACK-UP ANALYSIS" + " " * 12 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # Executa os exemplos
    exemplo_basico()
    exemplo_processo_manufatura()
    exemplo_sensibilidades_diferentes()
    exemplo_comparacao_quotas()
    
    # Exemplo de exportação (comentado para não criar arquivo)
    # exemplo_exportar_dados()
    
    print("\n" + "=" * 60)
    print("FIM DOS EXEMPLOS")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
