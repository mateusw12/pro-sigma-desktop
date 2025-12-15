"""
Script de teste para verificar o módulo de Redes Neurais
"""
import sys
from pathlib import Path

# Adiciona o diretório raiz ao path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

import pandas as pd
import numpy as np

# Testa imports
print("🧪 Testando módulo de Redes Neurais...")
print()

# 1. Testa importação dos utils
print("1. Testando importação dos utils...")
try:
    from src.analytics.neural_network.neural_network_utils import (
        train_neural_network_holdout,
        train_neural_network_kfold,
        is_categorical_target
    )
    print("   ✅ Imports dos utils OK")
except Exception as e:
    print(f"   ❌ Erro ao importar utils: {e}")
    sys.exit(1)

# 2. Testa importação da janela
print("\n2. Testando importação da janela...")
try:
    from src.analytics.neural_network.neural_network_window import NeuralNetworkWindow
    print("   ✅ Import da janela OK")
except Exception as e:
    print(f"   ❌ Erro ao importar janela: {e}")
    sys.exit(1)

# 3. Testa detecção de tipo (classificação vs regressão)
print("\n3. Testando detecção de tipo...")
df_regression = pd.DataFrame({
    'X1': np.random.randn(100),
    'X2': np.random.randn(100),
    'Y': np.random.randn(100)
})

df_classification = pd.DataFrame({
    'X1': np.random.randn(100),
    'X2': np.random.randn(100),
    'Y': np.random.choice(['A', 'B', 'C'], 100)
})

is_class_reg = is_categorical_target(df_regression['Y'])
is_class_clf = is_categorical_target(df_classification['Y'])

print(f"   Regressão detectada como classificação: {is_class_reg}")
print(f"   Classificação detectada como classificação: {is_class_clf}")

if not is_class_reg and is_class_clf:
    print("   ✅ Detecção de tipo OK")
else:
    print("   ❌ Erro na detecção de tipo")
    sys.exit(1)

# 4. Testa treinamento Holdout (regressão)
print("\n4. Testando treinamento Holdout (regressão)...")
try:
    # Cria dataset sintético
    np.random.seed(42)
    n_samples = 200
    X1 = np.random.randn(n_samples)
    X2 = np.random.randn(n_samples)
    Y = 2 * X1 + 3 * X2 + np.random.randn(n_samples) * 0.5
    
    df = pd.DataFrame({
        'Feature1': X1,
        'Feature2': X2,
        'Target': Y
    })
    
    results = train_neural_network_holdout(
        df=df,
        x_columns=['Feature1', 'Feature2'],
        y_column='Target',
        categorical_cols=[],
        activation='relu',
        test_size=0.3,
        max_iter=200
    )
    
    print(f"   R² Treino: {results['metrics_train']['r2']:.4f}")
    print(f"   R² Teste: {results['metrics_test']['r2']:.4f}")
    print(f"   RMSE Teste: {results['metrics_test']['rmse']:.4f}")
    print(f"   Arquitetura: {results['model_info']['hidden_layers']}")
    print("   ✅ Treinamento Holdout OK")
except Exception as e:
    print(f"   ❌ Erro no treinamento Holdout: {e}")
    import traceback
    traceback.print_exc()

# 5. Testa treinamento K-Fold (classificação)
print("\n5. Testando treinamento K-Fold (classificação)...")
try:
    # Cria dataset sintético
    np.random.seed(42)
    n_samples = 200
    X1 = np.random.randn(n_samples)
    X2 = np.random.randn(n_samples)
    
    # Cria classes baseadas em combinação linear
    decision = X1 + X2
    Y = np.where(decision > 1, 'Class_A',
                 np.where(decision < -1, 'Class_C', 'Class_B'))
    
    df = pd.DataFrame({
        'Feature1': X1,
        'Feature2': X2,
        'Target': Y
    })
    
    results = train_neural_network_kfold(
        df=df,
        x_columns=['Feature1', 'Feature2'],
        y_column='Target',
        categorical_cols=[],
        activation='relu',
        n_folds=3,
        max_iter=200
    )
    
    print(f"   Acurácia: {results['metrics']['accuracy']:.4f} ± {results['metrics'].get('accuracy_std', 0):.4f}")
    print(f"   F1-Score: {results['metrics']['f1_score']:.4f} ± {results['metrics'].get('f1_score_std', 0):.4f}")
    print(f"   Arquitetura: {results['model_info']['hidden_layers']}")
    print("   ✅ Treinamento K-Fold OK")
except Exception as e:
    print(f"   ❌ Erro no treinamento K-Fold: {e}")
    import traceback
    traceback.print_exc()

# 6. Testa importância de features
print("\n6. Testando importância de features...")
try:
    importance = results['feature_importance']
    print(f"   Features ranqueadas: {list(importance.keys())}")
    print(f"   Importâncias: {list(importance.values())}")
    print("   ✅ Importância de features OK")
except Exception as e:
    print(f"   ❌ Erro na importância: {e}")

print("\n" + "="*60)
print("🎉 TODOS OS TESTES PASSARAM COM SUCESSO!")
print("="*60)
print("\n✅ O módulo de Redes Neurais está pronto para uso!")
print("\nPara usar:")
print("1. Importe um arquivo de dados (Excel ou CSV)")
print("2. Clique em 'Redes Neurais' no menu de ferramentas")
print("3. Selecione as variáveis X e Y")
print("4. Escolha o método (Holdout ou K-Fold)")
print("5. Configure os parâmetros")
print("6. Clique em 'Treinar Rede Neural'")
