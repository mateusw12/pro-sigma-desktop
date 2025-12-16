# Configuração REML para COV

## O que é REML?

REML (Restricted Maximum Likelihood) é um método mais robusto que EMS para estimar componentes de variância em modelos mistos. É especialmente útil para:
- Dados desbalanceados
- Estruturas complexas de efeitos aleatórios
- Análises que exigem maior rigor estatístico

## Instalação Completa (Recomendado)

Para obter resultados idênticos ao R (lmer), instale o pymer4:

### 1. Instalar R
- Windows: https://cran.r-project.org/bin/windows/base/
- Certifique-se de adicionar R ao PATH durante instalação

### 2. Instalar pacote lme4 no R
Abra o R Console e execute:
```r
install.packages("lme4")
```

### 3. Instalar pymer4 no Python
```bash
pip install pymer4
```

## Instalação Básica (Fallback)

Se não puder instalar R, o Pro Sigma usará automaticamente statsmodels como fallback:
```bash
pip install statsmodels
```

**Nota**: O fallback é menos preciso que lmer mas funcional para a maioria dos casos.

## Verificação

Para verificar se pymer4 está funcionando:
```python
from pymer4.models import Lmer
print("✓ pymer4 instalado corretamente!")
```

## Troubleshooting

### Erro: "R not found"
- Certifique-se que R está no PATH
- Windows: Adicione `C:\Program Files\R\R-4.x.x\bin\x64` ao PATH

### Erro: "lme4 not installed"
- Execute `install.packages("lme4")` no R Console

### Fallback automático
Se pymer4 não estiver disponível, o sistema usará statsmodels automaticamente.
