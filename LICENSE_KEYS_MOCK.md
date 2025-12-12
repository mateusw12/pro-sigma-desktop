# 🔑 Chaves de Licença para Testes

Este arquivo contém chaves de licença válidas para testes e desenvolvimento do Pro Sigma.

**⚠️ ATENÇÃO:** Estas são chaves de TESTE e DESENVOLVIMENTO. Não use em produção!

---

## 📋 Chaves Disponíveis

Todas as chaves são válidas até: **31/12/2026**

### 🔹 Plano BÁSICO

**Chave:**
```
eyJwbGFuIjogImJhc2ljIiwgImV4cGlyYXRlZERhdGUiOiAiMjAyNi0xMi0zMSJ9fDE1YTA2ZTY4ZjI3NWRmN2Q=
```

**Ferramentas incluídas:**
- ✓ Análise de Variabilidade
- ✓ Process Capability (Cp, Cpk, Pp, Ppk)
- ✓ Testes de Hipótese (T, Z, ANOVA, Qui-quadrado)
- ✓ Teste de Distribuição (Normal, Weibull, etc)
- ✓ COV EMS
- ✓ Análise de Distribuição
- ✓ Analytics

---

### 🔸 Plano INTERMEDIÁRIO

**Chave:**
```
eyJwbGFuIjogImludGVybWVkaWF0ZSIsICJleHBpcmF0ZWREYXRlIjogIjIwMjYtMTItMzEifXw1NDhhZmIzMGQ2MWRjOWM0
```

**Ferramentas incluídas:**
- ✓ Todas do Plano Básico
- ✓ Text Analysis
- ✓ Testes de Normalidade (Shapiro-Wilk, KS, etc)
- ✓ Cartas de Controle (X-bar, R, S, P, NP, C, U)
- ✓ Dashboard
- ✓ Simulações Monte Carlo

---

### 🔺 Plano PRO

**Chave:**
```
eyJwbGFuIjogInBybyIsICJleHBpcmF0ZWREYXRlIjogIjIwMjYtMTItMzEifXw2YTkxNmJkYzljMjk0YjVm
```

**Ferramentas incluídas:**
- ✓ Todas do Plano Intermediário
- ✓ Regressão Simples
- ✓ Regressão Múltipla
- ✓ Análise Multivariada (PCA, Análise Fatorial, Cluster)
- ✓ StackUp (Análise de tolerâncias 2D)
- ✓ DOE (Design of Experiments)
- ✓ Space Filling (Latin Hypercube)
- ✓ Análise de Custos de Garantia

---

## 🚀 Como Usar

1. Execute a aplicação:
   ```bash
   python main.py
   ```

2. Na tela de ativação, cole uma das chaves acima

3. A licença será salva localmente em `~/.pro_sigma/license.dat`

4. Nas próximas execuções, você entrará direto na aplicação

---

## 🔄 Gerando Novas Chaves

Para gerar novas chaves de teste com datas diferentes:

```python
from src.core.license_manager import LicenseManager

lm = LicenseManager()

# Gerar chave para plano básico válida até 2027
key = lm.generate_license('basic', '2027-12-31')
print(key)
```

Ou execute:
```bash
python src/core/license_manager.py
```

---

## 🗑️ Remover Licença Salva

Para testar o fluxo de ativação novamente, delete o arquivo:
- **Windows:** `C:\Users\[seu-usuario]\.pro_sigma\license.dat`
- **Linux/Mac:** `~/.pro_sigma/license.dat`

Ou via código:
```python
from src.core.license_manager import LicenseManager
lm = LicenseManager()
lm.remove_license()
```

---

## 📝 Notas Técnicas

- As chaves são codificadas em Base64
- Contêm um hash SHA256 para verificação de integridade
- Formato interno: `{plan, expiratedDate}` + hash de validação
- A secret key usada é: `ProSigma2025SecretKey` (trocar em produção!)

---

**Última atualização:** Dezembro 2025
