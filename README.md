# Pro Sigma

Software desktop para análise estatística Six Sigma desenvolvido em Python.

## 📋 Descrição

Pro Sigma é uma aplicação desktop completa para análises estatísticas voltadas à metodologia Six Sigma. O software oferece diferentes planos com ferramentas especializadas para profissionais da qualidade e análise de dados.

## 🚀 Estrutura do Projeto

```
pro-sigma-desktop/
├── src/
│   ├── core/              # Funcionalidades principais
│   │   ├── __init__.py
│   │   └── license_manager.py
│   ├── ui/                # Interface gráfica
│   │   ├── login_window.py
│   │   └── home_page.py
│   ├── analytics/         # Ferramentas de análise
│   └── utils/             # Utilitários
├── assets/                # Recursos visuais
├── data/                  # Dados de análises
├── main.py               # Arquivo principal
├── requirements.txt      # Dependências
└── README.md
```

## 🔧 Instalação

1. Clone o repositório:
```bash
git clone [url-do-repositorio]
cd pro-sigma-desktop
```

2. Crie um ambiente virtual:
```bash
python -m venv venv
```

3. Ative o ambiente virtual:
- Windows:
```bash
venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```

4. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 🎮 Como Usar

1. Execute a aplicação:
```bash
python main.py
```

2. Na primeira execução, você será solicitado a inserir uma chave de licença.

3. Após a ativação, você terá acesso à página inicial onde poderá:
   - Importar arquivos Excel ou CSV
   - Selecionar ferramentas de análise disponíveis no seu plano

## 🔑 Sistema de Licenciamento

O Pro Sigma utiliza um sistema de licenciamento baseado em hash para controlar o acesso às funcionalidades. A licença contém:
- **plan**: Tipo de plano (basic, intermediate, pro)
- **expiratedDate**: Data de expiração da licença

### Chaves de Teste Prontas

Você pode usar as seguintes chaves para testar a aplicação (válidas até 31/12/2026):

**Plano Básico:**
```
eyJwbGFuIjogImJhc2ljIiwgImV4cGlyYXRlZERhdGUiOiAiMjAyNi0xMi0zMSJ9fDE1YTA2ZTY4ZjI3NWRmN2Q=
```

**Plano Intermediário:**
```
eyJwbGFuIjogImludGVybWVkaWF0ZSIsICJleHBpcmF0ZWREYXRlIjogIjIwMjYtMTItMzEifXw1NDhhZmIzMGQ2MWRjOWM0
```

**Plano Pro:**
```
eyJwbGFuIjogInBybyIsICJleHBpcmF0ZWREYXRlIjogIjIwMjYtMTItMzEifXw2YTkxNmJkYzljMjk0YjVm
```

Veja todas as chaves e detalhes no arquivo [LICENSE_KEYS_MOCK.md](LICENSE_KEYS_MOCK.md)

## 📦 Planos e Funcionalidades

### Plano Básico
- Análise de Variabilidade
- Process Capability (Cp, Cpk, Pp, Ppk)
- Testes de Hipótese (T, Z, ANOVA, Qui-quadrado)
- Teste de Distribuição
- COV EMS
- Análise de Distribuição
- Analytics

### Plano Intermediário
Todas do Básico +
- Text Analysis
- Testes de Normalidade
- Cartas de Controle
- Dashboard
- Monte Carlo

### Plano Pro
Todas do Intermediário +
- Regressão Simples e Múltipla
- Análise Multivariada (PCA, Fatorial, Cluster)
- StackUp
- DOE (Design of Experiments)
- Space Filling
- Custos de Garantia

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **CustomTkinter**: Interface gráfica moderna
- **Pandas**: Manipulação de dados
- **NumPy**: Cálculos numéricos
- **SciPy**: Análises estatísticas
- **Matplotlib/Seaborn/Plotly**: Visualizações
- **Statsmodels**: Modelos estatísticos
- **Scikit-learn**: Machine Learning

## 📁 Armazenamento Local

A aplicação armazena dados localmente em:
- **Windows**: `C:\Users\[usuario]\.pro_sigma\`
- **Linux/Mac**: `~/.pro_sigma/`

Arquivos armazenados:
- `license.dat`: Licença ativada
- Histórico de análises (em desenvolvimento)

## 🔐 Segurança

- Licenças são validadas usando hash SHA256
- Dados são armazenados localmente (privacidade)
- Nenhuma informação é enviada para servidores externos

## 📝 Desenvolvimento

O projeto está em desenvolvimento ativo. As ferramentas de análise serão implementadas progressivamente.

### Próximos Passos
- [ ] Implementar ferramentas do plano básico
- [ ] Sistema de histórico de análises
- [ ] Exportação de relatórios
- [ ] Implementar ferramentas intermediárias
- [ ] Implementar ferramentas pro

## 📄 Licença

Todos os direitos reservados © 2025 Pro Sigma

## 👤 Autor

Desenvolvido para análise estatística profissional Six Sigma.
