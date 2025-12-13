# 🚀 Executável Pro Sigma

## ✅ Executável Gerado com Sucesso!

O executável do Pro Sigma foi criado e está disponível em:

```
dist/ProSigma/ProSigma.exe
```

## 📁 Estrutura Gerada

```
dist/
└── ProSigma/
    ├── ProSigma.exe          ← Executável principal
    ├── data/                  ← Dados da aplicação
    ├── _internal/             ← Bibliotecas e dependências
    └── ... (outros arquivos)
```

## 🎯 Como Executar

### Opção 1: Duplo Clique
Navegue até `dist/ProSigma/` e dê duplo clique em `ProSigma.exe`

### Opção 2: Script Automático
Execute o arquivo `RUN_ProSigma.bat` na raiz do projeto

### Opção 3: Linha de Comando
```bash
cd dist\ProSigma
.\ProSigma.exe
```

## 📦 Distribuição

Para distribuir o aplicativo:

1. **Copie a pasta completa**: `dist/ProSigma/`
2. **Envie para o usuário final**: Toda a pasta `ProSigma`
3. **Não precisa Python instalado**: O executável é standalone!

### Criar ZIP para distribuição:
```bash
# Via PowerShell
Compress-Archive -Path "dist\ProSigma" -DestinationPath "ProSigma_v1.0.zip"
```

## ⚡ Teste de Performance

### Tempo de Inicialização
- **Desenvolvimento** (python main.py): ~2-3 segundos
- **Executável**: ~5-8 segundos (primeira vez), ~3-5 segundos (próximas)

### Tamanho do Executável
- Verifique o tamanho da pasta: `dist/ProSigma/`
- Geralmente: 300-500 MB (inclui todas as bibliotecas científicas)

### Consumo de Memória
Execute e monitore no Gerenciador de Tarefas:
- RAM inicial: ~150-200 MB
- RAM em uso: ~300-400 MB (depende das análises)

## 🔧 Customização

### Adicionar Ícone
1. Crie ou obtenha um arquivo `icon.ico`
2. Edite `ProSigma.spec` linha do `icon=`:
```python
icon='icon.ico',  # Seu ícone aqui
```
3. Recompile: `pyinstaller ProSigma.spec --clean`

### Gerar Executável Único (One-File)
Edite `ProSigma.spec` e substitua a seção `COLLECT` por:

```python
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ProSigma',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
```

Então recompile: `pyinstaller ProSigma.spec --clean`

⚠️ **Nota**: One-file é mais lento na inicialização (descompacta tudo temporariamente)

## 🐛 Solução de Problemas

### Executável não inicia
1. Execute via terminal para ver erros:
   ```bash
   cd dist\ProSigma
   .\ProSigma.exe
   ```

2. Verifique o log em: `build/ProSigma/warn-ProSigma.txt`

### Falta alguma dependência
Adicione em `ProSigma.spec` na lista `hiddenimports`:
```python
hiddenimports=[
    'customtkinter',
    'pandas',
    'seu_modulo_faltando',
],
```

### Antivírus bloqueia
- Normal para executáveis Python empacotados
- Adicione exceção no antivírus
- Em produção, assine digitalmente o executável

## 📊 Comparação de Desempenho

| Métrica | Python Script | Executável |
|---------|---------------|------------|
| Tempo de início | 2-3s | 5-8s |
| Tamanho | 2 MB | 400 MB |
| Requer Python | ✅ Sim | ❌ Não |
| Portabilidade | Baixa | Alta |
| Distribuição | Complexa | Simples |

## 🔄 Recompilar

Quando fizer alterações no código:

```bash
pyinstaller ProSigma.spec --clean
```

---

**Gerado em:** 12/12/2025
**PyInstaller:** 6.17.0
**Python:** 3.12.4
