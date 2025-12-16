"""
Interface gráfica para Análise de Redes Neurais
Suporta MLP com métodos Holdout e K-Fold
"""

import customtkinter as ctk
from tkinter import messagebox, filedialog
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import pickle
import json
from pathlib import Path
from datetime import datetime

from src.utils.lazy_imports import get_pandas, get_numpy
from src.utils.ui_components import add_chart_export_button
from src.analytics.neural_network.neural_network_utils import (
    train_neural_network_holdout,
    train_neural_network_kfold
)


class NeuralNetworkWindow(ctk.CTkToplevel):
    """Janela de Análise de Redes Neurais"""
    
    def __init__(self, parent, data):
        super().__init__(parent)
        
        self.title("Redes Neurais - MLP")
        self.geometry("1400x900")
        self.state('zoomed')
        
        self.data = data
        self.results = None
        
        self.configure(fg_color="#2b2b2b")
        
        # Protocolo de fechamento para não fechar programa todo
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        
        self._create_widgets()
        self._populate_columns()
    
    def _create_widgets(self):
        """Cria os widgets da interface"""
        
        # Container principal com scroll
        self.main_container = ctk.CTkScrollableFrame(
            self,
            scrollbar_button_color="gray30",
            scrollbar_button_hover_color="gray40"
        )
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Título
        title = ctk.CTkLabel(
            self.main_container,
            text="🧠 Análise de Redes Neurais (MLP)",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=(0, 5))
        
        # Descrição
        desc = ctk.CTkLabel(
            self.main_container,
            text="Multi-Layer Perceptron para Classificação e Regressão",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        desc.pack(pady=(0, 10))
        
        # Frame de configuração
        config_frame = ctk.CTkFrame(self.main_container)
        config_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(
            config_frame,
            text="⚙️ Configuração da Rede Neural",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(15, 10), padx=20, anchor="w")
        
        # Container para seleção de variáveis
        vars_container = ctk.CTkFrame(config_frame, fg_color="transparent")
        vars_container.pack(fill="x", padx=20, pady=(0, 10))
        
        # === Variáveis X (esquerda) ===
        x_frame = ctk.CTkFrame(vars_container)
        x_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        ctk.CTkLabel(
            x_frame,
            text="Variáveis Independentes (X):",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", pady=(5, 3), padx=10)
        
        self.x_columns_frame = ctk.CTkScrollableFrame(x_frame, height=120)
        self.x_columns_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # === Variável Y (direita) ===
        y_frame = ctk.CTkFrame(vars_container)
        y_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        ctk.CTkLabel(
            y_frame,
            text="Variável Dependente (Y) - Alvo:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", pady=(5, 3), padx=10)
        
        self.y_column_frame = ctk.CTkScrollableFrame(y_frame, height=120)
        self.y_column_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # === Colunas Categóricas ===
        categorical_frame = ctk.CTkFrame(config_frame)
        categorical_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            categorical_frame,
            text="📋 Seleção de Colunas Categóricas (para Encoding):",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", pady=(10, 5), padx=10)
        
        ctk.CTkLabel(
            categorical_frame,
            text="Marque as colunas que contêm dados categóricos (texto, categorias)",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        ).pack(anchor="w", pady=(0, 5), padx=10)
        
        self.categorical_columns_frame = ctk.CTkScrollableFrame(categorical_frame, height=100)
        self.categorical_columns_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # === Opções ===
        options_frame = ctk.CTkFrame(config_frame)
        options_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            options_frame,
            text="Opções de Treinamento:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        options_grid = ctk.CTkFrame(options_frame, fg_color="transparent")
        options_grid.pack(fill="x", padx=10, pady=(0, 10))
        
        # Método
        ctk.CTkLabel(
            options_grid,
            text="Método de Validação:",
            font=ctk.CTkFont(size=11, weight="bold")
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.method_var = tk.StringVar(value="holdout")
        method_combo = ctk.CTkComboBox(
            options_grid,
            variable=self.method_var,
            values=["holdout", "kfold"],
            width=150,
            state="readonly",
            command=self._on_method_change
        )
        method_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Função de ativação
        ctk.CTkLabel(
            options_grid,
            text="Função de Ativação:",
            font=ctk.CTkFont(size=11, weight="bold")
        ).grid(row=0, column=2, padx=(20, 5), pady=5, sticky="w")
        
        self.activation_var = tk.StringVar(value="relu")
        activation_combo = ctk.CTkComboBox(
            options_grid,
            variable=self.activation_var,
            values=["relu", "tanh", "logistic", "identity"],
            width=150,
            state="readonly"
        )
        activation_combo.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        
        # Test Size (Holdout)
        ctk.CTkLabel(
            options_grid,
            text="Tamanho do Teste (%):",
            font=ctk.CTkFont(size=11, weight="bold")
        ).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        
        self.test_size_var = tk.DoubleVar(value=30.0)
        self.test_size_entry = ctk.CTkEntry(
            options_grid,
            textvariable=self.test_size_var,
            width=150
        )
        self.test_size_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # N Folds (K-Fold)
        ctk.CTkLabel(
            options_grid,
            text="Número de Folds:",
            font=ctk.CTkFont(size=11, weight="bold")
        ).grid(row=1, column=2, padx=(20, 5), pady=5, sticky="w")
        
        self.n_folds_var = tk.IntVar(value=5)
        self.n_folds_entry = ctk.CTkEntry(
            options_grid,
            textvariable=self.n_folds_var,
            width=150,
            state="disabled"
        )
        self.n_folds_entry.grid(row=1, column=3, padx=5, pady=5, sticky="w")
        
        # Máximo de iterações
        ctk.CTkLabel(
            options_grid,
            text="Máximo de Iterações:",
            font=ctk.CTkFont(size=11, weight="bold")
        ).grid(row=2, column=0, padx=5, pady=5, sticky="w")
        
        self.max_iter_var = tk.IntVar(value=500)
        max_iter_entry = ctk.CTkEntry(
            options_grid,
            textvariable=self.max_iter_var,
            width=150
        )
        max_iter_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        # Container para botões
        buttons_container = ctk.CTkFrame(config_frame, fg_color="transparent")
        buttons_container.pack(pady=15)
        
        # Botão treinar
        train_btn = ctk.CTkButton(
            buttons_container,
            text="🚀 Treinar Rede Neural",
            command=self._train_network,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#1f6aa5",
            hover_color="#144870",
            height=40,
            width=200
        )
        train_btn.pack(side="left", padx=5)
        
        # Botão carregar modelo
        load_btn = ctk.CTkButton(
            buttons_container,
            text="📂 Carregar Modelo",
            command=self._load_model,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#2ecc71",
            hover_color="#27ae60",
            height=40,
            width=180
        )
        load_btn.pack(side="left", padx=5)
        
        # Checkbox para re-treinar modelo ao carregar
        self.retrain_on_load_var = tk.BooleanVar(value=False)
        retrain_check = ctk.CTkCheckBox(
            buttons_container,
            text="🔄 Re-treinar com novos dados (Fine-Tuning)",
            variable=self.retrain_on_load_var,
            font=ctk.CTkFont(size=11)
        )
        retrain_check.pack(side="left", padx=10)
        
        # Label explicativo
        retrain_info = ctk.CTkLabel(
            config_frame,
            text="💡 Marque a opção acima para refinar um modelo existente com novos dados",
            font=ctk.CTkFont(size=9),
            text_color="gray"
        )
        retrain_info.pack(pady=(0, 5))
        
        # Frame de loading
        self.loading_frame = ctk.CTkFrame(self.main_container)
        
        # Frame de resultados
        self.results_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.results_frame.pack(fill="both", expand=True, pady=(0, 10))
    
    def _show_loading(self, show=True):
        """Mostra ou esconde o spinner de loading"""
        if show:
            # Esconde resultados
            self.results_frame.pack_forget()
            
            # Mostra loading
            self.loading_frame.pack(fill="both", expand=True, pady=50)
            
            # Limpa loading frame
            for widget in self.loading_frame.winfo_children():
                widget.destroy()
            
            # Ícone de loading
            loading_label = ctk.CTkLabel(
                self.loading_frame,
                text="⏳",
                font=ctk.CTkFont(size=60)
            )
            loading_label.pack(pady=(20, 10))
            
            # Texto
            text_label = ctk.CTkLabel(
                self.loading_frame,
                text="Treinando Rede Neural...",
                font=ctk.CTkFont(size=18, weight="bold")
            )
            text_label.pack(pady=(0, 10))
            
            # Subtítulo
            subtext = ctk.CTkLabel(
                self.loading_frame,
                text="Otimizando hiperparâmetros com GridSearchCV\nIsso pode levar alguns segundos...",
                font=ctk.CTkFont(size=12),
                text_color="gray"
            )
            subtext.pack(pady=(0, 20))
            
            # Progress bar indeterminada
            progress = ctk.CTkProgressBar(self.loading_frame, mode="indeterminate", width=300)
            progress.pack(pady=10)
            progress.start()
            
            # Força atualização da UI
            self.update()
        else:
            # Esconde loading
            self.loading_frame.pack_forget()
            
            # Mostra resultados
            self.results_frame.pack(fill="both", expand=True, pady=(0, 10))
    
    def _on_method_change(self, value):
        """Atualiza campos quando método muda"""
        if value == "kfold":
            self.test_size_entry.configure(state="disabled")
            self.n_folds_entry.configure(state="normal")
        else:
            self.test_size_entry.configure(state="normal")
            self.n_folds_entry.configure(state="disabled")
    
    def _populate_columns(self):
        """Popula checkboxes para seleção de colunas"""
        if self.data is None:
            return
        
        pd = get_pandas()
        
        # Converte datetime
        for col in self.data.columns:
            if pd.api.types.is_datetime64_any_dtype(self.data[col]):
                self.data[col] = self.data[col].astype(str)
        
        # Todas as colunas (numéricas e categóricas)
        all_cols = self.data.columns.tolist()
        
        if len(all_cols) < 2:
            messagebox.showerror(
                "Erro",
                "É necessário ter pelo menos 2 colunas (1 X + 1 Y) para análise de Redes Neurais."
            )
            self.destroy()
            return
        
        # Checkboxes para X
        self.x_column_vars = {}
        for col in all_cols:
            var = tk.BooleanVar(value=True)
            check = ctk.CTkCheckBox(
                self.x_columns_frame,
                text=col,
                variable=var
            )
            check.pack(anchor="w", padx=5, pady=2)
            self.x_column_vars[col] = var
        
        # Radio buttons para Y (apenas uma)
        self.y_column_var = tk.StringVar(value=all_cols[-1])
        for col in all_cols:
            radio = ctk.CTkRadioButton(
                self.y_column_frame,
                text=col,
                variable=self.y_column_var,
                value=col
            )
            radio.pack(anchor="w", padx=5, pady=2)
        
        # Checkboxes para colunas categóricas
        # Detecta automaticamente colunas com tipo object ou category
        self.categorical_column_vars = {}
        auto_categorical = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in all_cols:
            var = tk.BooleanVar(value=(col in auto_categorical))
            check = ctk.CTkCheckBox(
                self.categorical_columns_frame,
                text=col,
                variable=var
            )
            check.pack(anchor="w", padx=5, pady=2)
            self.categorical_column_vars[col] = var
    
    def _train_network(self):
        """Treina a rede neural"""
        
        # Obtém colunas selecionadas
        x_columns = [col for col, var in self.x_column_vars.items() if var.get()]
        y_column = self.y_column_var.get()
        
        if len(x_columns) < 1:
            messagebox.showwarning("Aviso", "Selecione pelo menos 1 coluna X")
            return
        
        if y_column in x_columns:
            messagebox.showwarning("Aviso", "A variável Y não pode estar em X")
            return
        
        # Obtém colunas categóricas selecionadas pelo usuário
        categorical_cols = [col for col, var in self.categorical_column_vars.items() if var.get() and col in x_columns]
        
        try:
            # Mostra loading
            self._show_loading(True)
            
            method = self.method_var.get()
            activation = self.activation_var.get()
            max_iter = self.max_iter_var.get()
            
            if method == "holdout":
                test_size = self.test_size_var.get() / 100.0
                self.results = train_neural_network_holdout(
                    self.data,
                    x_columns,
                    y_column,
                    categorical_cols,
                    activation,
                    test_size,
                    max_iter
                )
            else:  # kfold
                n_folds = self.n_folds_var.get()
                self.results = train_neural_network_kfold(
                    self.data,
                    x_columns,
                    y_column,
                    categorical_cols,
                    activation,
                    n_folds,
                    max_iter
                )
            
            # Esconde loading
            self._show_loading(False)
            
            # Exibe resultados
            self._display_results()
            
        except Exception as e:
            # Esconde loading em caso de erro
            self._show_loading(False)
            messagebox.showerror("Erro", f"Erro ao treinar rede neural:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def _display_results(self):
        """Exibe resultados da análise"""
        
        # Limpa resultados anteriores
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        if not self.results:
            return
        
        # Título dos resultados
        if self.results.get('loaded_model', False):
            if self.results['model_info'].get('retrained', False):
                title_text = f"📊 Resultados - Modelo Re-Treinado 🔄 ({self.results.get('model_file', 'modelo.pkl')})"
            else:
                title_text = f"📊 Resultados - Modelo Carregado ({self.results.get('model_file', 'modelo.pkl')})"
        else:
            title_text = "📊 Resultados da Análise"
        
        result_title = ctk.CTkLabel(
            self.results_frame,
            text=title_text,
            font=ctk.CTkFont(size=18, weight="bold")
        )
        result_title.pack(pady=(10, 15))
        
        # Container para informações e métricas lado a lado
        info_metrics_container = ctk.CTkFrame(self.results_frame, fg_color="transparent")
        info_metrics_container.pack(fill="x", padx=10, pady=10)
        
        # Informações do modelo (esquerda)
        self._display_model_info(info_metrics_container)
        
        # Métricas (direita)
        self._display_metrics(info_metrics_container)
        
        # Feature Importance
        self._display_feature_importance()
        
        # Gráficos
        self._display_charts()
    
    def _display_model_info(self, parent):
        """Exibe informações do modelo"""
        info_frame = ctk.CTkFrame(parent)
        info_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        ctk.CTkLabel(
            info_frame,
            text="ℹ️ Informações do Modelo",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5), padx=10, anchor="w")
        
        model_info = self.results['model_info']
        
        # Tabela com informações principais
        headers = ['Parâmetro', 'Valor']
        data_rows = [
            ['Arquitetura', str(model_info['hidden_layers'])],
            ['Camadas Ocultas', str(model_info['n_layers'])],
            ['Iterações', str(model_info['n_iter'])],
            ['Loss Final', f"{model_info['loss']:.6f}"],
            ['Tipo', 'Classificação' if self.results['is_classification'] else 'Regressão']
        ]
        
        # Adiciona informação de re-treino se aplicável
        if model_info.get('retrained', False):
            data_rows.append(['Status', '🔄 Re-Treinado (Fine-Tuned)'])
            data_rows.append(['Iterações Extras', str(model_info.get('retrain_iterations', 0))])
        
        # Adiciona hiperparâmetros
        for key, value in model_info['best_params'].items():
            param_name = key.replace('_', ' ').title()
            data_rows.append([param_name, str(value)])
        
        self._create_compact_table(info_frame, headers, data_rows)
        
        # Botão para salvar modelo
        save_model_btn = ctk.CTkButton(
            info_frame,
            text="💾 Salvar Modelo Treinado",
            command=self._save_model,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#9b59b6",
            hover_color="#8e44ad",
            height=35,
            width=180
        )
        save_model_btn.pack(pady=(10, 10))
    
    def _save_model(self):
        """Salva o modelo treinado em arquivo"""
        if not self.results:
            messagebox.showwarning("Aviso", "Nenhum modelo treinado para salvar")
            return
        
        # Dialog para escolher onde salvar
        file_path = filedialog.asksaveasfilename(
            title="Salvar Modelo de Rede Neural",
            defaultextension=".pkl",
            filetypes=[
                ("Modelo ProSigma", "*.pkl"),
                ("Todos os arquivos", "*.*")
            ],
            initialfile=f"neural_network_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        )
        
        if not file_path:
            return
        
        try:
            # Obtém colunas selecionadas
            x_columns = [col for col, var in self.x_column_vars.items() if var.get()]
            y_column = self.y_column_var.get()
            
            # Obtém colunas categóricas selecionadas pelo usuário
            categorical_cols = [col for col, var in self.categorical_column_vars.items() if var.get() and col in x_columns]
            
            # Prepara dados para salvar
            model_data = {
                'model': self.results['model'],
                'preprocessor': self.results.get('preprocessor'),
                'is_classification': self.results['is_classification'],
                'feature_names': self.results['feature_names'],
                'x_columns': x_columns,
                'y_column': y_column,
                'categorical_cols': categorical_cols,
                'model_info': self.results['model_info'],
                'feature_importance': self.results['feature_importance'],
                'method': self.method_var.get(),
                'activation': self.activation_var.get(),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'prosigma_version': '1.0.0'
            }
            
            # Salva usando pickle
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Salva metadados em JSON (para fácil leitura)
            metadata_path = str(Path(file_path).with_suffix('.json'))
            metadata = {
                'tipo': 'Classificação' if model_data['is_classification'] else 'Regressão',
                'arquitetura': str(model_data['model_info']['hidden_layers']),
                'n_camadas': model_data['model_info']['n_layers'],
                'ativacao': model_data['activation'],
                'metodo': model_data['method'],
                'features_entrada': x_columns,
                'variavel_alvo': y_column,
                'features_categoricas': categorical_cols,
                'data_treinamento': model_data['timestamp'],
                'versao_prosigma': model_data['prosigma_version']
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
            
            messagebox.showinfo(
                "Sucesso",
                f"Modelo salvo com sucesso!\n\n"
                f"Arquivo: {Path(file_path).name}\n"
                f"Metadados: {Path(metadata_path).name}\n\n"
                f"Tipo: {metadata['tipo']}\n"
                f"Arquitetura: {metadata['arquitetura']}"
            )
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao salvar modelo:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def _load_model(self):
        """Carrega um modelo salvo e faz análise completa com dados atuais"""
        if self.data is None:
            messagebox.showerror(
                "Erro",
                "Nenhum dado carregado!\n\n"
                "Por favor, carregue um arquivo de dados antes de usar um modelo pré-treinado."
            )
            return
        
        file_path = filedialog.askopenfilename(
            title="Carregar Modelo de Rede Neural",
            filetypes=[
                ("Modelo ProSigma", "*.pkl"),
                ("Todos os arquivos", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # Mostra loading
            self._show_loading(True)
            
            # Carrega o modelo
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Verifica versão (compatibilidade)
            if 'prosigma_version' not in model_data:
                self._show_loading(False)
                response = messagebox.askyesno(
                    "Aviso",
                    "Este modelo foi salvo em uma versão antiga.\n"
                    "Pode haver problemas de compatibilidade.\n\n"
                    "Deseja continuar?"
                )
                if not response:
                    return
                self._show_loading(True)
            
            # Verifica se dados têm as colunas necessárias
            pd = get_pandas()
            missing_cols = set(model_data['x_columns']) - set(self.data.columns)
            if missing_cols:
                self._show_loading(False)
                messagebox.showerror(
                    "Erro - Colunas Faltando",
                    f"Os dados atuais não têm as colunas necessárias para este modelo.\n\n"
                    f"Colunas faltando:\n{', '.join(missing_cols)}\n\n"
                    f"Colunas necessárias:\n{', '.join(model_data['x_columns'])}\n\n"
                    f"Por favor, carregue um arquivo com as colunas corretas."
                )
                return
            
            # Verifica se Y existe nos dados (para comparação)
            has_y = model_data['y_column'] in self.data.columns
            
            # Verifica se deve re-treinar
            should_retrain = self.retrain_on_load_var.get() and has_y
            
            if should_retrain:
                # RE-TREINAMENTO (Fine-Tuning)
                self._show_loading(False)
                response = messagebox.askyesno(
                    "Re-treinar Modelo",
                    f"🔄 Você optou por re-treinar o modelo!\n\n"
                    f"O modelo atual será refinado (fine-tuned) com os novos dados.\n\n"
                    f"Dados atuais: {len(self.data)} observações\n"
                    f"Features: {len(model_data['x_columns'])}\n"
                    f"Arquitetura atual: {model_data['model_info']['hidden_layers']}\n\n"
                    f"Isso pode levar alguns segundos. Continuar?"
                )
                
                if not response:
                    return
                
                self._show_loading(True)
                
                # Re-treina o modelo
                y_pred, updated_model_info = self._retrain_model(
                    model_data, 
                    self.data[model_data['x_columns']].copy(),
                    self.data[model_data['y_column']].values
                )
                
                # Atualiza model_info com novos valores
                model_data['model_info'].update(updated_model_info)
                
            else:
                # Prepara dados para predição
                X = self.data[model_data['x_columns']].copy()
                
                # Transforma usando o preprocessor salvo
                if model_data['preprocessor'] is not None:
                    X_transformed = model_data['preprocessor'].transform(X)
                else:
                    X_transformed = X
                
                # Faz predições
                y_pred = model_data['model'].predict(X_transformed)
            
            # Monta results no mesmo formato que o treinamento
            self.results = {
                'model': model_data['model'],
                'preprocessor': model_data.get('preprocessor'),
                'is_classification': model_data['is_classification'],
                'feature_names': model_data['feature_names'],
                'feature_importance': model_data['feature_importance'],
                'model_info': model_data['model_info'],
                'y_pred': y_pred,
                'loaded_model': True,
                'model_file': Path(file_path).name
            }
            
            # Se tiver Y nos dados, calcula métricas
            if has_y:
                y_true = self.data[model_data['y_column']].values
                self.results['y_true'] = y_true
                
                # Calcula métricas
                if model_data['is_classification']:
                    from src.analytics.neural_network.neural_network_utils import calculate_metrics_classification
                    y_pred_proba = model_data['model'].predict_proba(X_transformed) if hasattr(model_data['model'], 'predict_proba') else None
                    self.results['metrics'] = calculate_metrics_classification(y_true, y_pred, y_pred_proba)
                else:
                    from src.analytics.neural_network.neural_network_utils import calculate_metrics_regression
                    self.results['metrics'] = calculate_metrics_regression(y_true, y_pred)
            else:
                # Sem Y, apenas predições
                self.results['y_true'] = None
                self.results['metrics'] = None
            
            # Atualiza interface para refletir modelo carregado
            self.method_var.set(model_data.get('method', 'holdout'))
            self.activation_var.set(model_data['activation'])
            
            # Esconde loading
            self._show_loading(False)
            
            # Exibe resultados completos
            self._display_results()
            
            # Mensagem de sucesso
            if has_y:
                if should_retrain:
                    status_msg = (
                        f"✅ Modelo re-treinado com sucesso!\n\n"
                        f"Arquivo original: {Path(file_path).name}\n"
                        f"Tipo: {'Classificação' if model_data['is_classification'] else 'Regressão'}\n"
                        f"Arquitetura: {model_data['model_info']['hidden_layers']}\n\n"
                        f"🔄 O modelo foi refinado com {len(y_pred)} novas observações.\n"
                        f"Iterações adicionais: {model_data['model_info'].get('retrain_iterations', 'N/A')}\n"
                        f"Loss após re-treino: {model_data['model_info']['loss']:.6f}\n\n"
                        f"💡 Dica: Salve este modelo atualizado para preservar o aprendizado!"
                    )
                else:
                    status_msg = (
                        f"✅ Modelo carregado com sucesso!\n\n"
                        f"Arquivo: {Path(file_path).name}\n"
                        f"Tipo: {'Classificação' if model_data['is_classification'] else 'Regressão'}\n"
                        f"Arquitetura: {model_data['model_info']['hidden_layers']}\n\n"
                        f"Predições realizadas para {len(y_pred)} observações.\n"
                        f"Métricas calculadas com base nos dados atuais.\n\n"
                        f"💡 Marque 'Re-treinar' para refinar o modelo com estes dados."
                    )
                
                messagebox.showinfo("Modelo Carregado", status_msg)
            else:
                messagebox.showinfo(
                    "Modelo Carregado",
                    f"✅ Modelo carregado com sucesso!\n\n"
                    f"Arquivo: {Path(file_path).name}\n"
                    f"Tipo: {'Classificação' if model_data['is_classification'] else 'Regressão'}\n"
                    f"Arquitetura: {model_data['model_info']['hidden_layers']}\n\n"
                    f"Predições realizadas para {len(y_pred)} observações.\n\n"
                    f"⚠️ Nota: Coluna '{model_data['y_column']}' não encontrada nos dados.\n"
                    f"Apenas predições serão mostradas (sem métricas de comparação)."
                )
            
        except Exception as e:
            self._show_loading(False)
            messagebox.showerror("Erro", f"Erro ao carregar modelo:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def _retrain_model(self, model_data, X, y):
        """Re-treina (fine-tune) o modelo com novos dados"""
        try:
            # Transforma features se necessário
            if model_data['preprocessor'] is not None:
                X_transformed = model_data['preprocessor'].transform(X)
            else:
                X_transformed = X
            
            # Obtém modelo atual
            model = model_data['model']
            
            # Guarda número de iterações atual
            initial_iterations = model.n_iter_
            
            # Configura para treino incremental (warm_start permite continuar treinamento)
            model.warm_start = True
            
            # Define número adicional de iterações para fine-tuning
            # Usar menos iterações que o treinamento inicial (10-20% do max_iter)
            additional_iterations = max(50, int(self.max_iter_var.get() * 0.2))
            model.max_iter = additional_iterations
            
            # Re-treina com novos dados
            model.fit(X_transformed, y)
            
            # Calcula nova loss
            new_loss = model.loss_
            
            # Faz predições
            y_pred = model.predict(X_transformed)
            
            # Atualiza informações do modelo
            updated_info = {
                'n_iter': model.n_iter_,
                'loss': new_loss,
                'retrain_iterations': model.n_iter_ - initial_iterations,
                'retrained': True
            }
            
            # Recalcula feature importance com novos dados
            from src.analytics.neural_network.neural_network_utils import calculate_feature_importance
            new_importance = calculate_feature_importance(model, X_transformed, y, model_data['feature_names'])
            model_data['feature_importance'] = new_importance
            
            return y_pred, updated_info
            
        except Exception as e:
            raise Exception(f"Erro ao re-treinar modelo: {str(e)}")
    
    def _create_compact_table(self, parent, headers, data_rows):
        """Cria uma tabela compacta estilo Minitab"""
        table_frame = ctk.CTkFrame(parent, fg_color="#f0f0f0")
        table_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        # Header
        header_frame = ctk.CTkFrame(table_frame, fg_color="#2E86DE", height=30)
        header_frame.pack(fill="x", padx=2, pady=2)
        header_frame.pack_propagate(False)
        
        for i, header in enumerate(headers):
            weight = 1 if i == 0 else 1
            header_label = ctk.CTkLabel(
                header_frame,
                text=header,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color="white"
            )
            header_label.pack(side="left", expand=True, fill="both", padx=5)
        
        # Data rows
        for row in data_rows:
            row_frame = ctk.CTkFrame(table_frame, fg_color="white", height=25)
            row_frame.pack(fill="x", padx=2, pady=1)
            row_frame.pack_propagate(False)
            
            for i, cell in enumerate(row):
                cell_label = ctk.CTkLabel(
                    row_frame,
                    text=str(cell),
                    font=ctk.CTkFont(size=10),
                    text_color="black",
                    anchor="w" if i == 0 else "e"
                )
                cell_label.pack(side="left", expand=True, fill="both", padx=5)
    
    def _display_metrics(self, parent):
        """Exibe métricas de performance"""
        metrics_frame = ctk.CTkFrame(parent)
        metrics_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        ctk.CTkLabel(
            metrics_frame,
            text="📈 Métricas de Performance",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 10), padx=10, anchor="w")
        
        # Se for modelo carregado sem métricas
        if self.results.get('loaded_model', False) and self.results.get('metrics') is None:
            no_metrics_label = ctk.CTkLabel(
                metrics_frame,
                text="⚠️ Métricas não disponíveis\n\n"
                     "Os dados atuais não contêm a\n"
                     "coluna alvo para comparação.\n\n"
                     "Apenas predições foram geradas.",
                font=ctk.CTkFont(size=11),
                text_color="orange",
                justify="center"
            )
            no_metrics_label.pack(pady=50)
            return
        
        method = self.method_var.get()
        is_classification = self.results['is_classification']
        
        if method == "holdout" and not self.results.get('loaded_model', False):
            self._display_holdout_metrics(metrics_frame, is_classification)
        else:
            self._display_kfold_metrics(metrics_frame, is_classification)
    
    def _display_holdout_metrics(self, parent, is_classification):
        """Exibe métricas do método Holdout"""
        metrics_train = self.results['metrics_train']
        metrics_test = self.results['metrics_test']
        
        if is_classification:
            headers = ['Métrica', 'Treino', 'Teste']
            data_rows = [
                ['Acurácia', f"{metrics_train['accuracy']:.4f}", f"{metrics_test['accuracy']:.4f}"],
                ['Precisão', f"{metrics_train['precision']:.4f}", f"{metrics_test['precision']:.4f}"],
                ['Recall', f"{metrics_train['recall']:.4f}", f"{metrics_test['recall']:.4f}"],
                ['F1-Score', f"{metrics_train['f1_score']:.4f}", f"{metrics_test['f1_score']:.4f}"],
                ['ROC-AUC', f"{metrics_train['roc_auc']:.4f}", f"{metrics_test['roc_auc']:.4f}"]
            ]
        else:
            headers = ['Métrica', 'Treino', 'Teste']
            data_rows = [
                ['MSE', f"{metrics_train['mse']:.6f}", f"{metrics_test['mse']:.6f}"],
                ['RMSE', f"{metrics_train['rmse']:.6f}", f"{metrics_test['rmse']:.6f}"],
                ['R²', f"{metrics_train['r2']:.4f}", f"{metrics_test['r2']:.4f}"],
                ['Média', f"{metrics_train['mean']:.4f}", f"{metrics_test['mean']:.4f}"],
                ['Desvio Padrão', f"{metrics_train['std']:.4f}", f"{metrics_test['std']:.4f}"]
            ]
        
        self._create_compact_table(parent, headers, data_rows)
    
    def _display_kfold_metrics(self, parent, is_classification):
        """Exibe métricas do método K-Fold ou modelo carregado"""
        metrics = self.results['metrics']
        
        if is_classification:
            headers = ['Métrica', 'Média', 'Desvio Padrão']
            data_rows = [
                ['Acurácia', f"{metrics['accuracy']:.4f}", f"{metrics.get('accuracy_std', 0):.4f}"],
                ['Precisão', f"{metrics['precision']:.4f}", f"{metrics.get('precision_std', 0):.4f}"],
                ['Recall', f"{metrics['recall']:.4f}", f"{metrics.get('recall_std', 0):.4f}"],
                ['F1-Score', f"{metrics['f1_score']:.4f}", f"{metrics.get('f1_score_std', 0):.4f}"],
                ['ROC-AUC', f"{metrics['roc_auc']:.4f}", f"{metrics.get('roc_auc_std', 0):.4f}"]
            ]
        else:
            # Para modelo carregado sem std, mostra apenas valor
            if self.results.get('loaded_model', False):
                headers = ['Métrica', 'Valor', '']
            else:
                headers = ['Métrica', 'Média', 'Desvio Padrão']
            
            data_rows = [
                ['MSE', f"{metrics['mse']:.6f}", f"{metrics.get('mse_std', 0):.6f}"],
                ['RMSE', f"{metrics['rmse']:.6f}", f"{metrics.get('rmse_std', 0):.6f}"],
                ['R²', f"{metrics['r2']:.4f}", f"{metrics.get('r2_std', 0):.4f}"]
            ]
        
        self._create_compact_table(parent, headers, data_rows)
    
    def _display_feature_importance(self):
        """Exibe importância das features"""
        importance_frame = ctk.CTkFrame(self.results_frame)
        importance_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            importance_frame,
            text="🎯 Importância das Variáveis",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 10), padx=10, anchor="w")
        
        feature_importance = self.results['feature_importance']
        
        headers = ['Variável', 'Importância']
        data_rows = [[var, f"{imp:.6f}"] for var, imp in feature_importance.items()]
        
        self._create_compact_table(importance_frame, headers, data_rows)
    
    def _display_charts(self):
        """Exibe gráficos"""
        charts_frame = ctk.CTkFrame(self.results_frame)
        charts_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            charts_frame,
            text="📊 Visualizações",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 10), padx=10, anchor="w")
        
        # Container para gráficos
        charts_container = ctk.CTkFrame(charts_frame, fg_color="transparent")
        charts_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Gráfico 1: Predições
        self._create_predictions_chart(charts_container)
        
        # Gráfico 2: Arquitetura da Rede Neural
        self._create_network_architecture_chart(charts_container)
        
        # Gráfico 3: Feature Importance
        self._create_importance_chart(charts_container)
        
        # Gráfico 4: Confusion Matrix (se classificação)
        if self.results['is_classification']:
            self._create_confusion_matrix_chart(charts_container)
    
    def _create_predictions_chart(self, parent):
        """Cria gráfico de predições"""
        chart_frame = ctk.CTkFrame(parent)
        chart_frame.pack(fill="both", expand=True, pady=10)
        
        ctk.CTkLabel(
            chart_frame,
            text="📈 Real vs Predito",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(pady=(10, 5), padx=10, anchor="w")
        
        fig = Figure(figsize=(8, 5), dpi=100, facecolor='white')
        ax = fig.add_subplot(111)
        
        method = self.method_var.get()
        
        # Modelo carregado ou K-Fold
        if self.results.get('loaded_model', False) or method == "kfold":
            y_true = self.results.get('y_true')
            y_pred = self.results['y_pred']
        else:  # Holdout
            y_true = self.results['y_test']
            y_pred = self.results['y_pred_test']
        
        # Se não tiver y_true, só mostra predições
        if y_true is None:
            np = get_numpy()
            y_pred_array = np.array(y_pred)
            indices = range(len(y_pred_array))
            
            ax.plot(indices, y_pred_array, 's-', color='#ff7f0e', linewidth=2, markersize=5,
                    label='Predições', alpha=0.8)
            ax.set_xlabel('Observação', fontsize=11, fontweight='bold')
            ax.set_ylabel('Valor Predito', fontsize=11, fontweight='bold')
            ax.set_title('Predições do Modelo', fontsize=13, fontweight='bold', pad=10)
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
        else:
            # Ordena por y_true
            np = get_numpy()
            y_true_array = np.array(y_true)
            y_pred_array = np.array(y_pred)
            sorted_indices = np.argsort(y_true_array)
            y_true_sorted = y_true_array[sorted_indices]
            y_pred_sorted = y_pred_array[sorted_indices]
            
            indices = range(len(y_true_sorted))
            ax.plot(indices, y_true_sorted, 'o-', color='#2ca02c', linewidth=2, markersize=5,
                    label='Real', alpha=0.8)
            ax.plot(indices, y_pred_sorted, 's--', color='#ff7f0e', linewidth=2, markersize=5,
                    label='Predito', alpha=0.8)
            
            ax.set_xlabel('Observação (ordenada)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Valor', fontsize=11, fontweight='bold')
            ax.set_title('Comparação Real vs Predito', fontsize=13, fontweight='bold', pad=10)
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
        
        fig.tight_layout()
        
        canvas_frame = ctk.CTkFrame(chart_frame, fg_color="#ffffff", corner_radius=5)
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        canvas = FigureCanvasTkAgg(fig, canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
        export_btn = add_chart_export_button(chart_frame, fig, "neural_network_predictions")
        export_btn.pack(pady=(0, 10))
    
    def _create_network_architecture_chart(self, parent):
        """Cria gráfico da arquitetura da rede neural"""
        chart_frame = ctk.CTkFrame(parent)
        chart_frame.pack(fill="both", expand=True, pady=10)
        
        ctk.CTkLabel(
            chart_frame,
            text="🧠 Arquitetura da Rede Neural",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(pady=(10, 5), padx=10, anchor="w")
        
        fig = Figure(figsize=(10, 5), dpi=100, facecolor='white')
        ax = fig.add_subplot(111)
        
        # Obtém arquitetura
        model_info = self.results['model_info']
        hidden_layers = model_info['hidden_layers']
        
        # Converte tupla para lista
        if isinstance(hidden_layers, tuple):
            layer_sizes = list(hidden_layers)
        else:
            layer_sizes = [hidden_layers]
        
        # Adiciona camada de entrada e saída
        n_features = len(self.results['feature_names'])
        n_output = 1  # Simplificado
        
        all_layers = [n_features] + layer_sizes + [n_output]
        n_layers = len(all_layers)
        
        # Configurações de desenho
        max_neurons = max(all_layers)
        layer_spacing = 1.5
        neuron_radius = 0.15
        
        # Desenha conexões primeiro (ficam atrás)
        for i in range(n_layers - 1):
            n_current = all_layers[i]
            n_next = all_layers[i + 1]
            
            x_current = i * layer_spacing
            x_next = (i + 1) * layer_spacing
            
            # Limita número de conexões desenhadas para não poluir
            max_connections = min(n_current * n_next, 100)
            step_current = max(1, n_current // 10)
            step_next = max(1, n_next // 10)
            
            for j in range(0, n_current, step_current):
                y_current = (max_neurons - n_current) / 2 + j
                for k in range(0, n_next, step_next):
                    y_next = (max_neurons - n_next) / 2 + k
                    ax.plot([x_current, x_next], [y_current, y_next],
                           'gray', alpha=0.15, linewidth=0.3, zorder=1)
        
        # Desenha neurônios
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        for i, n_neurons in enumerate(all_layers):
            x = i * layer_spacing
            color = colors[min(i, len(colors) - 1)]
            
            # Centraliza verticalmente
            y_start = (max_neurons - n_neurons) / 2
            
            for j in range(n_neurons):
                y = y_start + j
                
                # Desenha círculo do neurônio
                circle = mpatches.Circle((x, y), neuron_radius, color=color,
                                   ec='black', linewidth=1.5, zorder=3)
                ax.add_patch(circle)
        
        # Labels das camadas
        layer_names = ['Input\n({})'.format(all_layers[0])]
        for idx, size in enumerate(layer_sizes):
            layer_names.append('Hidden {}\n({})'.format(idx + 1, size))
        layer_names.append('Output\n({})'.format(all_layers[-1]))
        
        for i, name in enumerate(layer_names):
            x = i * layer_spacing
            ax.text(x, -0.8, name, ha='center', va='top',
                   fontsize=10, fontweight='bold')
        
        # Configurações do plot
        ax.set_xlim(-0.5, (n_layers - 1) * layer_spacing + 0.5)
        ax.set_ylim(-1.5, max_neurons + 0.5)
        ax.axis('off')
        ax.set_aspect('equal')
        
        # Título com informações
        activation = model_info['best_params'].get('activation', 'relu')
        solver = model_info['best_params'].get('solver', 'adam')
        title = f"Camadas: {len(layer_sizes)} | Ativação: {activation} | Otimizador: {solver}"
        ax.text(0.5, 0.98, title, transform=ax.transAxes,
               ha='center', va='top', fontsize=11, fontweight='bold')
        
        fig.tight_layout()
        
        canvas_frame = ctk.CTkFrame(chart_frame, fg_color="#ffffff", corner_radius=5)
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        canvas = FigureCanvasTkAgg(fig, canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
        export_btn = add_chart_export_button(chart_frame, fig, "neural_network_architecture")
        export_btn.pack(pady=(0, 10))
    
    def _create_importance_chart(self, parent):
        """Cria gráfico de importância"""
        chart_frame = ctk.CTkFrame(parent)
        chart_frame.pack(fill="both", expand=True, pady=10)
        
        ctk.CTkLabel(
            chart_frame,
            text="🎯 Importância das Variáveis",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(pady=(10, 5), padx=10, anchor="w")
        
        fig = Figure(figsize=(8, 5), dpi=100, facecolor='white')
        ax = fig.add_subplot(111)
        
        feature_importance = self.results['feature_importance']
        variables = list(feature_importance.keys())[:10]  # Top 10
        importance_values = [feature_importance[v] for v in variables]
        
        colors = ['#1f77b4' if imp > 0 else '#ff7f0e' for imp in importance_values]
        
        ax.barh(variables, importance_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Importância', fontsize=11, fontweight='bold')
        ax.set_title('Top 10 Variáveis Mais Importantes', fontsize=13, fontweight='bold', pad=10)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        fig.tight_layout()
        
        canvas_frame = ctk.CTkFrame(chart_frame, fg_color="#ffffff", corner_radius=5)
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        canvas = FigureCanvasTkAgg(fig, canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
        export_btn = add_chart_export_button(chart_frame, fig, "neural_network_importance")
        export_btn.pack(pady=(0, 10))
    
    def _create_confusion_matrix_chart(self, parent):
        """Cria gráfico de matriz de confusão"""
        chart_frame = ctk.CTkFrame(parent)
        chart_frame.pack(fill="both", expand=True, pady=10)
        
        ctk.CTkLabel(
            chart_frame,
            text="📊 Matriz de Confusão",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(pady=(10, 5), padx=10, anchor="w")
        
        method = self.method_var.get()
        
        # Modelo carregado ou K-Fold
        if self.results.get('loaded_model', False) or method == "kfold":
            cm = self.results['metrics']['confusion_matrix']
        else:  # Holdout
            cm = self.results['metrics_test']['confusion_matrix']
        
        np = get_numpy()
        cm_array = np.array(cm)
        
        fig = Figure(figsize=(6, 5), dpi=100, facecolor='white')
        ax = fig.add_subplot(111)
        
        im = ax.imshow(cm_array, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        n_classes = cm_array.shape[0]
        ax.set(xticks=np.arange(n_classes),
               yticks=np.arange(n_classes),
               xlabel='Predito',
               ylabel='Real',
               title='Matriz de Confusão')
        
        # Adiciona valores nas células
        thresh = cm_array.max() / 2.
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, format(cm_array[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm_array[i, j] > thresh else "black",
                       fontsize=12, fontweight='bold')
        
        fig.tight_layout()
        
        canvas_frame = ctk.CTkFrame(chart_frame, fg_color="#ffffff", corner_radius=5)
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        canvas = FigureCanvasTkAgg(fig, canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
        export_btn = add_chart_export_button(chart_frame, fig, "neural_network_confusion_matrix")
        export_btn.pack(pady=(0, 10))
