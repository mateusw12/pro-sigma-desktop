"""
Interface para Modelos de Árvore
Decision Tree, Random Forest e Gradient Boosting
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog
import threading
import pickle
import json
from pathlib import Path
from datetime import datetime

# Lazy imports
def get_pandas():
    import pandas as pd
    return pd

def get_numpy():
    import numpy as np
    return np


class TreeModelsWindow(ctk.CTkToplevel):
    def __init__(self, parent, data):
        super().__init__(parent)
        
        self.title("🌳 Modelos de Árvore - ProSigma")
        self.geometry("1400x900")
        
        # Previne que o fechamento desta janela feche o app inteiro
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        
        self.data = data.copy() if data is not None else None
        self.results = None
        
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
            text="🌳 Modelos de Árvore",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=(0, 5))
        
        # Descrição
        desc = ctk.CTkLabel(
            self.main_container,
            text="Decision Tree, Random Forest e Gradient Boosting",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        desc.pack(pady=(0, 10))
        
        # Frame de configuração
        config_frame = ctk.CTkFrame(self.main_container)
        config_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(
            config_frame,
            text="⚙️ Configuração do Modelo",
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
        
        # Tipo de Modelo
        ctk.CTkLabel(
            options_grid,
            text="Tipo de Modelo:",
            font=ctk.CTkFont(size=11, weight="bold")
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.model_type_var = tk.StringVar(value="🌳 Decision Tree (Árvore de Decisão)")
        model_combo = ctk.CTkComboBox(
            options_grid,
            variable=self.model_type_var,
            values=[
                "🌳 Decision Tree (Árvore de Decisão)",
                "🌲 Random Forest (Floresta Aleatória)",
                "⚡ Gradient Boosting (Boosting de Gradiente)"
            ],
            width=320,
            state="readonly",
            command=self._on_model_change
        )
        model_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Test Size
        ctk.CTkLabel(
            options_grid,
            text="Tamanho do Teste (%):",
            font=ctk.CTkFont(size=11, weight="bold")
        ).grid(row=0, column=2, padx=(20, 5), pady=5, sticky="w")
        
        self.test_size_var = tk.DoubleVar(value=30.0)
        test_size_entry = ctk.CTkEntry(
            options_grid,
            textvariable=self.test_size_var,
            width=150
        )
        test_size_entry.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        
        # === Parâmetros específicos do Decision Tree ===
        self.dt_params_frame = ctk.CTkFrame(options_grid, fg_color="transparent")
        self.dt_params_frame.grid(row=1, column=0, columnspan=4, sticky="ew", pady=5)
        
        ctk.CTkLabel(
            self.dt_params_frame,
            text="Profundidade Máxima:",
            font=ctk.CTkFont(size=11, weight="bold")
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.max_depth_var = tk.StringVar(value="None")
        max_depth_entry = ctk.CTkEntry(
            self.dt_params_frame,
            textvariable=self.max_depth_var,
            width=150
        )
        max_depth_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # === Parâmetros específicos do Random Forest ===
        self.rf_params_frame = ctk.CTkFrame(options_grid, fg_color="transparent")
        self.rf_params_frame.grid(row=1, column=0, columnspan=4, sticky="ew", pady=5)
        self.rf_params_frame.grid_remove()  # Esconde inicialmente
        
        ctk.CTkLabel(
            self.rf_params_frame,
            text="Número de Árvores:",
            font=ctk.CTkFont(size=11, weight="bold")
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.n_estimators_var = tk.IntVar(value=100)
        n_estimators_entry = ctk.CTkEntry(
            self.rf_params_frame,
            textvariable=self.n_estimators_var,
            width=150
        )
        n_estimators_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # === Parâmetros específicos do Gradient Boosting ===
        self.gb_params_frame = ctk.CTkFrame(options_grid, fg_color="transparent")
        self.gb_params_frame.grid(row=1, column=0, columnspan=4, sticky="ew", pady=5)
        self.gb_params_frame.grid_remove()  # Esconde inicialmente
        
        ctk.CTkLabel(
            self.gb_params_frame,
            text="Taxa de Aprendizado:",
            font=ctk.CTkFont(size=11, weight="bold")
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.learning_rate_var = tk.DoubleVar(value=0.1)
        learning_rate_entry = ctk.CTkEntry(
            self.gb_params_frame,
            textvariable=self.learning_rate_var,
            width=150
        )
        learning_rate_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        ctk.CTkLabel(
            self.gb_params_frame,
            text="Número de Árvores:",
            font=ctk.CTkFont(size=11, weight="bold")
        ).grid(row=0, column=2, padx=(20, 5), pady=5, sticky="w")
        
        self.gb_n_estimators_var = tk.IntVar(value=100)
        gb_n_estimators_entry = ctk.CTkEntry(
            self.gb_params_frame,
            textvariable=self.gb_n_estimators_var,
            width=150
        )
        gb_n_estimators_entry.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        
        # Container para botões
        buttons_container = ctk.CTkFrame(config_frame, fg_color="transparent")
        buttons_container.pack(pady=15)
        
        # Botão Treinar
        train_btn = ctk.CTkButton(
            buttons_container,
            text="🚀 Treinar Modelo",
            command=self._train_model,
            font=ctk.CTkFont(size=13, weight="bold"),
            height=35,
            width=180
        )
        train_btn.pack(side="left", padx=5)
        
        # Botão Salvar Modelo
        save_btn = ctk.CTkButton(
            buttons_container,
            text="💾 Salvar Modelo",
            command=self._save_model,
            font=ctk.CTkFont(size=13, weight="bold"),
            height=35,
            width=180,
            fg_color="green",
            hover_color="darkgreen"
        )
        save_btn.pack(side="left", padx=5)
        
        # Botão Carregar Modelo
        load_btn = ctk.CTkButton(
            buttons_container,
            text="📂 Carregar Modelo",
            command=self._load_model,
            font=ctk.CTkFont(size=13, weight="bold"),
            height=35,
            width=180,
            fg_color="orange",
            hover_color="darkorange"
        )
        load_btn.pack(side="left", padx=5)
        
        # Loading frame
        self.loading_frame = ctk.CTkFrame(self.main_container)
        
        # Results frame
        self.results_frame = ctk.CTkFrame(self.main_container)
    
    def _on_model_change(self, value):
        """Atualiza parâmetros visíveis quando modelo muda"""
        self.dt_params_frame.grid_remove()
        self.rf_params_frame.grid_remove()
        self.gb_params_frame.grid_remove()
        
        if "Decision Tree" in value:
            self.dt_params_frame.grid()
        elif "Random Forest" in value:
            self.rf_params_frame.grid()
        elif "Gradient Boosting" in value:
            self.gb_params_frame.grid()
    
    def _populate_columns(self):
        """Popula checkboxes para seleção de colunas"""
        if self.data is None:
            return
        
        pd = get_pandas()
        
        # Converte datetime
        for col in self.data.columns:
            if pd.api.types.is_datetime64_any_dtype(self.data[col]):
                self.data[col] = self.data[col].astype(str)
        
        all_cols = self.data.columns.tolist()
        
        if len(all_cols) < 2:
            messagebox.showerror(
                "Erro",
                "É necessário ter pelo menos 2 colunas (1 X + 1 Y) para análise."
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
        
        # Radio buttons para Y
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
    
    def _train_model(self):
        """Treina o modelo selecionado"""
        # Obtém colunas selecionadas
        x_columns = [col for col, var in self.x_column_vars.items() if var.get()]
        y_column = self.y_column_var.get()
        
        if len(x_columns) < 1:
            messagebox.showwarning("Aviso", "Selecione pelo menos 1 coluna X")
            return
        
        if y_column in x_columns:
            messagebox.showwarning("Aviso", "A variável Y não pode estar em X")
            return
        
        # Obtém colunas categóricas
        categorical_cols = [col for col, var in self.categorical_column_vars.items() if var.get() and col in x_columns]
        
        try:
            # Mostra loading
            self._show_loading(True)
            
            model_type_label = self.model_type_var.get()
            # Extrai o tipo real
            if "Decision Tree" in model_type_label:
                model_type = "decision_tree"
            elif "Random Forest" in model_type_label:
                model_type = "random_forest"
            else:
                model_type = "gradient_boosting"
            
            test_size = self.test_size_var.get() / 100.0
            
            def train_thread():
                try:
                    X = self.data[x_columns].copy()
                    y = self.data[y_column].copy()
                    
                    if model_type == "decision_tree":
                        from src.analytics.tree_models.tree_models_utils import train_decision_tree
                        
                        max_depth_str = self.max_depth_var.get()
                        max_depth = None if max_depth_str == "None" else int(max_depth_str)
                        
                        self.results = train_decision_tree(
                            X, y, categorical_cols, test_size, max_depth=max_depth
                        )
                    
                    elif model_type == "random_forest":
                        from src.analytics.tree_models.tree_models_utils import train_random_forest
                        
                        n_estimators = self.n_estimators_var.get()
                        
                        self.results = train_random_forest(
                            X, y, categorical_cols, test_size, n_estimators=n_estimators
                        )
                    
                    elif model_type == "gradient_boosting":
                        from src.analytics.tree_models.tree_models_utils import train_gradient_boosting
                        
                        learning_rate = self.learning_rate_var.get()
                        n_estimators = self.gb_n_estimators_var.get()
                        
                        self.results = train_gradient_boosting(
                            X, y, categorical_cols, test_size, 
                            n_estimators=n_estimators, learning_rate=learning_rate
                        )
                    
                    self.after(0, self._display_results)
                    
                except Exception as e:
                    self.after(0, lambda: messagebox.showerror("Erro", f"Erro ao treinar modelo: {str(e)}"))
                    self.after(0, lambda: self._show_loading(False))
            
            thread = threading.Thread(target=train_thread, daemon=True)
            thread.start()
            
        except Exception as e:
            self._show_loading(False)
            messagebox.showerror("Erro", f"Erro ao treinar modelo: {str(e)}")
    
    def _show_loading(self, show: bool):
        """Mostra/esconde loading"""
        if show:
            self.results_frame.pack_forget()
            self.loading_frame.pack(fill="both", expand=True, pady=10)
            
            for widget in self.loading_frame.winfo_children():
                widget.destroy()
            
            ctk.CTkLabel(
                self.loading_frame,
                text="⏳ Treinando modelo...",
                font=ctk.CTkFont(size=16, weight="bold")
            ).pack(pady=20)
            
            progress = ctk.CTkProgressBar(self.loading_frame, mode="indeterminate")
            progress.pack(pady=10, padx=50, fill="x")
            progress.start()
        else:
            self.loading_frame.pack_forget()
            self.results_frame.pack(fill="both", expand=True, pady=(0, 10))
    
    def _display_results(self):
        """Exibe resultados do treinamento"""
        self._show_loading(False)
        
        # Limpa resultados anteriores
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        if not self.results:
            return
        
        # Título
        ctk.CTkLabel(
            self.results_frame,
            text="📊 Resultados do Treinamento",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(15, 10))
        
        # Container para tabelas lado a lado
        tables_container = ctk.CTkFrame(self.results_frame, fg_color="transparent")
        tables_container.pack(fill="x", padx=20, pady=10)
        
        # Informações do Modelo (esquerda)
        model_info_frame = ctk.CTkFrame(tables_container)
        model_info_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        self._display_model_info(model_info_frame)
        
        # Métricas (direita)
        metrics_frame = ctk.CTkFrame(tables_container)
        metrics_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        self._display_metrics(metrics_frame)
        
        # Gráficos
        charts_container = ctk.CTkFrame(self.results_frame, fg_color="transparent")
        charts_container.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Feature Importance (esquerda)
        importance_frame = ctk.CTkFrame(charts_container)
        importance_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        self._display_feature_importance_chart(importance_frame)
        
        # Gráfico específico (direita)
        specific_chart_frame = ctk.CTkFrame(charts_container)
        specific_chart_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        if self.results['is_classification']:
            self._display_confusion_matrix(specific_chart_frame)
        else:
            self._display_predictions_chart(specific_chart_frame)
        
        # Visualização da Árvore (apenas Decision Tree)
        model_type = self.results['model_info'].get('model_type', '')
        if 'Decision Tree' in model_type:
            tree_frame = ctk.CTkFrame(self.results_frame)
            tree_frame.pack(fill="both", expand=True, padx=20, pady=10)
            self._display_tree_visualization(tree_frame)
    
    def _display_model_info(self, parent):
        """Exibe informações do modelo em formato de tabela compacta"""
        ctk.CTkLabel(
            parent,
            text="ℹ️ Informações do Modelo",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 10), padx=10, anchor="w")
        
        info = self.results['model_info']
        
        # Prepara dados para tabela
        headers = ['Parâmetro', 'Valor']
        data_rows = []
        
        # Ordem de exibição
        display_order = [
            ('model_type', 'Tipo do Modelo'),
            ('n_estimators', 'Número de Árvores'),
            ('max_depth', 'Profundidade Máxima'),
            ('learning_rate', 'Taxa de Aprendizado'),
            ('subsample', 'Subsample'),
            ('n_leaves', 'Número de Folhas'),
            ('n_features', 'Número de Features'),
            ('n_samples_train', 'Amostras de Treino'),
            ('n_samples_test', 'Amostras de Teste')
        ]
        
        for key, label in display_order:
            if key in info:
                data_rows.append([label, str(info[key])])
        
        # Adiciona outros parâmetros não listados
        for key, value in info.items():
            if key not in [item[0] for item in display_order]:
                label = key.replace('_', ' ').title()
                data_rows.append([label, str(value)])
        
        self._create_compact_table(parent, headers, data_rows)
    
    def _display_metrics(self, parent):
        """Exibe métricas de desempenho em formato de tabela compacta"""
        ctk.CTkLabel(
            parent,
            text="📈 Métricas de Desempenho (Teste)",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 10), padx=10, anchor="w")
        
        test_metrics = self.results['test_metrics']
        
        # Prepara dados para tabela
        if self.results['is_classification']:
            headers = ['Métrica', 'Valor']
            data_rows = [
                ['Acurácia', f"{test_metrics.get('accuracy', 0):.4f}"],
                ['Precisão', f"{test_metrics.get('precision', 0):.4f}"],
                ['Recall', f"{test_metrics.get('recall', 0):.4f}"],
                ['F1-Score', f"{test_metrics.get('f1_score', 0):.4f}"]
            ]
            if 'roc_auc' in test_metrics:
                data_rows.append(['ROC AUC', f"{test_metrics['roc_auc']:.4f}"])
        else:
            headers = ['Métrica', 'Valor']
            data_rows = [
                ['R² Score', f"{test_metrics.get('r2_score', 0):.4f}"],
                ['RMSE', f"{test_metrics.get('rmse', 0):.6f}"],
                ['MAE', f"{test_metrics.get('mae', 0):.6f}"],
                ['MSE', f"{test_metrics.get('mse', 0):.6f}"]
            ]
        
        self._create_compact_table(parent, headers, data_rows)
    
    def _create_compact_table(self, parent, headers, data_rows):
        """Cria tabela compacta com estilo padronizado"""
        # Container para a tabela
        table_container = ctk.CTkFrame(parent)
        table_container.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Header
        header_frame = ctk.CTkFrame(table_container, fg_color="#1f538d", height=30)
        header_frame.pack(fill="x", pady=(0, 2))
        header_frame.pack_propagate(False)
        
        for header in headers:
            header_label = ctk.CTkLabel(
                header_frame,
                text=header,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color="white"
            )
            header_label.pack(side="left", expand=True, fill="both", padx=5)
        
        # Rows
        for row in data_rows:
            row_frame = ctk.CTkFrame(table_container, fg_color="#2b2b2b", height=25)
            row_frame.pack(fill="x", pady=1)
            row_frame.pack_propagate(False)
            
            for i, cell in enumerate(row):
                cell_label = ctk.CTkLabel(
                    row_frame,
                    text=str(cell),
                    font=ctk.CTkFont(size=10),
                    text_color="white",
                    anchor="w" if i == 0 else "e"
                )
                cell_label.pack(side="left", expand=True, fill="both", padx=5)
    
    def _display_feature_importance(self, parent):
        """Exibe importância das features (ANTIGO - mantido para compatibilidade)"""
        self._display_feature_importance_chart(parent)
    
    def _display_feature_importance_chart(self, parent):
        """Exibe gráfico de barras da importância das features"""
        ctk.CTkLabel(
            parent,
            text="🎯 Importância das Features",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            import matplotlib
            import numpy as np
            matplotlib.use('TkAgg')
            
            importance = self.results['feature_importance']
            
            # Pega top 10 features
            top_features = dict(list(importance.items())[:min(10, len(importance))])
            
            # Cria figura
            fig, ax = plt.subplots(figsize=(6, 4), facecolor='#2b2b2b')
            ax.set_facecolor('#2b2b2b')
            
            features = list(top_features.keys())
            values = list(top_features.values())
            
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
            bars = ax.barh(features, values, color=colors)
            
            ax.set_xlabel('Importância', color='white', fontsize=10)
            ax.set_title('Top 10 Features Mais Importantes', color='white', fontsize=11, weight='bold')
            ax.tick_params(colors='white', labelsize=8)
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Adiciona valores nas barras
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f'{width:.4f}',
                       ha='left', va='center', color='white', fontsize=7, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='#1f538d', alpha=0.7))
            
            plt.tight_layout()
            
            # Adiciona ao frame
            canvas = FigureCanvasTkAgg(fig, parent)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
            
        except Exception as e:
            # Fallback para tabela se matplotlib falhar
            ctk.CTkLabel(
                parent,
                text=f"Erro ao gerar gráfico: {str(e)}",
                text_color="orange"
            ).pack(pady=10)
    
    def _display_confusion_matrix(self, parent):
        """Exibe matriz de confusão para classificação"""
        ctk.CTkLabel(
            parent,
            text="📊 Matriz de Confusão",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            import matplotlib
            import numpy as np
            matplotlib.use('TkAgg')
            
            cm = self.results['test_metrics'].get('confusion_matrix', [])
            
            if not cm:
                ctk.CTkLabel(
                    parent,
                    text="Matriz de confusão não disponível",
                    text_color="gray"
                ).pack(pady=20)
                return
            
            # Cria figura
            fig, ax = plt.subplots(figsize=(5, 4), facecolor='#2b2b2b')
            
            im = ax.imshow(cm, cmap='Blues', aspect='auto')
            
            # Labels
            n_classes = len(cm)
            ax.set_xticks(range(n_classes))
            ax.set_yticks(range(n_classes))
            ax.set_xticklabels(range(n_classes), color='white')
            ax.set_yticklabels(range(n_classes), color='white')
            
            ax.set_xlabel('Predito', color='white', fontsize=10)
            ax.set_ylabel('Real', color='white', fontsize=10)
            ax.set_title('Matriz de Confusão (Teste)', color='white', fontsize=11, weight='bold')
            
            # Adiciona valores nas células
            for i in range(n_classes):
                for j in range(n_classes):
                    text = ax.text(j, i, str(cm[i][j]),
                                 ha="center", va="center",
                                 color="white" if cm[i][j] < np.max(cm) / 2 else "black",
                                 fontsize=10, weight='bold')
            
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            
            # Adiciona ao frame
            canvas = FigureCanvasTkAgg(fig, parent)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
            
        except Exception as e:
            ctk.CTkLabel(
                parent,
                text=f"Erro ao gerar matriz: {str(e)}",
                text_color="orange"
            ).pack(pady=10)
    
    def _display_predictions_chart(self, parent):
        """Exibe gráfico de predições vs real para regressão"""
        ctk.CTkLabel(
            parent,
            text="📈 Predições vs Valores Reais",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            import matplotlib
            matplotlib.use('TkAgg')
            
            y_test = self.results.get('y_test', [])
            y_pred = self.results.get('y_pred_test', [])
            
            if not y_test or not y_pred:
                ctk.CTkLabel(
                    parent,
                    text="Dados de predição não disponíveis",
                    text_color="gray"
                ).pack(pady=20)
                return
            
            # Cria figura
            fig, ax = plt.subplots(figsize=(5, 4), facecolor='#2b2b2b')
            ax.set_facecolor('#2b2b2b')
            
            # Scatter plot
            ax.scatter(y_test, y_pred, alpha=0.6, c='#4CAF50', edgecolors='white', linewidth=0.5, s=50)
            
            # Linha diagonal (predição perfeita)
            min_val = min(min(y_test), min(y_pred))
            max_val = max(max(y_test), max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predição Perfeita')
            
            ax.set_xlabel('Valores Reais', color='white', fontsize=10)
            ax.set_ylabel('Predições', color='white', fontsize=10)
            ax.set_title('Predições vs Valores Reais (Teste)', color='white', fontsize=11, weight='bold')
            ax.tick_params(colors='white', labelsize=8)
            ax.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white')
            ax.grid(True, alpha=0.3, color='gray')
            
            for spine in ax.spines.values():
                spine.set_color('white')
            
            plt.tight_layout()
            
            # Adiciona ao frame
            canvas = FigureCanvasTkAgg(fig, parent)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
            
        except Exception as e:
            ctk.CTkLabel(
                parent,
                text=f"Erro ao gerar gráfico: {str(e)}",
                text_color="orange"
            ).pack(pady=10)
    
    def _display_tree_visualization(self, parent):
        """Exibe visualização da estrutura da árvore"""
        ctk.CTkLabel(
            parent,
            text="🌳 Visualização da Árvore de Decisão",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from sklearn.tree import plot_tree
            import matplotlib
            matplotlib.use('TkAgg')
            
            model = self.results['model']
            feature_names = self.results['feature_names']
            
            # Cria figura grande
            fig, ax = plt.subplots(figsize=(14, 8), facecolor='#2b2b2b')
            ax.set_facecolor('#2b2b2b')
            
            # Plota árvore
            plot_tree(
                model,
                feature_names=feature_names,
                filled=True,
                rounded=True,
                fontsize=8,
                max_depth=4,  # Limita profundidade para visualização
                ax=ax
            )
            
            ax.set_title('Estrutura da Árvore (max_depth=4 para visualização)', 
                        color='white', fontsize=12, weight='bold', pad=20)
            
            plt.tight_layout()
            
            # Adiciona ao frame com scrollbar
            canvas_frame = ctk.CTkScrollableFrame(parent, height=500)
            canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            canvas = FigureCanvasTkAgg(fig, canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            ctk.CTkLabel(
                parent,
                text=f"Erro ao gerar visualização da árvore: {str(e)}",
                text_color="orange"
            ).pack(pady=10)
    
    def _save_model(self):
        """Salva o modelo treinado"""
        if not self.results:
            messagebox.showwarning("Aviso", "Nenhum modelo treinado para salvar")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Salvar Modelo",
            defaultextension=".pkl",
            filetypes=[
                ("Modelo ProSigma", "*.pkl"),
                ("Todos os arquivos", "*.*")
            ],
            initialfile=f"tree_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        )
        
        if not file_path:
            return
        
        try:
            # Obtém colunas selecionadas
            x_columns = [col for col, var in self.x_column_vars.items() if var.get()]
            y_column = self.y_column_var.get()
            categorical_cols = [col for col, var in self.categorical_column_vars.items() if var.get() and col in x_columns]
            
            # Extrai tipo do modelo
            model_type_label = self.model_type_var.get()
            if "Decision Tree" in model_type_label:
                model_type = "decision_tree"
            elif "Random Forest" in model_type_label:
                model_type = "random_forest"
            else:
                model_type = "gradient_boosting"
            
            # Prepara dados para salvar
            model_data = {
                'model': self.results['model'],
                'encoders': self.results['encoders'],
                'y_encoder': self.results.get('y_encoder'),
                'is_classification': self.results['is_classification'],
                'feature_names': self.results['feature_names'],
                'x_columns': x_columns,
                'y_column': y_column,
                'categorical_cols': categorical_cols,
                'model_info': self.results['model_info'],
                'feature_importance': self.results['feature_importance'],
                'model_type': model_type,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'prosigma_version': '1.0.0'
            }
            
            # Salva modelo em pickle
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Salva metadata em JSON
            json_path = Path(file_path).with_suffix('.json')
            metadata = {
                'model_type': model_data['model_type'],
                'model_info': model_data['model_info'],
                'x_columns': x_columns,
                'y_column': y_column,
                'categorical_cols': categorical_cols,
                'is_classification': model_data['is_classification'],
                'timestamp': model_data['timestamp'],
                'prosigma_version': model_data['prosigma_version']
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            messagebox.showinfo(
                "Sucesso",
                f"Modelo salvo com sucesso!\n\n"
                f"Arquivo: {Path(file_path).name}\n"
                f"Metadata: {json_path.name}"
            )
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao salvar modelo: {str(e)}")
    
    def _load_model(self):
        """Carrega um modelo salvo"""
        file_path = filedialog.askopenfilename(
            title="Carregar Modelo",
            filetypes=[
                ("Modelo ProSigma", "*.pkl"),
                ("Todos os arquivos", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            self._show_loading(True)
            
            # Carrega modelo
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Verifica compatibilidade
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
            
            # Verifica colunas
            pd = get_pandas()
            missing_cols = set(model_data['x_columns']) - set(self.data.columns)
            if missing_cols:
                self._show_loading(False)
                messagebox.showerror(
                    "Erro - Colunas Faltando",
                    f"Os dados atuais não têm as colunas necessárias.\n\n"
                    f"Colunas faltando: {', '.join(missing_cols)}"
                )
                return
            
            # Prepara dados para predição
            X = self.data[model_data['x_columns']].copy()
            
            # Aplica encoding
            if model_data['encoders']:
                for col, encoder in model_data['encoders'].items():
                    if col in X.columns:
                        X[col] = encoder.transform(X[col].astype(str))
            
            # Faz predições
            y_pred = model_data['model'].predict(X)
            
            # Monta results
            self.results = {
                'model': model_data['model'],
                'encoders': model_data['encoders'],
                'y_encoder': model_data.get('y_encoder'),
                'is_classification': model_data['is_classification'],
                'feature_names': model_data['feature_names'],
                'feature_importance': model_data['feature_importance'],
                'model_info': model_data['model_info'],
                'test_metrics': {},
                'y_pred_test': y_pred.tolist(),
                'y_test': []
            }
            
            # Se temos Y nos dados, calcula métricas
            if model_data['y_column'] in self.data.columns:
                from src.analytics.tree_models.tree_models_utils import (
                    calculate_metrics_classification,
                    calculate_metrics_regression
                )
                
                y_true = self.data[model_data['y_column']].copy()
                
                if model_data['y_encoder']:
                    y_true_encoded = model_data['y_encoder'].transform(y_true)
                else:
                    y_true_encoded = y_true.values
                
                if model_data['is_classification']:
                    metrics = calculate_metrics_classification(y_true_encoded, y_pred, model_data['model'], X)
                else:
                    metrics = calculate_metrics_regression(y_true_encoded, y_pred)
                
                self.results['test_metrics'] = metrics
                self.results['y_test'] = y_true_encoded.tolist()
            
            self.after(0, self._display_results)
            
        except Exception as e:
            self._show_loading(False)
            messagebox.showerror("Erro", f"Erro ao carregar modelo: {str(e)}")
