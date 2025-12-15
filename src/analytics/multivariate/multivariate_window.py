"""
Multivariate Analysis Window
Interface para Análise Multivariada com Matriz de Correlação
"""

import customtkinter as ctk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns

from src.utils.lazy_imports import get_pandas, get_numpy
from src.analytics.multivariate.multivariate_utils import (
    perform_multivariate_analysis,
    validate_multivariate_data
)


class MultivariateWindow(ctk.CTkToplevel):
    """Janela para Análise Multivariada"""
    
    def __init__(self, parent, data):
        super().__init__(parent)
        self.title("Análise Multivariada - Matriz de Correlação")
        self.geometry("1400x800")
        self.minsize(1200, 700)
        
        # Maximizar janela
        try:
            self.state("zoomed")
        except Exception:
            pass
        
        # Configurar como modal
        self.transient(parent)
        self.grab_set()
        
        self.data = data
        self.correlation_matrix = None
        self.column_names = []
        self.normalized_df = None
        self.current_fig = None
        self.canvas_widget = None
        self.column_checkboxes = {}  # Armazena checkboxes
        self.column_vars = {}  # Armazena variáveis dos checkboxes
        
        self._build_ui()
        self._create_column_selection()
    
    def _build_ui(self):
        """Constrói a interface"""
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=16, pady=16)
        
        # ===== PAINEL ESQUERDO =====
        left_panel = ctk.CTkFrame(main_frame, width=450)
        left_panel.pack(side="left", fill="both", expand=False, padx=(0, 8))
        left_panel.pack_propagate(False)
        
        # Título
        ctk.CTkLabel(
            left_panel,
            text="Análise Multivariada",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(12, 8))
        
        # Seleção de colunas
        ctk.CTkLabel(
            left_panel,
            text="Selecione as Variáveis (X):",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", padx=12, pady=(8, 4))
        
        # Frame scrollável para checkboxes
        self.columns_frame = ctk.CTkScrollableFrame(left_panel, height=200)
        self.columns_frame.pack(fill="x", padx=12, pady=(0, 8))
        
        # Método de correlação
        ctk.CTkLabel(
            left_panel,
            text="Método de Correlação:",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", padx=12, pady=(8, 4))
        
        self.method_var = ctk.StringVar(value="pearson")
        method_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
        method_frame.pack(fill="x", padx=12, pady=(0, 8))
        
        ctk.CTkRadioButton(
            method_frame,
            text="Pearson",
            variable=self.method_var,
            value="pearson"
        ).pack(side="left", padx=4)
        
        ctk.CTkRadioButton(
            method_frame,
            text="Spearman",
            variable=self.method_var,
            value="spearman"
        ).pack(side="left", padx=4)
        
        ctk.CTkRadioButton(
            method_frame,
            text="Kendall",
            variable=self.method_var,
            value="kendall"
        ).pack(side="left", padx=4)
        
        # Botão calcular
        ctk.CTkButton(
            left_panel,
            text="🔍 Calcular Correlação",
            command=self._calculate_correlation,
            height=36,
            fg_color="#27AE60",
            hover_color="#1E8449"
        ).pack(fill="x", padx=12, pady=(0, 12))
        
        # Tabela de correlação
        ctk.CTkLabel(
            left_panel,
            text="Matriz de Correlação:",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", padx=12, pady=(8, 4))
        
        # Frame para tabela
        table_frame = ctk.CTkFrame(left_panel)
        table_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        
        # Scrollbars
        scroll_y = ttk.Scrollbar(table_frame, orient="vertical")
        scroll_x = ttk.Scrollbar(table_frame, orient="horizontal")
        
        # Treeview para matriz de correlação
        self.correlation_tree = ttk.Treeview(
            table_frame,
            yscrollcommand=scroll_y.set,
            xscrollcommand=scroll_x.set,
            selectmode='none'
        )
        
        scroll_y.config(command=self.correlation_tree.yview)
        scroll_x.config(command=self.correlation_tree.xview)
        
        scroll_y.pack(side="right", fill="y")
        scroll_x.pack(side="bottom", fill="x")
        self.correlation_tree.pack(fill="both", expand=True)
        
        # ===== PAINEL DIREITO =====
        right_panel = ctk.CTkFrame(main_frame)
        right_panel.pack(side="right", fill="both", expand=True)
        
        # Título
        ctk.CTkLabel(
            right_panel,
            text="Visualizações",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=12, pady=(8, 8))
        
        # Botões de visualização
        btn_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
        btn_frame.pack(fill="x", padx=12, pady=(0, 8))
        
        ctk.CTkButton(
            btn_frame,
            text="📊 Heatmap de Correlação",
            command=self._show_heatmap,
            height=32,
            fg_color="#2E86DE",
            hover_color="#1B5AA3"
        ).pack(side="left", padx=2)
        
        ctk.CTkButton(
            btn_frame,
            text="📈 Scatter Plot Matrix",
            command=self._show_scatter_matrix,
            height=32,
            fg_color="#27AE60",
            hover_color="#1E8449"
        ).pack(side="left", padx=2)
        
        # Container para visualizações
        self.viz_container = ctk.CTkFrame(right_panel, fg_color="white")
        self.viz_container.pack(fill="both", expand=True, padx=12, pady=(0, 12))
    
    def _create_column_selection(self):
        """Cria checkboxes para seleção de colunas"""
        # Seleciona apenas colunas numéricas
        numeric_data = self.data.select_dtypes(include=['number'])
        
        if len(numeric_data.columns) < 2:
            messagebox.showerror("Erro", "São necessárias pelo menos 2 colunas numéricas")
            self.destroy()
            return
        
        # Cria checkbox para cada coluna
        for col in numeric_data.columns:
            var = ctk.BooleanVar(value=True)  # Todas selecionadas por padrão
            self.column_vars[col] = var
            
            checkbox = ctk.CTkCheckBox(
                self.columns_frame,
                text=col,
                variable=var,
                font=ctk.CTkFont(size=11)
            )
            checkbox.pack(anchor="w", pady=2)
            self.column_checkboxes[col] = checkbox
    
    def _calculate_correlation(self):
        """Calcula a correlação com as colunas selecionadas"""
        # Obtém colunas selecionadas
        selected_columns = [col for col, var in self.column_vars.items() if var.get()]
        
        if len(selected_columns) < 2:
            messagebox.showwarning("Aviso", "Selecione pelo menos 2 variáveis")
            return
        
        # Filtra dados
        selected_data = self.data[selected_columns]
        
        # Valida
        is_valid, error_msg = validate_multivariate_data(selected_data)
        if not is_valid:
            messagebox.showerror("Erro de Validação", error_msg)
            return
        
        try:
            pd = get_pandas()
            
            # Calcula matriz de correlação com o método selecionado
            method = self.method_var.get()
            corr_matrix = selected_data.corr(method=method)
            
            self.correlation_matrix = corr_matrix.values
            self.column_names = list(selected_data.columns)
            self.normalized_df = selected_data
            
            # Preenche tabela
            self._fill_correlation_table()
            
            # Mostra heatmap
            self._show_heatmap()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao calcular correlação: {str(e)}")
    
    def _fill_correlation_table(self):
        """Preenche a tabela com a matriz de correlação"""
        # Limpa a tabela antes de preencher
        for item in self.correlation_tree.get_children():
            self.correlation_tree.delete(item)
        
        # Configura colunas
        columns = ["Variável"] + self.column_names
        self.correlation_tree["columns"] = columns
        self.correlation_tree["show"] = "headings"
        
        # Cabeçalhos
        for col in columns:
            self.correlation_tree.heading(col, text=col)
            self.correlation_tree.column(col, width=100, anchor="center")
        
        # Preenche dados
        for i, row_name in enumerate(self.column_names):
            row_data = [row_name]
            for j, val in enumerate(self.correlation_matrix[i]):
                # Formata com cor
                row_data.append(f"{val:.3f}")
            
            # Insere linha
            self.correlation_tree.insert("", "end", values=row_data, tags=(f"row{i}",))
            
            # Aplica cores baseado em valores
            for j, val in enumerate(self.correlation_matrix[i]):
                if val <= -0.75:
                    self.correlation_tree.tag_configure(f"row{i}", foreground="red")
                elif val >= 0.75 and i != j:
                    self.correlation_tree.tag_configure(f"row{i}", foreground="blue")
    
    def _show_heatmap(self):
        """Mostra heatmap da matriz de correlação"""
        # Remove visualização anterior
        if self.canvas_widget:
            self.canvas_widget.get_tk_widget().destroy()
        
        # Cria figura
        fig = Figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        # Heatmap com seaborn
        sns.heatmap(
            self.correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            xticklabels=self.column_names,
            yticklabels=self.column_names,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title("Matriz de Correlação", fontsize=14, fontweight='bold', pad=20)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        fig.tight_layout()
        
        # Cria canvas
        self.canvas_widget = FigureCanvasTkAgg(fig, master=self.viz_container)
        self.canvas_widget.draw()
        self.canvas_widget.get_tk_widget().pack(fill="both", expand=True)
        
        self.current_fig = fig
    
    def _show_scatter_matrix(self):
        """Mostra scatter plot matrix"""
        # Remove visualização anterior
        if self.canvas_widget:
            self.canvas_widget.get_tk_widget().destroy()
        
        pd = get_pandas()
        
        # Recria DataFrame para scatter
        df_plot = pd.DataFrame(
            {col: self.data[col].values for col in self.column_names}
        )
        
        # Calcula número de variáveis
        n_vars = len(self.column_names)
        
        # Cria figura
        fig, axes = plt.subplots(n_vars, n_vars, figsize=(12, 10))
        fig.suptitle("Scatter Plot Matrix", fontsize=14, fontweight='bold')
        
        # Preenche matriz de scatter plots
        for i in range(n_vars):
            for j in range(n_vars):
                ax = axes[i, j] if n_vars > 1 else axes
                
                if i == j:
                    # Diagonal: histograma
                    ax.hist(df_plot[self.column_names[i]], bins=20, 
                           color='skyblue', edgecolor='black', alpha=0.7)
                    ax.set_ylabel('')
                else:
                    # Fora da diagonal: scatter plot
                    ax.scatter(
                        df_plot[self.column_names[j]], 
                        df_plot[self.column_names[i]],
                        alpha=0.5,
                        s=20,
                        color='black'
                    )
                    
                    # Linha de tendência
                    try:
                        np = get_numpy()
                        z = np.polyfit(df_plot[self.column_names[j]], 
                                      df_plot[self.column_names[i]], 1)
                        p = np.poly1d(z)
                        ax.plot(df_plot[self.column_names[j]], 
                               p(df_plot[self.column_names[j]]),
                               "r-", linewidth=1, alpha=0.8)
                    except:
                        pass
                
                # Labels apenas nas bordas
                if i == n_vars - 1:
                    ax.set_xlabel(self.column_names[j], fontsize=9)
                else:
                    ax.set_xlabel('')
                
                if j == 0:
                    ax.set_ylabel(self.column_names[i], fontsize=9)
                else:
                    ax.set_ylabel('')
                
                # Remove ticks para limpeza visual
                ax.tick_params(labelsize=7)
        
        fig.tight_layout()
        
        # Cria canvas
        self.canvas_widget = FigureCanvasTkAgg(fig, master=self.viz_container)
        self.canvas_widget.draw()
        self.canvas_widget.get_tk_widget().pack(fill="both", expand=True)
        
        self.current_fig = fig
