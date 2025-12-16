"""
Multivariate Analysis Window
Interface para Análise Multivariada com Matriz de Correlação MELHORADA
"""

import customtkinter as ctk
from tkinter import messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from scipy.cluster import hierarchy

from src.analytics.multivariate.multivariate_utils import (
    calculate_correlation_with_pvalues,
    calculate_vif,
    calculate_hierarchical_clustering,
    interpret_correlation,
    interpret_vif,
    validate_multivariate_data
)


class MultivariateWindow(ctk.CTkToplevel):
    """Janela para Análise Multivariada com Heatmap e P-Values"""
    
    def __init__(self, parent, data):
        super().__init__(parent)
        self.title("Análise Multivariada - Matriz de Correlação")
        self.geometry("1400x900")
        
        # Maximizar janela
        self.state('zoomed')
        self.lift()
        self.focus_force()
        
        self.data = data
        self.selected_data = None
        self.corr_matrix = None
        self.pvalue_matrix = None
        self.vif_data = None
        self.column_vars = {}
        
        self._setup_ui()
        self._load_columns()
    
    def _setup_ui(self):
        """Configura interface"""
        # Frame principal com scroll
        main_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Título
        title = ctk.CTkLabel(
            main_frame,
            text="Análise Multivariada",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=(0, 10))
        
        # Descrição
        desc = ctk.CTkLabel(
            main_frame,
            text="Matriz de Correlação com P-Values, Heatmap, VIF e Clustering Hierárquico",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        desc.pack(pady=(0, 20))
        
        # Frame de configuração
        config_frame = ctk.CTkFrame(main_frame)
        config_frame.pack(fill="x", pady=(0, 20))
        
        config_title = ctk.CTkLabel(
            config_frame,
            text="Seleção de Variáveis",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        config_title.grid(row=0, column=0, columnspan=2, sticky="w", padx=15, pady=(15, 10))
        
        # Lista de variáveis
        var_label = ctk.CTkLabel(config_frame, text="Variáveis Numéricas:", anchor="w")
        var_label.grid(row=1, column=0, sticky="w", padx=15, pady=5)
        
        self.var_scroll_frame = ctk.CTkScrollableFrame(config_frame, height=150)
        self.var_scroll_frame.grid(row=2, column=0, sticky="nsew", padx=15, pady=5, rowspan=3)
        
        # Método de correlação
        method_label = ctk.CTkLabel(config_frame, text="Método:", anchor="w")
        method_label.grid(row=1, column=1, sticky="w", padx=15, pady=5)
        
        self.method_var = ctk.StringVar(value="pearson")
        method_combo = ctk.CTkComboBox(
            config_frame,
            values=["pearson", "spearman", "kendall"],
            variable=self.method_var,
            width=150
        )
        method_combo.grid(row=2, column=1, sticky="w", padx=15, pady=5)
        
        # Botões de ação
        action_frame = ctk.CTkFrame(main_frame)
        action_frame.pack(fill="x", pady=(0, 20))
        
        analyze_btn = ctk.CTkButton(
            action_frame,
            text="▶ Calcular Correlação",
            command=self._analyze,
            width=200,
            height=40,
            fg_color="#2E86DE",
            hover_color="#1c5fa8"
        )
        analyze_btn.pack(side="left", padx=15, pady=15)
        
        vif_btn = ctk.CTkButton(
            action_frame,
            text="📊 Calcular VIF",
            command=self._calculate_vif_analysis,
            width=200,
            height=40
        )
        vif_btn.pack(side="left", padx=5, pady=15)
        
        clear_btn = ctk.CTkButton(
            action_frame,
            text="🗑 Limpar Resultados",
            command=self._clear_results,
            width=200,
            height=40,
            fg_color="#6c757d",
            hover_color="#5a6268"
        )
        clear_btn.pack(side="left", padx=5, pady=15)
        
        # Frame de resultados
        self.results_frame = ctk.CTkFrame(main_frame)
        self.results_frame.pack(fill="both", expand=True)
        
        results_title = ctk.CTkLabel(
            self.results_frame,
            text="Resultados da Análise",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        results_title.pack(pady=15)
    
    def _load_columns(self):
        """Carrega colunas numéricas"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            messagebox.showerror("Erro", "São necessárias pelo menos 2 colunas numéricas")
            self.destroy()
            return
        
        for col in numeric_cols:
            var = ctk.BooleanVar(value=True)
            check = ctk.CTkCheckBox(
                self.var_scroll_frame,
                text=col,
                variable=var
            )
            check.pack(anchor="w", pady=2)
            self.column_vars[col] = var
    
    def _get_selected_columns(self):
        """Retorna colunas selecionadas"""
        return [col for col, var in self.column_vars.items() if var.get()]
    
    def _analyze(self):
        """Executa análise de correlação"""
        selected_cols = self._get_selected_columns()
        
        if len(selected_cols) < 2:
            messagebox.showwarning("Aviso", "Selecione pelo menos 2 variáveis")
            return
        
        try:
            self.selected_data = self.data[selected_cols]
            
            # Validar
            is_valid, error_msg = validate_multivariate_data(self.selected_data)
            if not is_valid:
                messagebox.showerror("Erro", error_msg)
                return
            
            # Calcular correlação com p-values
            self.corr_matrix, self.pvalue_matrix = calculate_correlation_with_pvalues(self.selected_data)
            
            # Mostrar resultados
            self._display_results()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao calcular correlação:\n{str(e)}")
    
    def _calculate_vif_analysis(self):
        """Calcula VIF para multicolinearidade"""
        if self.selected_data is None:
            messagebox.showwarning("Aviso", "Execute a análise de correlação primeiro")
            return
        
        try:
            self.vif_data = calculate_vif(self.selected_data)
            self._display_vif()
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao calcular VIF:\n{str(e)}")
    
    def _display_results(self):
        """Mostra resultados"""
        # Limpar anterior
        for widget in self.results_frame.winfo_children():
            if widget.winfo_class() != 'CTkLabel' or widget.cget('text') != 'Resultados da Análise':
                widget.destroy()
        
        # Tabela de correlação
        self._display_correlation_table()
        
        # Visualizações
        self._display_visualizations()
    
    def _display_correlation_table(self):
        """Mostra tabela de correlação com p-values"""
        table_frame = ctk.CTkFrame(self.results_frame)
        table_frame.pack(fill="x", padx=15, pady=10)
        
        title = ctk.CTkLabel(
            table_frame,
            text="Matriz de Correlação (com significância estatística)",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        title.pack(pady=10)
        
        # Criar tabela
        columns = self.selected_data.columns.tolist()
        
        scroll_frame = ctk.CTkScrollableFrame(table_frame, height=200)
        scroll_frame.pack(fill="x", padx=15, pady=10)
        
        # Headers
        header_frame = ctk.CTkFrame(scroll_frame)
        header_frame.pack(fill="x")
        
        ctk.CTkLabel(
            header_frame,
            text="",
            width=120,
            height=30,
            fg_color="#1f538d",
            corner_radius=0
        ).grid(row=0, column=0, sticky="ew", padx=1, pady=1)
        
        for j, col in enumerate(columns):
            label = ctk.CTkLabel(
                header_frame,
                text=col,
                font=ctk.CTkFont(size=10, weight="bold"),
                fg_color="#1f538d",
                corner_radius=0,
                width=100,
                height=30
            )
            label.grid(row=0, column=j+1, sticky="ew", padx=1, pady=1)
        
        # Dados
        for i, row_name in enumerate(columns):
            row_frame = ctk.CTkFrame(scroll_frame)
            row_frame.pack(fill="x")
            
            # Nome da linha
            label = ctk.CTkLabel(
                row_frame,
                text=row_name,
                font=ctk.CTkFont(size=10, weight="bold"),
                fg_color="#2b2b2b",
                corner_radius=0,
                width=120,
                height=25
            )
            label.grid(row=0, column=0, sticky="ew", padx=1, pady=1)
            
            for j in range(len(columns)):
                corr_val = self.corr_matrix[i, j]
                pval = self.pvalue_matrix[i, j]
                
                # Interpretar
                if i != j:
                    interp = interpret_correlation(corr_val, pval)
                    text = f"{corr_val:.3f}{interp['significance']}"
                    
                    # Cor baseada em força
                    if abs(corr_val) >= 0.7:
                        fg_color = "#2d5016" if pval < 0.05 else "#3d4d26"
                    elif abs(corr_val) >= 0.5:
                        fg_color = "#5a4a1a" if pval < 0.05 else "#4a4a2a"
                    else:
                        fg_color = "#2b2b2b"
                else:
                    text = "1.000"
                    fg_color = "#1f538d"
                
                label = ctk.CTkLabel(
                    row_frame,
                    text=text,
                    font=ctk.CTkFont(size=10),
                    fg_color=fg_color,
                    corner_radius=0,
                    width=100,
                    height=25
                )
                label.grid(row=0, column=j+1, sticky="ew", padx=1, pady=1)
        
        # Legenda
        legend_label = ctk.CTkLabel(
            table_frame,
            text="*** p<0.001  ** p<0.01  * p<0.05  ns não significativo",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        legend_label.pack(pady=5)
    
    def _display_visualizations(self):
        """Mostra visualizações"""
        viz_frame = ctk.CTkFrame(self.results_frame)
        viz_frame.pack(fill="both", expand=True, padx=15, pady=10)
        
        title = ctk.CTkLabel(
            viz_frame,
            text="Visualizações",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        title.pack(pady=10)
        
        # Criar figura com 4 subplots
        fig = Figure(figsize=(16, 12))
        
        columns = self.selected_data.columns.tolist()
        
        # 1. Heatmap de correlação
        ax1 = fig.add_subplot(221)
        sns.heatmap(
            self.corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            xticklabels=columns,
            yticklabels=columns,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax1
        )
        ax1.set_title("Heatmap de Correlação", fontsize=12, weight='bold')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=9)
        plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=9)
        
        # 2. Heatmap de p-values
        ax2 = fig.add_subplot(222)
        
        # Criar máscara para p-values significativos
        pvalue_display = np.where(self.pvalue_matrix < 0.05, self.pvalue_matrix, np.nan)
        
        sns.heatmap(
            self.pvalue_matrix,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd_r",
            vmin=0,
            vmax=0.05,
            xticklabels=columns,
            yticklabels=columns,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "P-value"},
            ax=ax2
        )
        ax2.set_title("P-Values (vermelho = significativo)", fontsize=12, weight='bold')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", fontsize=9)
        plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=9)
        
        # 3. Dendrograma de clustering hierárquico
        ax3 = fig.add_subplot(223)
        try:
            linkage_matrix = calculate_hierarchical_clustering(self.corr_matrix)
            hierarchy.dendrogram(
                linkage_matrix,
                labels=columns,
                ax=ax3,
                leaf_font_size=9,
                color_threshold=0.7
            )
            ax3.set_title("Clustering Hierárquico de Variáveis", fontsize=12, weight='bold')
            ax3.set_xlabel("Variáveis", fontsize=10)
            ax3.set_ylabel("Distância (1 - |correlação|)", fontsize=10)
            plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
        except:
            ax3.text(0.5, 0.5, "Erro ao gerar dendrograma", 
                    ha='center', va='center', transform=ax3.transAxes)
        
        # 4. Scatter plot das duas variáveis mais correlacionadas
        ax4 = fig.add_subplot(224)
        
        # Encontrar par com maior correlação (exceto diagonal)
        corr_no_diag = self.corr_matrix.copy()
        np.fill_diagonal(corr_no_diag, 0)
        max_idx = np.unravel_index(np.abs(corr_no_diag).argmax(), corr_no_diag.shape)
        
        var1 = columns[max_idx[0]]
        var2 = columns[max_idx[1]]
        corr_value = corr_no_diag[max_idx]
        pval = self.pvalue_matrix[max_idx]
        
        ax4.scatter(self.selected_data[var1], self.selected_data[var2], 
                   alpha=0.6, s=50, edgecolors='black')
        
        # Linha de tendência
        z = np.polyfit(self.selected_data[var1], self.selected_data[var2], 1)
        p = np.poly1d(z)
        ax4.plot(self.selected_data[var1], p(self.selected_data[var1]), 
                "r-", linewidth=2, alpha=0.8)
        
        ax4.set_xlabel(var1, fontsize=10)
        ax4.set_ylabel(var2, fontsize=10)
        ax4.set_title(f"Maior Correlação: r={corr_value:.3f}, p={pval:.4f}", 
                     fontsize=12, weight='bold')
        ax4.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        # Adicionar canvas
        canvas = FigureCanvasTkAgg(fig, viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def _display_vif(self):
        """Mostra análise VIF"""
        # Criar frame para VIF
        vif_frame = ctk.CTkFrame(self.results_frame)
        vif_frame.pack(fill="x", padx=15, pady=10)
        
        title = ctk.CTkLabel(
            vif_frame,
            text="VIF - Variance Inflation Factor (Multicolinearidade)",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        title.pack(pady=10)
        
        # Tabela VIF
        table_frame = ctk.CTkFrame(vif_frame)
        table_frame.pack(padx=15, pady=10)
        
        # Headers
        headers = ['Variável', 'VIF', 'Status', 'Interpretação']
        for col_idx, header in enumerate(headers):
            width = 120 if col_idx == 0 else 100 if col_idx == 1 else 150
            label = ctk.CTkLabel(
                table_frame,
                text=header,
                font=ctk.CTkFont(size=11, weight="bold"),
                fg_color="#1f538d",
                corner_radius=0,
                width=width,
                height=30
            )
            label.grid(row=0, column=col_idx, sticky="ew", padx=1, pady=1)
        
        # Dados
        sorted_vif = sorted(self.vif_data.items(), key=lambda x: x[1], reverse=True)
        
        for row_idx, (var_name, vif_value) in enumerate(sorted_vif):
            interp = interpret_vif(vif_value)
            
            color_map = {'green': '#2d5016', 'yellow': '#5a4a1a', 'red': '#5a1a1a'}
            fg_color = color_map.get(interp['color'], '#2b2b2b')
            
            # Nome
            label = ctk.CTkLabel(
                table_frame,
                text=var_name,
                font=ctk.CTkFont(size=10),
                fg_color="#2b2b2b",
                corner_radius=0,
                width=120,
                height=25
            )
            label.grid(row=row_idx + 1, column=0, sticky="ew", padx=1, pady=1)
            
            # VIF
            vif_text = f"{vif_value:.2f}" if not np.isinf(vif_value) else "∞"
            label = ctk.CTkLabel(
                table_frame,
                text=vif_text,
                font=ctk.CTkFont(size=10),
                fg_color=fg_color,
                corner_radius=0,
                width=100,
                height=25
            )
            label.grid(row=row_idx + 1, column=1, sticky="ew", padx=1, pady=1)
            
            # Status
            label = ctk.CTkLabel(
                table_frame,
                text=interp['status'],
                font=ctk.CTkFont(size=10),
                fg_color=fg_color,
                corner_radius=0,
                width=150,
                height=25
            )
            label.grid(row=row_idx + 1, column=2, sticky="ew", padx=1, pady=1)
            
            # Mensagem
            label = ctk.CTkLabel(
                table_frame,
                text=interp['message'],
                font=ctk.CTkFont(size=9),
                fg_color="#2b2b2b",
                corner_radius=0,
                width=250,
                height=25,
                anchor="w"
            )
            label.grid(row=row_idx + 1, column=3, sticky="ew", padx=1, pady=1)
        
        # Nota explicativa
        note = ctk.CTkLabel(
            vif_frame,
            text="VIF < 5: Sem problemas  |  VIF 5-10: Moderado  |  VIF > 10: Alto (remover variável)",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        note.pack(pady=5)
    
    def _clear_results(self):
        """Limpa resultados"""
        for widget in self.results_frame.winfo_children():
            if widget.winfo_class() != 'CTkLabel' or widget.cget('text') != 'Resultados da Análise':
                widget.destroy()
        
        self.corr_matrix = None
        self.pvalue_matrix = None
        self.vif_data = None
