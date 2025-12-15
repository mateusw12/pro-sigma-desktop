"""
Interface gráfica para Análise de Regressão Não Linear
"""

import customtkinter as ctk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.utils.lazy_imports import get_pandas, get_numpy
from src.utils.ui_components import (
    create_minitab_style_table,
    add_chart_export_button
)
from src.analytics.nonlinear.nonlinear_utils import (
    calculate_nonlinear_regression,
    get_model_name_pt,
    get_model_equation
)


class NonlinearWindow(ctk.CTkToplevel):
    """Janela de Análise de Regressão Não Linear"""
    
    def __init__(self, parent, data):
        super().__init__(parent)
        
        self.title("Regressão Não Linear")
        self.geometry("1400x900")
        
        # Maximiza a janela
        self.state('zoomed')
        
        self.data = data
        self.results = None
        self.selected_model = None
        
        # Referências para frames que precisam ser atualizados
        self.equation_label = None
        self.params_scroll_frame = None
        
        # Configuração de cor de fundo
        self.configure(fg_color="#2b2b2b")
        
        self._create_widgets()
        self._populate_columns()
    
    def _create_widgets(self):
        """Cria os widgets da interface"""
        
        # Frame principal com scroll
        main_frame = ctk.CTkScrollableFrame(self, fg_color="#2b2b2b")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # ===== SELEÇÃO DE COLUNAS =====
        selection_frame = ctk.CTkFrame(main_frame, fg_color="#1e1e1e", corner_radius=10)
        selection_frame.pack(fill="x", padx=10, pady=10)
        
        title_label = ctk.CTkLabel(
            selection_frame,
            text="📊 Seleção de Variáveis",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#ffffff"
        )
        title_label.pack(pady=10)
        
        # Grid para seleção
        selection_grid = ctk.CTkFrame(selection_frame, fg_color="transparent")
        selection_grid.pack(fill="x", padx=20, pady=10)
        
        # Coluna X
        ctk.CTkLabel(
            selection_grid,
            text="Coluna X:",
            font=ctk.CTkFont(size=13),
            text_color="#ffffff"
        ).grid(row=0, column=0, sticky="w", padx=5, pady=5)
        
        self.x_combo = ctk.CTkComboBox(
            selection_grid,
            width=250,
            state="readonly"
        )
        self.x_combo.grid(row=0, column=1, padx=5, pady=5)
        self.x_combo.set("")
        
        # Coluna Y
        ctk.CTkLabel(
            selection_grid,
            text="Coluna Y:",
            font=ctk.CTkFont(size=13),
            text_color="#ffffff"
        ).grid(row=1, column=0, sticky="w", padx=5, pady=5)
        
        self.y_combo = ctk.CTkComboBox(
            selection_grid,
            width=250,
            state="readonly"
        )
        self.y_combo.grid(row=1, column=1, padx=5, pady=5)
        self.y_combo.set("")
        
        # Botão calcular
        calculate_button = ctk.CTkButton(
            selection_frame,
            text="🔍 Calcular Modelos",
            command=self._calculate_analysis,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#1f6aa5",
            hover_color="#144870",
            height=40,
            width=200
        )
        calculate_button.pack(pady=15)
        
        # ===== FRAME DE RESULTADOS =====
        self.results_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        self.results_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    def _populate_columns(self):
        """Popula comboboxes com colunas disponíveis"""
        if self.data is None:
            return
        
        pd = get_pandas()
        
        # Converte colunas datetime para string
        for col in self.data.columns:
            if pd.api.types.is_datetime64_any_dtype(self.data[col]):
                self.data[col] = self.data[col].astype(str)
        
        # Identifica colunas numéricas
        numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) < 2:
            messagebox.showerror(
                "Erro",
                "É necessário ter pelo menos 2 colunas numéricas para análise de regressão não linear."
            )
            self.destroy()
            return
        
        # Popula comboboxes
        self.x_combo.configure(values=numeric_cols)
        self.x_combo.set(numeric_cols[0])
        
        self.y_combo.configure(values=numeric_cols)
        self.y_combo.set(numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0])
    
    def _calculate_analysis(self):
        """Calcula a análise de regressão não linear"""
        x_column = self.x_combo.get()
        y_column = self.y_combo.get()
        
        if not x_column or not y_column:
            messagebox.showwarning("Aviso", "Selecione as colunas X e Y")
            return
        
        if x_column == y_column:
            messagebox.showwarning("Aviso", "As colunas X e Y devem ser diferentes")
            return
        
        try:
            # Calcula regressão
            self.results = calculate_nonlinear_regression(
                self.data,
                x_column,
                y_column
            )
            
            if not self.results["metrics"]:
                messagebox.showerror(
                    "Erro",
                    "Não foi possível ajustar nenhum modelo aos dados."
                )
                return
            
            # Seleciona modelo com melhor R²
            best_model = max(
                self.results["metrics"].items(),
                key=lambda x: x[1]["rSquared"]
            )[0]
            self.selected_model = best_model
            
            # Exibe resultados
            self._display_results(x_column, y_column)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao calcular análise:\n{str(e)}")
    
    def _display_results(self, x_column, y_column):
        """Exibe os resultados da análise"""
        
        # Limpa resultados anteriores
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Cria container para layout
        container = ctk.CTkFrame(self.results_frame, fg_color="transparent")
        container.pack(fill="both", expand=True)
        
        # ===== LADO ESQUERDO - GRÁFICO =====
        left_frame = ctk.CTkFrame(container, fg_color="#1e1e1e", corner_radius=10)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        # Título do gráfico
        chart_title = ctk.CTkLabel(
            left_frame,
            text="📈 Ajuste do Modelo",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#ffffff"
        )
        chart_title.pack(pady=10)
        
        # Frame para gráfico
        self.chart_frame = ctk.CTkFrame(left_frame, fg_color="#ffffff", corner_radius=5)
        self.chart_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self._create_chart(x_column, y_column)
        
        # ===== LADO DIREITO - TABELAS =====
        right_frame = ctk.CTkFrame(container, fg_color="transparent")
        right_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        # Tabela de resumo dos modelos
        summary_frame = ctk.CTkFrame(right_frame, fg_color="#1e1e1e", corner_radius=10)
        summary_frame.pack(fill="both", expand=True, pady=(0, 5))
        
        summary_title = ctk.CTkLabel(
            summary_frame,
            text="📊 Resumo dos Modelos",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#ffffff"
        )
        summary_title.pack(pady=10)
        
        self._create_summary_table(summary_frame)
        
        # Equação do modelo selecionado
        equation_frame = ctk.CTkFrame(right_frame, fg_color="#1e1e1e", corner_radius=10)
        equation_frame.pack(fill="x", pady=5)
        
        equation_title = ctk.CTkLabel(
            equation_frame,
            text="📐 Equação do Modelo",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#ffffff"
        )
        equation_title.pack(pady=10)
        
        equation_text = get_model_equation(self.selected_model)
        self.equation_label = ctk.CTkLabel(
            equation_frame,
            text=equation_text,
            font=ctk.CTkFont(size=12),
            text_color="#ffffff",
            wraplength=400
        )
        self.equation_label.pack(pady=10)
        
        # Tabela de parâmetros
        params_frame = ctk.CTkFrame(right_frame, fg_color="#1e1e1e", corner_radius=10)
        params_frame.pack(fill="both", expand=True, pady=(5, 0))
        
        params_title = ctk.CTkLabel(
            params_frame,
            text="🔢 Parâmetros do Modelo",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#ffffff"
        )
        params_title.pack(pady=10)
        
        # Frame container para a tabela que será recriada
        self.params_container = ctk.CTkFrame(params_frame, fg_color="transparent")
        self.params_container.pack(fill="both", expand=True, padx=10, pady=5)
        
        self._create_params_table()
    
    def _create_chart(self, x_column, y_column):
        """Cria o gráfico de dispersão com curva ajustada"""
        
        # Limpa chart_frame
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        # Cria figura matplotlib
        fig = Figure(figsize=(8, 6), dpi=100, facecolor='white')
        ax = fig.add_subplot(111)
        
        # Dados originais
        original = self.results["original"]
        x_orig = [p["x"] for p in original]
        y_orig = [p["y"] for p in original]
        
        # Plot dados originais
        ax.scatter(x_orig, y_orig, alpha=0.6, s=50, color='#1f77b4', label='Dados Observados')
        
        # Curva ajustada do modelo selecionado
        if self.selected_model in self.results["predictions"]:
            pred = self.results["predictions"][self.selected_model]
            x_pred = [p["x"] for p in pred]
            y_pred = [p["y"] for p in pred]
            
            model_name = get_model_name_pt(self.selected_model)
            ax.plot(x_pred, y_pred, 'r-', linewidth=2, label=f'{model_name}')
        
        ax.set_xlabel(x_column, fontsize=11, fontweight='bold')
        ax.set_ylabel(y_column, fontsize=11, fontweight='bold')
        ax.set_title('Regressão Não Linear', fontsize=13, fontweight='bold', pad=10)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        fig.tight_layout()
        
        # Canvas
        canvas = FigureCanvasTkAgg(fig, self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Botão de exportar
        export_frame = ctk.CTkFrame(self.chart_frame, fg_color="transparent")
        export_frame.pack(side="bottom", pady=5)
        add_chart_export_button(export_frame, fig, "regressao_nao_linear")
    
    def _create_summary_table(self, parent):
        """Cria tabela de resumo dos modelos"""
        
        # Prepara dados
        table_data = []
        for model_name, metrics in self.results["metrics"].items():
            table_data.append({
                "Modelo": get_model_name_pt(model_name),
                "R²": f"{metrics['rSquared']:.4f}",
                "AIC": f"{metrics['aic']:.2f}",
                "BIC": f"{metrics['bic']:.2f}",
                "_model_key": model_name  # chave interna
            })
        
        # Ordena por R² decrescente
        table_data.sort(key=lambda x: float(x["R²"]), reverse=True)
        
        # Frame com scroll
        scroll_frame = ctk.CTkScrollableFrame(
            parent,
            fg_color="#2b2b2b",
            height=250
        )
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Cabeçalhos
        headers = ["Modelo", "R²", "AIC", "BIC"]
        header_frame = ctk.CTkFrame(scroll_frame, fg_color="#1f6aa5")
        header_frame.pack(fill="x", pady=(0, 2))
        
        for i, header in enumerate(headers):
            label = ctk.CTkLabel(
                header_frame,
                text=header,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color="#ffffff",
                width=100
            )
            label.grid(row=0, column=i, padx=5, pady=5, sticky="ew")
        
        header_frame.grid_columnconfigure(0, weight=2)
        for i in range(1, len(headers)):
            header_frame.grid_columnconfigure(i, weight=1)
        
        # Linhas de dados com radio buttons
        self.model_var = ctk.StringVar(value=self.selected_model)
        
        for row_data in table_data:
            model_key = row_data["_model_key"]
            
            # Frame da linha
            row_frame = ctk.CTkFrame(scroll_frame, fg_color="#1e1e1e")
            row_frame.pack(fill="x", pady=1)
            
            # Radio button
            radio = ctk.CTkRadioButton(
                row_frame,
                text="",
                variable=self.model_var,
                value=model_key,
                command=lambda mk=model_key: self._on_model_selected(mk),
                width=20
            )
            radio.grid(row=0, column=0, padx=5, pady=5)
            
            # Dados
            for i, header in enumerate(headers):
                label = ctk.CTkLabel(
                    row_frame,
                    text=row_data[header],
                    font=ctk.CTkFont(size=10),
                    text_color="#ffffff",
                    width=100
                )
                label.grid(row=0, column=i+1, padx=5, pady=5, sticky="ew")
            
            row_frame.grid_columnconfigure(1, weight=2)
            for i in range(2, len(headers)+1):
                row_frame.grid_columnconfigure(i, weight=1)
    
    def _create_params_table(self):
        """Cria tabela de parâmetros do modelo selecionado"""
        
        # Limpa container anterior
        for widget in self.params_container.winfo_children():
            widget.destroy()
        
        if self.selected_model not in self.results["metrics"]:
            return
        
        coef = self.results["metrics"][self.selected_model]["coef"]
        
        # Prepara dados
        table_data = []
        for param_name, param_value in coef.items():
            table_data.append({
                "Parâmetro": param_name,
                "Valor": f"{param_value:.6f}"
            })
        
        # Frame com scroll
        scroll_frame = ctk.CTkScrollableFrame(
            self.params_container,
            fg_color="#2b2b2b",
            height=200
        )
        scroll_frame.pack(fill="both", expand=True)
        
        # Cabeçalhos
        headers = ["Parâmetro", "Valor"]
        header_frame = ctk.CTkFrame(scroll_frame, fg_color="#1f6aa5")
        header_frame.pack(fill="x", pady=(0, 2))
        
        for i, header in enumerate(headers):
            label = ctk.CTkLabel(
                header_frame,
                text=header,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color="#ffffff"
            )
            label.grid(row=0, column=i, padx=5, pady=5, sticky="ew")
        
        header_frame.grid_columnconfigure(0, weight=1)
        header_frame.grid_columnconfigure(1, weight=1)
        
        # Linhas de dados
        for row_data in table_data:
            row_frame = ctk.CTkFrame(scroll_frame, fg_color="#1e1e1e")
            row_frame.pack(fill="x", pady=1)
            
            for i, header in enumerate(headers):
                label = ctk.CTkLabel(
                    row_frame,
                    text=row_data[header],
                    font=ctk.CTkFont(size=10),
                    text_color="#ffffff"
                )
                label.grid(row=0, column=i, padx=5, pady=5, sticky="ew")
            
            row_frame.grid_columnconfigure(0, weight=1)
            row_frame.grid_columnconfigure(1, weight=1)
    
    def _on_model_selected(self, model_key):
        """Callback quando um modelo é selecionado na tabela"""
        self.selected_model = model_key
        
        # Atualiza gráfico
        x_column = self.x_combo.get()
        y_column = self.y_combo.get()
        self._create_chart(x_column, y_column)
        
        # Atualiza equação
        if self.equation_label:
            self.equation_label.configure(text=get_model_equation(self.selected_model))
        
        # Atualiza tabela de parâmetros
        self._create_params_table()
