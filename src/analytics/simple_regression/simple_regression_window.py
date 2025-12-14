"""
Simple Regression Window
Interface for simple linear regression analysis
"""
import customtkinter as ctk
from tkinter import messagebox
from src.utils.lazy_imports import get_pandas, get_numpy, get_matplotlib_figure, get_matplotlib_backend, get_matplotlib
from src.utils.ui_components import create_minitab_style_table

from .simple_regression_utils import (
    calculate_simple_regression,
    create_anova_table,
    create_coefficients_table,
    create_summary_table,
    create_regression_plot,
    create_residuals_plot,
    create_histogram_residuals
)

# Lazy-loaded libraries
_pd = None
_np = None
_plt = None
_Figure = None
_FigureCanvasTkAgg = None

def _ensure_libs():
    """Ensure heavy libraries are loaded"""
    global _pd, _np, _plt, _Figure, _FigureCanvasTkAgg
    if _pd is None:
        _pd = get_pandas()
        _np = get_numpy()
        _plt = get_matplotlib()
        _Figure = get_matplotlib_figure()
        _FigureCanvasTkAgg = get_matplotlib_backend()
    return _pd, _np, _plt, _Figure, _FigureCanvasTkAgg


class SimpleRegressionWindow(ctk.CTkToplevel):
    def __init__(self, parent, df):
        super().__init__(parent)
        
        # Load heavy libraries (lazy)
        pd, np, plt, Figure, FigureCanvasTkAgg = _ensure_libs()
        self.pd = pd
        self.np = np
        self.plt = plt
        self.Figure = Figure
        self.FigureCanvasTkAgg = FigureCanvasTkAgg
        
        self.df = df
        
        # Window configuration
        self.title("Regressão Linear Simples")
        
        # Allow resizing
        self.resizable(True, True)
        self.minsize(1200, 800)
        
        # Start maximized (full screen)
        self.state('zoomed')  # Windows maximized
        
        # Make modal
        self.transient(parent)
        self.grab_set()
        
        self.create_widgets()
    
    def _format_value(self, value):
        """Format numeric values with max 5 decimal places"""
        if isinstance(value, (int, float, self.np.number)):
            if abs(value) < 0.00001 and value != 0:
                return f"{value:.5e}"  # Scientific notation for very small numbers
            elif abs(value) >= 10000:
                return f"{value:.5g}"  # Compact format for large numbers
            else:
                # Round to max 5 decimals, remove trailing zeros
                formatted = f"{value:.5f}".rstrip('0').rstrip('.')
                return formatted if formatted else '0'
        return str(value)
    
    def create_widgets(self):
        # Main container with scrollable frame
        self.main_container = ctk.CTkScrollableFrame(self)
        self.main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title = ctk.CTkLabel(
            self.main_container,
            text="📈 Regressão Linear Simples",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=(0, 20))
        
        # Description
        desc = ctk.CTkLabel(
            self.main_container,
            text="Análise de regressão com uma variável independente (X) e uma variável dependente (Y)",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        desc.pack(pady=(0, 20))
        
        # Configuration Frame
        config_frame = ctk.CTkFrame(self.main_container)
        config_frame.pack(fill="x", pady=(0, 20))
        
        # Variable Selection Frame
        vars_frame = ctk.CTkFrame(config_frame)
        vars_frame.pack(fill="x", padx=20, pady=10)
        
        # Get numeric columns only
        numeric_columns = self.df.select_dtypes(include=[self.np.number]).columns.tolist()
        
        if len(numeric_columns) < 2:
            messagebox.showerror(
                "Erro",
                "São necessárias pelo menos 2 colunas numéricas para análise de regressão."
            )
            self.destroy()
            return
        
        # X Variable Selection
        x_frame = ctk.CTkFrame(vars_frame, fg_color="transparent")
        x_frame.pack(side="left", padx=20, pady=10, expand=True)
        
        ctk.CTkLabel(
            x_frame,
            text="Variável Independente (X):",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(0, 5))
        
        self.x_var = ctk.StringVar()
        x_menu = ctk.CTkOptionMenu(
            x_frame,
            variable=self.x_var,
            values=numeric_columns,
            width=250,
            font=ctk.CTkFont(size=12)
        )
        x_menu.pack()
        
        # Y Variable Selection
        y_frame = ctk.CTkFrame(vars_frame, fg_color="transparent")
        y_frame.pack(side="left", padx=20, pady=10, expand=True)
        
        ctk.CTkLabel(
            y_frame,
            text="Variável Dependente (Y):",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(0, 5))
        
        self.y_var = ctk.StringVar()
        y_menu = ctk.CTkOptionMenu(
            y_frame,
            variable=self.y_var,
            values=numeric_columns,
            width=250,
            font=ctk.CTkFont(size=12)
        )
        y_menu.pack()
        
        # Set default values
        if len(numeric_columns) >= 2:
            self.x_var.set(numeric_columns[0])
            self.y_var.set(numeric_columns[1])
        
        # Options Frame
        options_frame = ctk.CTkFrame(config_frame)
        options_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            options_frame,
            text="Opções de Visualização:",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        options_inner = ctk.CTkFrame(options_frame)
        options_inner.pack(fill="x", padx=10, pady=(0, 10))
        
        self.show_residuals_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            options_inner,
            text="Gráficos de Diagnóstico de Resíduos",
            variable=self.show_residuals_var,
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=10)
        
        self.show_histogram_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            options_inner,
            text="Histograma de Resíduos",
            variable=self.show_histogram_var,
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=10)
        
        # Generate button
        generate_btn = ctk.CTkButton(
            config_frame,
            text="📊 Executar Análise de Regressão",
            command=self.generate_analysis,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40
        )
        generate_btn.pack(pady=20)
        
        # Results container (initially empty)
        self.results_container = ctk.CTkFrame(self.main_container)
        self.results_container.pack(fill="both", expand=True, pady=(0, 20))
    
    def generate_analysis(self):
        """Generate regression analysis"""
        try:
            # Get selected variables
            x_col = self.x_var.get()
            y_col = self.y_var.get()
            
            if not x_col or not y_col:
                messagebox.showerror(
                    "Erro",
                    "Por favor, selecione as variáveis X e Y."
                )
                return
            
            if x_col == y_col:
                messagebox.showerror(
                    "Erro",
                    "As variáveis X e Y devem ser diferentes."
                )
                return
            
            # Clear previous results
            for widget in self.results_container.winfo_children():
                widget.destroy()
            
            # Get data and remove NaN
            data = self.df[[x_col, y_col]].dropna()
            
            if len(data) < 3:
                messagebox.showerror(
                    "Erro",
                    "São necessárias pelo menos 3 observações válidas para análise de regressão."
                )
                return
            
            X = data[x_col].values
            y = data[y_col].values
            
            # Calculate regression
            results = calculate_simple_regression(X, y)
            
            # Show results
            self.show_summary_table(results, x_col, y_col)
            self.show_anova_table(results)
            self.show_coefficients_table(results, x_col, y_col)
            self.show_regression_plot(X, y, results, x_col, y_col)
            
            # Optional diagnostic plots
            if self.show_residuals_var.get():
                self.show_residuals_plots(X, results, x_col)
            
            if self.show_histogram_var.get():
                self.show_histogram_plot(results)
            
        except Exception as e:
            messagebox.showerror(
                "Erro",
                f"Erro ao gerar análise:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
    
    def show_summary_table(self, results, x_col, y_col):
        """Show summary statistics table"""
        summary_frame = ctk.CTkFrame(self.results_container)
        summary_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            summary_frame,
            text="📊 Resumo do Modelo",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        summary_df = create_summary_table(results, x_col, y_col)
        
        headers = summary_df.columns.tolist()
        # Format values to max 5 decimal places
        data_rows = [[self._format_value(val) for val in row] for row in summary_df.values.tolist()]
        
        create_minitab_style_table(
            summary_frame,
            headers=headers,
            data_rows=data_rows,
            title="Estatísticas do Modelo"
        )
    
    def show_anova_table(self, results):
        """Show ANOVA table"""
        anova_frame = ctk.CTkFrame(self.results_container)
        anova_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            anova_frame,
            text="📋 Tabela ANOVA",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        anova_df = create_anova_table(results)
        
        headers = anova_df.columns.tolist()
        # Format values to max 5 decimal places
        data_rows = [[self._format_value(val) for val in row] for row in anova_df.values.tolist()]
        
        tree = create_minitab_style_table(
            anova_frame,
            headers=headers,
            data_rows=data_rows,
            title="Análise de Variância"
        )
        
        # Highlight significant p-values in red
        p_value_col_idx = None
        for idx, header in enumerate(headers):
            if 'p-value' in header.lower() or 'pvalue' in header.lower():
                p_value_col_idx = idx
                break
        
        if p_value_col_idx is not None:
            for item in tree.get_children():
                values = tree.item(item)['values']
                try:
                    p_val = float(values[p_value_col_idx])
                    if p_val < 0.05:
                        tree.tag_configure(item, foreground='#FF5555')
                        tree.item(item, tags=(item,))
                except (ValueError, TypeError):
                    pass
    
    def show_coefficients_table(self, results, x_col, y_col):
        """Show coefficients table"""
        coef_frame = ctk.CTkFrame(self.results_container)
        coef_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            coef_frame,
            text="📐 Coeficientes da Regressão",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        coef_df = create_coefficients_table(results, x_col, y_col)
        
        headers = coef_df.columns.tolist()
        # Format values to max 5 decimal places
        data_rows = [[self._format_value(val) for val in row] for row in coef_df.values.tolist()]
        
        tree = create_minitab_style_table(
            coef_frame,
            headers=headers,
            data_rows=data_rows,
            title="Estimativas dos Parâmetros"
        )
        
        # Highlight significant p-values in red
        p_value_col_idx = None
        for idx, header in enumerate(headers):
            if 'p-value' in header.lower() or 'pvalue' in header.lower():
                p_value_col_idx = idx
                break
        
        if p_value_col_idx is not None:
            for item in tree.get_children():
                values = tree.item(item)['values']
                try:
                    p_val = float(values[p_value_col_idx])
                    if p_val < 0.05:
                        tree.tag_configure(item, foreground='#FF5555')
                        tree.item(item, tags=(item,))
                except (ValueError, TypeError):
                    pass
        
        # Add interpretation
        interpretation = ctk.CTkTextbox(coef_frame, height=100, font=ctk.CTkFont(size=11))
        interpretation.pack(padx=20, pady=10, fill="x")
        
        intercept = results['coefficients'][0]
        slope = results['coefficients'][1]
        p_value_slope = results['p_values'][1]
        
        interp_text = f"Equação da Regressão: {y_col} = {intercept:.4f} + {slope:.4f} × {x_col}\n\n"
        
        if p_value_slope < 0.001:
            interp_text += f"✓ O coeficiente de {x_col} é altamente significativo (p < 0.001).\n"
        elif p_value_slope < 0.05:
            interp_text += f"✓ O coeficiente de {x_col} é significativo (p = {p_value_slope:.4f}).\n"
        else:
            interp_text += f"✗ O coeficiente de {x_col} não é significativo (p = {p_value_slope:.4f}).\n"
        
        if slope > 0:
            interp_text += f"  Para cada unidade de aumento em {x_col}, {y_col} aumenta em média {abs(slope):.4f}."
        else:
            interp_text += f"  Para cada unidade de aumento em {x_col}, {y_col} diminui em média {abs(slope):.4f}."
        
        interpretation.insert("1.0", interp_text)
        interpretation.configure(state="disabled")
    
    def show_regression_plot(self, X, y, results, x_col, y_col):
        """Show regression plot"""
        plot_frame = ctk.CTkFrame(self.results_container)
        plot_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            plot_frame,
            text="📈 Gráfico de Regressão",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        fig = create_regression_plot(X, y, results, x_col, y_col)
        
        canvas = self.FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    def show_residuals_plots(self, X, results, x_col):
        """Show residual diagnostic plots"""
        residuals_frame = ctk.CTkFrame(self.results_container)
        residuals_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            residuals_frame,
            text="🔍 Diagnóstico de Resíduos",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        fig = create_residuals_plot(X, results, x_col)
        
        canvas = self.FigureCanvasTkAgg(fig, master=residuals_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    def show_histogram_plot(self, results):
        """Show histogram of residuals"""
        hist_frame = ctk.CTkFrame(self.results_container)
        hist_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            hist_frame,
            text="📊 Distribuição dos Resíduos",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        fig = create_histogram_residuals(results)
        
        canvas = self.FigureCanvasTkAgg(fig, master=hist_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
