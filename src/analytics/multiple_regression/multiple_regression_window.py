"""
Multiple Regression Window
Interface for multiple linear regression analysis
"""
import customtkinter as ctk
from tkinter import messagebox
import tkinter as tk
from src.utils.lazy_imports import get_pandas, get_numpy, get_matplotlib_figure, get_matplotlib_backend, get_matplotlib
from src.utils.ui_components import create_minitab_style_table
from typing import List, Tuple

from .multiple_regression_utils import (
    calculate_multiple_regression,
    create_interaction_terms,
    backward_elimination,
    create_anova_table,
    create_coefficients_table,
    create_summary_table,
    create_regression_plot,
    create_residuals_plot,
    create_histogram_residuals,
    create_line_plot_predictions
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


class MultipleRegressionWindow(ctk.CTkToplevel):
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
        self.interaction_vars = {}  # Store interaction checkboxes
        
        # Window configuration
        self.title("Regressão Linear Múltipla")
        
        # Allow resizing
        self.resizable(True, True)
        self.minsize(1400, 900)
        
        # Start maximized (full screen)
        self.state('zoomed')  # Windows maximized
        
        # Make modal
        self.transient(parent)
        self.grab_set()
        
        self.create_widgets()
    
    def _format_value(self, value):
        """Format numeric values with max 5 decimal places"""
        if isinstance(value, (int, float, self.np.number)):
            if self.pd.isna(value):
                return '-'
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
        # Main container with scrollable frame (improved scroll)
        self.main_container = ctk.CTkScrollableFrame(
            self,
            scrollbar_button_color="gray30",
            scrollbar_button_hover_color="gray40"
        )
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Configure mouse wheel scrolling
        self.main_container._parent_canvas.configure(scrollregion=self.main_container._parent_canvas.bbox("all"))
        
        # Title
        title = ctk.CTkLabel(
            self.main_container,
            text="📊 Regressão Linear Múltipla",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=(0, 20))
        
        # Description
        desc = ctk.CTkLabel(
            self.main_container,
            text="Análise de regressão com múltiplas variáveis independentes (X) e dependentes (Y)\nSuporte para termos de interação e seleção de modelo",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        desc.pack(pady=(0, 20))
        
        # Configuration Frame
        config_frame = ctk.CTkFrame(self.main_container)
        config_frame.pack(fill="x", pady=(0, 20))
        
        # Get numeric and categorical columns
        numeric_columns = self.df.select_dtypes(include=[self.np.number]).columns.tolist()
        categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(numeric_columns) < 1:
            messagebox.showerror(
                "Erro",
                "É necessária pelo menos 1 coluna numérica para análise de regressão."
            )
            self.destroy()
            return
        
        # === X Variables Selection - Numeric (Multiple) ===
        x_frame = ctk.CTkFrame(config_frame)
        x_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            x_frame,
            text="Variáveis Independentes Numéricas (X):",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", pady=(10, 5), padx=10)
        
        # Scrollable frame for X checkboxes
        x_scroll = ctk.CTkScrollableFrame(x_frame, height=120)
        x_scroll.pack(fill="x", padx=10, pady=(0, 10))
        
        self.x_vars = {}
        for col in numeric_columns:
            var = tk.BooleanVar()
            cb = ctk.CTkCheckBox(
                x_scroll,
                text=col,
                variable=var,
                font=ctk.CTkFont(size=12),
                command=self.update_interactions
            )
            cb.pack(anchor="w", padx=10, pady=2)
            self.x_vars[col] = var
        
        # === X Categorical Variables Selection ===
        if categorical_columns:
            x_cat_frame = ctk.CTkFrame(config_frame)
            x_cat_frame.pack(fill="x", padx=20, pady=10)
            
            ctk.CTkLabel(
                x_cat_frame,
                text="Variáveis Independentes Categóricas (X):",
                font=ctk.CTkFont(size=14, weight="bold")
            ).pack(anchor="w", pady=(10, 5), padx=10)
            
            ctk.CTkLabel(
                x_cat_frame,
                text="⚠️ Variáveis categóricas serão codificadas automaticamente como variáveis dummy (k-1 categorias)",
                font=ctk.CTkFont(size=10),
                text_color="gray"
            ).pack(anchor="w", pady=(0, 5), padx=10)
            
            # Scrollable frame for categorical X checkboxes
            x_cat_scroll = ctk.CTkScrollableFrame(x_cat_frame, height=100)
            x_cat_scroll.pack(fill="x", padx=10, pady=(0, 10))
            
            self.x_cat_vars = {}
            for col in categorical_columns:
                var = tk.BooleanVar()
                # Show unique categories count
                n_categories = self.df[col].nunique()
                cb = ctk.CTkCheckBox(
                    x_cat_scroll,
                    text=f"{col} ({n_categories} categorias)",
                    variable=var,
                    font=ctk.CTkFont(size=12)
                )
                cb.pack(anchor="w", padx=10, pady=2)
                self.x_cat_vars[col] = var
        else:
            self.x_cat_vars = {}
        
        # === Interaction Terms ===
        interaction_frame = ctk.CTkFrame(config_frame)
        interaction_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            interaction_frame,
            text="Termos de Interação (X₁ × X₂):",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", pady=(10, 5), padx=10)
        
        self.interaction_scroll = ctk.CTkScrollableFrame(interaction_frame, height=100)
        self.interaction_scroll.pack(fill="x", padx=10, pady=(0, 10))
        
        ctk.CTkLabel(
            self.interaction_scroll,
            text="Selecione pelo menos 2 variáveis X para habilitar interações",
            text_color="gray",
            font=ctk.CTkFont(size=11)
        ).pack(pady=10)
        
        # === Y Variables Selection (Multiple) ===
        y_frame = ctk.CTkFrame(config_frame)
        y_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            y_frame,
            text="Variáveis Dependentes (Y) - uma análise para cada:",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", pady=(10, 5), padx=10)
        
        # Scrollable frame for Y checkboxes
        y_scroll = ctk.CTkScrollableFrame(y_frame, height=120)
        y_scroll.pack(fill="x", padx=10, pady=(0, 10))
        
        self.y_vars = {}
        for col in numeric_columns:
            var = tk.BooleanVar()
            cb = ctk.CTkCheckBox(
                y_scroll,
                text=col,
                variable=var,
                font=ctk.CTkFont(size=12)
            )
            cb.pack(anchor="w", padx=10, pady=2)
            self.y_vars[col] = var
        
        # === Options ===
        options_frame = ctk.CTkFrame(config_frame)
        options_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            options_frame,
            text="Opções de Análise:",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        options_inner = ctk.CTkFrame(options_frame)
        options_inner.pack(fill="x", padx=10, pady=(0, 10))
        
        # Model type selection
        model_type_frame = ctk.CTkFrame(options_inner, fg_color="transparent")
        model_type_frame.pack(side="left", padx=10, pady=5)
        
        ctk.CTkLabel(
            model_type_frame,
            text="Tipo de Modelo:",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(side="left", padx=(0, 10))
        
        self.model_type = ctk.StringVar(value="full")
        ctk.CTkRadioButton(
            model_type_frame,
            text="Completo (todas variáveis)",
            variable=self.model_type,
            value="full",
            font=ctk.CTkFont(size=11)
        ).pack(side="left", padx=5)
        
        ctk.CTkRadioButton(
            model_type_frame,
            text="Reduzido (backward elimination)",
            variable=self.model_type,
            value="reduced",
            font=ctk.CTkFont(size=11)
        ).pack(side="left", padx=5)
        
        # Visualization options
        viz_frame = ctk.CTkFrame(options_inner, fg_color="transparent")
        viz_frame.pack(side="left", padx=20, pady=5)
        
        self.show_residuals_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            viz_frame,
            text="Gráficos de Resíduos",
            variable=self.show_residuals_var,
            font=ctk.CTkFont(size=11)
        ).pack(side="left", padx=5)
        
        self.show_histogram_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            viz_frame,
            text="Histograma",
            variable=self.show_histogram_var,
            font=ctk.CTkFont(size=11)
        ).pack(side="left", padx=5)
        
        # Generate button
        generate_btn = ctk.CTkButton(
            config_frame,
            text="📊 Executar Análise de Regressão Múltipla",
            command=self.generate_analysis,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40
        )
        generate_btn.pack(pady=20)
        
        # Results container (initially empty)
        self.results_container = ctk.CTkFrame(self.main_container)
        self.results_container.pack(fill="both", expand=True, pady=(0, 20))
    
    def update_interactions(self):
        """Update interaction checkboxes based on selected X variables"""
        # Clear previous interaction widgets
        for widget in self.interaction_scroll.winfo_children():
            widget.destroy()
        
        self.interaction_vars = {}
        
        # Get selected X variables
        selected_x = [col for col, var in self.x_vars.items() if var.get()]
        
        if len(selected_x) < 2:
            ctk.CTkLabel(
                self.interaction_scroll,
                text="Selecione pelo menos 2 variáveis X para habilitar interações",
                text_color="gray",
                font=ctk.CTkFont(size=11)
            ).pack(pady=10)
            return
        
        # Create interaction checkboxes for all pairs
        ctk.CTkLabel(
            self.interaction_scroll,
            text=f"Selecione os termos de interação desejados:",
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(anchor="w", padx=5, pady=(5, 10))
        
        for i, x1 in enumerate(selected_x):
            for x2 in selected_x[i+1:]:
                interaction_name = f"{x1} × {x2}"
                var = tk.BooleanVar()
                cb = ctk.CTkCheckBox(
                    self.interaction_scroll,
                    text=interaction_name,
                    variable=var,
                    font=ctk.CTkFont(size=11)
                )
                cb.pack(anchor="w", padx=20, pady=2)
                self.interaction_vars[(x1, x2)] = var
    
    def generate_analysis(self):
        """Generate regression analysis"""
        try:
            # Get selected X variables (numeric)
            selected_x = [col for col, var in self.x_vars.items() if var.get()]
            
            # Get selected X variables (categorical)
            selected_x_cat = [col for col, var in self.x_cat_vars.items() if var.get()]
            
            # Get selected Y variables
            selected_y = [col for col, var in self.y_vars.items() if var.get()]
            
            # Validation
            if len(selected_x) < 1 and len(selected_x_cat) < 1:
                messagebox.showerror(
                    "Erro",
                    "Por favor, selecione pelo menos uma variável X (numérica ou categórica)."
                )
                return
            
            if len(selected_y) < 1:
                messagebox.showerror(
                    "Erro",
                    "Por favor, selecione pelo menos uma variável Y."
                )
                return
            
            # Check overlap
            overlap = (set(selected_x) | set(selected_x_cat)) & set(selected_y)
            if overlap:
                messagebox.showerror(
                    "Erro",
                    f"As seguintes variáveis não podem ser X e Y ao mesmo tempo:\n{', '.join(overlap)}"
                )
                return
            
            # Get selected interactions (only for numeric variables)
            selected_interactions = [(x1, x2) for (x1, x2), var in self.interaction_vars.items() if var.get()]
            
            # Clear previous results
            for widget in self.results_container.winfo_children():
                widget.destroy()
            
            # Perform analysis for each Y
            for y_idx, y_col in enumerate(selected_y):
                self.analyze_for_y(y_col, selected_x, selected_x_cat, selected_interactions, y_idx)
            
            # Update scroll region to accommodate all results
            self.main_container.update_idletasks()
            self.main_container._parent_canvas.configure(scrollregion=self.main_container._parent_canvas.bbox("all"))
            
            # Scroll to top to show results
            self.main_container._parent_canvas.yview_moveto(0)
            
        except Exception as e:
            messagebox.showerror(
                "Erro",
                f"Erro ao gerar análise:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
    
    def analyze_for_y(self, y_col: str, x_cols: List[str], x_cat_cols: List[str], 
                     interactions: List[Tuple[str, str]], y_index: int):
        """Perform regression analysis for a specific Y variable"""
        from .multiple_regression_utils import encode_categorical_variables
        
        # Create separator if not first Y
        if y_index > 0:
            separator = ctk.CTkFrame(self.results_container, height=3, fg_color="gray40")
            separator.pack(fill="x", padx=20, pady=20)
        
        # Y Title
        y_title_frame = ctk.CTkFrame(self.results_container)
        y_title_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            y_title_frame,
            text=f"🎯 Análise para Variável Dependente: {y_col}",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#2E86DE"
        ).pack(pady=15)
        
        # Prepare data
        all_cols = x_cols + x_cat_cols + [y_col]
        data = self.df[all_cols].dropna()
        
        total_predictors = len(x_cols) + len(x_cat_cols)
        if len(data) < total_predictors + 2:
            error_label = ctk.CTkLabel(
                self.results_container,
                text=f"⚠️ Dados insuficientes para {y_col}: mínimo {total_predictors + 2} observações necessárias",
                text_color="orange",
                font=ctk.CTkFont(size=12)
            )
            error_label.pack(pady=10)
            return
        
        # Create X DataFrame - start with numeric variables
        X_df = data[x_cols].copy() if x_cols else self.pd.DataFrame(index=data.index)
        
        # Encode categorical variables if any
        dummy_mapping = {}
        if x_cat_cols:
            cat_data = data[x_cat_cols].copy()
            encoded_cat_df, dummy_mapping = encode_categorical_variables(cat_data, x_cat_cols)
            
            # Merge encoded categorical with numeric X
            if not X_df.empty:
                X_df = self.pd.concat([X_df, encoded_cat_df], axis=1)
            else:
                X_df = encoded_cat_df
            
            # Show encoding information
            info_text = "📋 Codificação de Variáveis Categóricas:\n"
            for cat_var, mapping in dummy_mapping.items():
                ref_cat = mapping['reference']
                dummy_names = mapping['dummies']
                info_text += f"  • {cat_var}: Referência = '{ref_cat}', Dummies = {len(dummy_names)}\n"
            
            info_label = ctk.CTkLabel(
                self.results_container,
                text=info_text,
                font=ctk.CTkFont(size=10),
                text_color="gray",
                justify="left"
            )
            info_label.pack(pady=5, padx=20, anchor="w")
        
        # Add interaction terms (only for numeric variables)
        if interactions:
            X_df = create_interaction_terms(X_df, interactions)
        
        X = X_df.values
        y = data[y_col].values
        variable_names = X_df.columns.tolist()
        
        # Calculate regression
        model_type = self.model_type.get()
        
        if model_type == "reduced":
            # Backward elimination
            selected_indices, results = backward_elimination(X, y, variable_names, alpha=0.05)
            
            # Show info about reduction
            if len(selected_indices) < len(variable_names):
                removed_vars = [var for i, var in enumerate(variable_names) if i not in selected_indices]
                info_label = ctk.CTkLabel(
                    self.results_container,
                    text=f"ℹ️ Modelo Reduzido: {len(removed_vars)} variável(is) removida(s) - {', '.join(removed_vars)}",
                    font=ctk.CTkFont(size=11),
                    text_color="gray"
                )
                info_label.pack(pady=5)
        else:
            # Full model
            results = calculate_multiple_regression(X, y, variable_names)
        
        # Show results
        self.show_summary_table(results, y_col)
        self.show_anova_table(results)
        self.show_coefficients_table(results)
        self.show_regression_plot(y, results, y_col)
        self.show_line_plot(y, results, y_col)
        
        # Optional diagnostic plots
        if self.show_residuals_var.get():
            self.show_residuals_plots(results, y_col)
        
        if self.show_histogram_var.get():
            self.show_histogram_plot(results)
    
    def show_summary_table(self, results, y_col):
        """Show summary statistics table"""
        summary_frame = ctk.CTkFrame(self.results_container)
        summary_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            summary_frame,
            text="📊 Resumo do Modelo",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        summary_df = create_summary_table(results, y_col)
        
        headers = summary_df.columns.tolist()
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
                    p_val_str = str(values[p_value_col_idx])
                    if p_val_str and p_val_str != '-':
                        p_val = float(p_val_str)
                        if p_val < 0.05:
                            tree.tag_configure(item, foreground='#FF5555')
                            tree.item(item, tags=(item,))
                except (ValueError, TypeError):
                    pass
    
    def show_coefficients_table(self, results):
        """Show coefficients table"""
        coef_frame = ctk.CTkFrame(self.results_container)
        coef_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            coef_frame,
            text="📐 Coeficientes da Regressão",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        coef_df = create_coefficients_table(results)
        
        headers = coef_df.columns.tolist()
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
                    p_val_str = str(values[p_value_col_idx])
                    if p_val_str and p_val_str != '-':
                        p_val = float(p_val_str)
                        if p_val < 0.05:
                            tree.tag_configure(item, foreground='#FF5555')
                            tree.item(item, tags=(item,))
                except (ValueError, TypeError):
                    pass
        
        # Add VIF interpretation
        interpretation = ctk.CTkTextbox(coef_frame, height=80, font=ctk.CTkFont(size=11))
        interpretation.pack(padx=20, pady=10, fill="x")
        
        interp_text = "Interpretação VIF (Variance Inflation Factor):\n"
        interp_text += "• VIF < 5: Multicolinearidade baixa ✓\n"
        interp_text += "• VIF 5-10: Multicolinearidade moderada ⚠\n"
        interp_text += "• VIF > 10: Multicolinearidade alta (considere remover variável) ✗"
        
        interpretation.insert("1.0", interp_text)
        interpretation.configure(state="disabled")
    
    def show_regression_plot(self, y_true, results, y_col):
        """Show regression plot"""
        plot_frame = ctk.CTkFrame(self.results_container)
        plot_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            plot_frame,
            text="📈 Valores Preditos vs Reais (Scatter)",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        fig = create_regression_plot(y_true, results['y_pred'], y_col, results)
        
        canvas = self.FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    def show_line_plot(self, y_true, results, y_col):
        """Show line plot of predictions"""
        plot_frame = ctk.CTkFrame(self.results_container)
        plot_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            plot_frame,
            text="📉 Comparação Real vs Predito (Linha)",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        fig = create_line_plot_predictions(y_true, results['y_pred'], y_col, results)
        
        canvas = self.FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    def show_residuals_plots(self, results, y_col):
        """Show residual diagnostic plots"""
        residuals_frame = ctk.CTkFrame(self.results_container)
        residuals_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            residuals_frame,
            text="🔍 Diagnóstico de Resíduos",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        fig = create_residuals_plot(results['y_pred'], results, y_col)
        
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
