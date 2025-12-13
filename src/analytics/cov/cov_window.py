"""
COV EMS Analysis Window
Component of Variance - Expected Mean Squares
Supports both Nested (Hierarchical) and Crossed analysis
"""
import customtkinter as ctk
from tkinter import messagebox
from src.utils.lazy_imports import get_numpy, get_pandas, get_scipy_stats, get_matplotlib, get_matplotlib_figure, get_matplotlib_backend
from typing import List, Dict

from src.analytics.cov.cov_utils import (
    remove_punctuation,
    replace_data_frame,
    fit_linear_regression,
    calculate_anova_table,
    construct_main_effects_formula,
    construct_interaction_effects_formula,
    combine_strings,
    calculate_mean_and_amplitude,
    calculate_variation_table,
    check_balanced,
    calculate_mean_square,
    calculate_percent_total,
    get_replace_label_crossed
)




# Lazy-loaded libraries
_pd = None
_np = None
_stats = None
_plt = None
_Figure = None
_FigureCanvasTkAgg = None

def _ensure_libs():
    """Carrega bibliotecas pesadas apenas quando necessário"""
    global _pd, _np, _stats, _plt, _Figure, _FigureCanvasTkAgg
    if _pd is None:
        _pd = get_pandas()
        _np = get_numpy()
        _stats = get_scipy_stats()
        _plt = get_matplotlib()
        _Figure = get_matplotlib_figure()
        _FigureCanvasTkAgg = get_matplotlib_backend()
    return _pd, _np, _stats, _plt, _Figure, _FigureCanvasTkAgg


class CovEmsWindow(ctk.CTkToplevel):
    def __init__(self, parent, df):
        super().__init__(parent)

        # Carrega bibliotecas pesadas (lazy)
        pd, np, stats, plt, Figure, FigureCanvasTkAgg = _ensure_libs()
        self.pd = pd
        self.np = np
        self.stats = stats
        self.plt = plt
        self.Figure = Figure
        self.FigureCanvasTkAgg = FigureCanvasTkAgg
        
        self.df = df
        self.results = None
        self.analysis_type = "crossed"  # crossed ou nested
        
        # Window configuration
        self.title("Análise COV EMS - Componentes de Variância")
        
        # Allow resizing
        self.resizable(True, True)
        self.minsize(1000, 700)
        
        # Start maximized
        self.state('zoomed')
        
        # Make modal
        self.transient(parent)
        self.grab_set()
        
        self.create_widgets()
    
    def create_widgets(self):
        # Main container
        self.main_container = ctk.CTkScrollableFrame(self)
        self.main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title = ctk.CTkLabel(
            self.main_container,
            text="📊 Análise COV EMS - Componentes de Variância",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=(0, 20))
        
        # Configuration Frame
        config_frame = ctk.CTkFrame(self.main_container)
        config_frame.pack(fill="x", pady=(0, 20))
        
        # Analysis Type Section
        type_section = ctk.CTkFrame(config_frame)
        type_section.pack(fill="x", padx=20, pady=20)
        
        ctk.CTkLabel(
            type_section,
            text="Tipo de Análise",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", pady=(0, 10))
        
        type_frame = ctk.CTkFrame(type_section, fg_color="transparent")
        type_frame.pack(fill="x", pady=5)
        
        self.analysis_type_var = ctk.StringVar(value="crossed")
        
        ctk.CTkRadioButton(
            type_frame,
            text="Cruzada (Crossed) - Efeitos independentes",
            variable=self.analysis_type_var,
            value="crossed",
            command=self.on_analysis_type_change
        ).pack(side="left", padx=10)
        
        ctk.CTkRadioButton(
            type_frame,
            text="Hierárquica (Nested) - Efeitos aninhados",
            variable=self.analysis_type_var,
            value="nested",
            command=self.on_analysis_type_change
        ).pack(side="left", padx=10)
        
        # Column Selection Section
        col_section = ctk.CTkFrame(config_frame)
        col_section.pack(fill="x", padx=20, pady=(0, 20))
        
        ctk.CTkLabel(
            col_section,
            text="Seleção de Colunas",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", pady=(0, 10))
        
        # Info text
        self.info_label = ctk.CTkLabel(
            col_section,
            text="Selecione os fatores X (causas) e as respostas Y que deseja analisar:",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.info_label.pack(anchor="w", pady=(0, 10))
        
        # X Columns (Factors) - Multiple selection
        x_frame = ctk.CTkFrame(col_section, fg_color="transparent")
        x_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(
            x_frame,
            text="Fatores X (causas/fontes):",
            width=200,
            font=ctk.CTkFont(weight="bold")
        ).pack(side="left", padx=(0, 10))
        
        # Scrollable frame for X checkboxes
        self.x_scroll_frame = ctk.CTkScrollableFrame(x_frame, height=150)
        self.x_scroll_frame.pack(side="left", fill="both", expand=True)
        
        self.x_checkboxes = []
        self.x_vars = []
        
        for col in self.df.columns:
            var = ctk.StringVar(value="off")
            cb = ctk.CTkCheckBox(
                self.x_scroll_frame,
                text=col,
                variable=var,
                onvalue="on",
                offvalue="off"
            )
            cb.pack(anchor="w", pady=2)
            self.x_checkboxes.append(cb)
            self.x_vars.append((col, var))
        
        # Y Columns (Responses) - Multiple selection
        y_frame = ctk.CTkFrame(col_section, fg_color="transparent")
        y_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(
            y_frame,
            text="Respostas Y (efeitos):",
            width=200,
            font=ctk.CTkFont(weight="bold")
        ).pack(side="left", padx=(0, 10))
        
        # Scrollable frame for Y checkboxes
        self.y_scroll_frame = ctk.CTkScrollableFrame(y_frame, height=100)
        self.y_scroll_frame.pack(side="left", fill="both", expand=True)
        
        self.y_checkboxes = []
        self.y_vars = []
        
        for col in self.df.columns:
            var = ctk.StringVar(value="off")
            cb = ctk.CTkCheckBox(
                self.y_scroll_frame,
                text=col,
                variable=var,
                onvalue="on",
                offvalue="off"
            )
            cb.pack(anchor="w", pady=2)
            self.y_checkboxes.append(cb)
            self.y_vars.append((col, var))
        
        # Action Buttons
        button_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        button_frame.pack(fill="x", padx=20, pady=20)
        
        self.calculate_btn = ctk.CTkButton(
            button_frame,
            text="🔍 Calcular COV EMS",
            command=self.calculate_cov,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#2E86DE"
        )
        self.calculate_btn.pack(side="left", padx=10)
        
        self.clear_btn = ctk.CTkButton(
            button_frame,
            text="🗑️ Limpar Resultados",
            command=self.clear_results,
            height=40,
            font=ctk.CTkFont(size=14),
            fg_color="#95A5A6"
        )
        self.clear_btn.pack(side="left", padx=10)
        
        # Results Container
        self.results_container = ctk.CTkFrame(self.main_container)
        self.results_container.pack(fill="both", expand=True, pady=(0, 20))
    
    def on_analysis_type_change(self):
        """Update UI when analysis type changes"""
        analysis_type = self.analysis_type_var.get()
        if analysis_type == "crossed":
            self.info_label.configure(
                text="Análise Cruzada: Fatores independentes (ex: Operador e Máquina)"
            )
        else:
            self.info_label.configure(
                text="Análise Hierárquica: Fatores aninhados (ex: Lote dentro de Fornecedor)"
            )
    
    def validate_inputs(self) -> bool:
        """Validate user inputs"""
        # Get selected X columns
        x_cols = [col for col, var in self.x_vars if var.get() == "on"]
        y_cols = [col for col, var in self.y_vars if var.get() == "on"]
        
        if len(x_cols) < 1:
            messagebox.showerror("Erro", "Selecione pelo menos um fator X!")
            return False
        
        if len(y_cols) < 1:
            messagebox.showerror("Erro", "Selecione pelo menos uma resposta Y!")
            return False
        
        # Check for overlap
        overlap = set(x_cols) & set(y_cols)
        if overlap:
            messagebox.showerror(
                "Erro",
                f"As colunas {overlap} não podem ser X e Y ao mesmo tempo!"
            )
            return False
        
        return True
    
    def calculate_cov(self):
        """Main calculation function"""
        if not self.validate_inputs():
            return
        
        self.calculate_btn.configure(state="disabled", text="⏳ Calculando...")
        self.update()
        
        try:
            # Get selected columns
            x_cols = [col for col, var in self.x_vars if var.get() == "on"]
            y_cols = [col for col, var in self.y_vars if var.get() == "on"]
            analysis_type = self.analysis_type_var.get()
            
            # Prepare data
            all_cols = x_cols + y_cols
            work_df = self.df[all_cols].copy()
            work_df = remove_punctuation(work_df)
            
            # Remove columns with all identical values
            work_df = replace_data_frame(x_cols.copy(), work_df)
            
            # Check if data is balanced
            is_balanced = check_balanced(work_df, analysis_type)
            
            if analysis_type == "nested" and not is_balanced:
                messagebox.showerror(
                    "Erro",
                    "Análise Nested requer dados balanceados!\n"
                    "Todos os fatores devem ter o mesmo número de repetições."
                )
                return
            
            # Calculate for each response
            self.results = {}
            
            for y_col in y_cols:
                if analysis_type == "crossed":
                    result = self.calculate_crossed(work_df, x_cols, y_col)
                else:
                    result = self.calculate_nested(work_df, x_cols, y_col)
                
                if result:
                    self.results[y_col] = result
            
            # Display results
            self.display_results(is_balanced)
            
        except Exception as e:
            messagebox.showerror("Erro no Cálculo", f"Erro ao calcular COV:\n{str(e)}")
            print(f"COV calculation error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.calculate_btn.configure(state="normal", text="🔍 Calcular COV EMS")
    
    def calculate_crossed(self, df, x_cols: List[str], y_col: str) -> Dict:
        """Calculate crossed (factorial) analysis"""
        # Prepare data
        analysis_df = df[x_cols + [y_col]].copy()
        analysis_df = analysis_df.dropna()
        
        # Rename Y column
        analysis_df.columns = [*analysis_df.columns[:-1], "Y"]
        columns = analysis_df.columns.tolist()
        
        # Check for duplicate rows (excluding response)
        df_without_response = analysis_df.drop(columns=["Y"])
        are_rows_unique = not df_without_response.duplicated().any()
        
        if are_rows_unique:
            # Remove last X column if all rows are unique
            last_column_name = df_without_response.columns[-1]
            analysis_df = analysis_df.drop(columns=[last_column_name])
            columns = analysis_df.columns.tolist()
        
        # Build formula
        main_effects = construct_main_effects_formula(columns)
        interaction_effects = construct_interaction_effects_formula(columns)
        
        s_string_splited = interaction_effects.split("+")
        combined_string = combine_strings(s_string_splited)
        
        total_string = "Y ~ " + main_effects
        if combined_string != "":
            total_string += " + " + combined_string
        
        # Fit model
        model = fit_linear_regression(analysis_df, total_string)
        
        if model is None:
            return None
        
        # Calculate ANOVA
        anova_table = calculate_anova_table(model)
        
        sum_sq = anova_table["sum_sq"]
        df_values = anova_table["df"]
        
        sum_mq = calculate_mean_square(sum_sq, df_values)
        
        total_sum_sq = sum(sum_sq)
        total_df = sum(df_values)
        
        total_percentual = calculate_percent_total(sum_sq, total_sum_sq)
        total_percentual.append("")
        
        # Build result
        sources = []
        variances = []
        within_total = 0
        
        for idx, key in enumerate(anova_table.index):
            source_name = "Erro" if key == "Residual" else get_replace_label_crossed(key)
            
            sources.append({
                "key": source_name,
                "df": df_values.iloc[idx] if idx < len(df_values) else "",
                "sSquare": sum_sq.iloc[idx] if idx < len(sum_sq) else total_sum_sq,
                "mSquare": sum_mq[idx] if idx < len(sum_mq) else (total_sum_sq / total_df),
                "fRatio": anova_table["F"].iloc[idx] if idx < len(anova_table["F"]) else "",
                "probF": anova_table["PR(>F)"].iloc[idx] if idx < len(anova_table["PR(>F)"]) else ""
            })
            
            variance_name = "Within" if key == "Residual" else get_replace_label_crossed(key)
            percentage = total_percentual[idx] if idx < len(total_percentual) else 100
            
            variances.append({
                "key": variance_name,
                "total": percentage
            })
            
            if key != "Residual" and idx < len(total_percentual):
                within_total += total_percentual[idx]
        
        # Fix Within percentage
        for v in variances:
            if v["key"] == "Within":
                v["total"] = 100 - within_total
        
        # Add Total row to variances
        variances.append({
            "key": "Total",
            "total": 100
        })
        
        # Add total row to sources
        sources.append({
            "key": "Total",
            "df": total_df,
            "sSquare": total_sum_sq,
            "mSquare": total_sum_sq / total_df,
            "fRatio": "",
            "probF": ""
        })
        
        return {
            "type": "crossed",
            "sources": sources,
            "variances": variances
        }
    
    def calculate_nested(self, df, x_cols: List[str], y_col: str) -> Dict:
        """Calculate nested (hierarchical) analysis"""
        analysis_df = df[x_cols + [y_col]].copy()
        analysis_df = analysis_df.dropna()
        
        columns = analysis_df.columns.tolist()
        columns_x = x_cols.copy()
        
        # Add line index
        line_quantity = list(self.np.arange(0, len(analysis_df)))
        analysis_df.insert(len(columns), "line", line_quantity)
        
        # Calculate mean and amplitude
        r_bar = calculate_mean_and_amplitude(analysis_df, columns + ["line"], columns_x, None)
        
        # Calculate variation table
        variation_table = calculate_variation_table(r_bar, columns_x)
        
        # Build variances result
        variances = []
        variables_reverse = columns_x.copy()
        variables_reverse.reverse()
        
        has_negative = False
        for var_name in variables_reverse:
            variance_val = variation_table[var_name]['variance']
            if variance_val < 0:
                has_negative = True
            
            variances.append({
                "key": var_name,
                "variance": variance_val,
                "desvpad": variation_table[var_name]['desvpad'],
                "total": variation_table[var_name]['percentage']
            })
        
        variances.append({
            "key": "Total",
            "variance": variation_table['total'],
            "desvpad": "",
            "total": 100
        })
        
        return {
            "type": "nested",
            "variances": variances,
            "has_negative": has_negative
        }
    
    def display_results(self, is_balanced: bool):
        """Display calculation results"""
        # Clear previous results
        for widget in self.results_container.winfo_children():
            widget.destroy()
        
        if not self.results:
            return
        
        # Results title
        title = ctk.CTkLabel(
            self.results_container,
            text="📈 Resultados da Análise COV EMS",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title.pack(pady=(10, 10))
        
        # Balanced status
        status_text = "✓ Dados Balanceados" if is_balanced else "⚠ Dados Não Balanceados"
        status_color = "green" if is_balanced else "orange"
        
        status_label = ctk.CTkLabel(
            self.results_container,
            text=status_text,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=status_color
        )
        status_label.pack(pady=(0, 20))
        
        # Create tabs for each response
        if len(self.results) > 1:
            tabview = ctk.CTkTabview(self.results_container)
            tabview.pack(fill="both", expand=True, padx=10, pady=10)
            
            for response_name, result_data in self.results.items():
                tab = tabview.add(response_name)
                self.display_result_for_response(tab, result_data, response_name)
        else:
            # Single response - no tabs needed
            response_name = list(self.results.keys())[0]
            result_data = self.results[response_name]
            self.display_result_for_response(self.results_container, result_data, response_name)
    
    def display_result_for_response(self, parent, result_data: Dict, response_name: str):
        """Display results for a single response"""
        content_frame = ctk.CTkFrame(parent)
        content_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left side - Tables
        left_frame = ctk.CTkFrame(content_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Right side - Chart
        right_frame = ctk.CTkFrame(content_frame)
        right_frame.pack(side="left", fill="both", expand=True)
        
        if result_data["type"] == "crossed":
            self.display_crossed_tables(left_frame, result_data)
        else:
            self.display_nested_table(left_frame, result_data)
        
        self.display_variance_chart(right_frame, result_data, response_name)
    
    def display_crossed_tables(self, parent, result_data: Dict):
        """Display ANOVA and variance tables for crossed analysis"""
        # ANOVA Table
        anova_frame = ctk.CTkFrame(parent)
        anova_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(
            anova_frame,
            text="Tabela ANOVA",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # Create scrollable frame for table
        table_scroll = ctk.CTkScrollableFrame(anova_frame, height=250)
        table_scroll.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Headers
        headers = ["Fonte", "GL", "SQ", "QM", "F", "Prob > F"]
        header_frame = ctk.CTkFrame(table_scroll)
        header_frame.pack(fill="x", pady=5)
        
        for col, header in enumerate(headers):
            ctk.CTkLabel(
                header_frame,
                text=header,
                font=ctk.CTkFont(weight="bold"),
                width=100
            ).grid(row=0, column=col, padx=5, pady=5)
        
        # Data rows
        for idx, row in enumerate(result_data["sources"]):
            row_frame = ctk.CTkFrame(table_scroll)
            row_frame.pack(fill="x", pady=2)
            
            # Color significant p-values
            is_significant = isinstance(row.get("probF"), (int, float)) and row.get("probF", 1) < 0.05
            text_color = "red" if is_significant else None
            
            ctk.CTkLabel(row_frame, text=row["key"], width=100).grid(row=0, column=0, padx=5, pady=3)
            ctk.CTkLabel(row_frame, text=f"{row['df']}" if row['df'] != "" else "", width=100).grid(row=0, column=1, padx=5, pady=3)
            
            sq_text = f"{row['sSquare']:.4f}" if isinstance(row['sSquare'], (int, float)) else ""
            ctk.CTkLabel(row_frame, text=sq_text, width=100).grid(row=0, column=2, padx=5, pady=3)
            
            mq_text = f"{row['mSquare']:.4f}" if isinstance(row['mSquare'], (int, float)) else ""
            ctk.CTkLabel(row_frame, text=mq_text, width=100).grid(row=0, column=3, padx=5, pady=3)
            
            f_text = f"{row['fRatio']:.4f}" if isinstance(row['fRatio'], (int, float)) else ""
            ctk.CTkLabel(row_frame, text=f_text, width=100).grid(row=0, column=4, padx=5, pady=3)
            
            if isinstance(row.get("probF"), (int, float)):
                prob_text = f"{row['probF']:.4f}" if row['probF'] >= 0.0001 else "< 0.0001*"
            else:
                prob_text = ""
            ctk.CTkLabel(row_frame, text=prob_text, width=100, text_color=text_color).grid(row=0, column=5, padx=5, pady=3)
        
        # Variance Components Table
        var_frame = ctk.CTkFrame(parent)
        var_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(
            var_frame,
            text="Componentes de Variância",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        var_table_frame = ctk.CTkFrame(var_frame)
        var_table_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        # Headers
        ctk.CTkLabel(var_table_frame, text="Componente", font=ctk.CTkFont(weight="bold"), width=200).grid(row=0, column=0, padx=5, pady=5)
        ctk.CTkLabel(var_table_frame, text="% Total", font=ctk.CTkFont(weight="bold"), width=150).grid(row=0, column=1, padx=5, pady=5)
        
        # Data
        for idx, row in enumerate(result_data["variances"], start=1):
            # Bold text for Total row
            font_weight = "bold" if row["key"] == "Total" else "normal"
            ctk.CTkLabel(var_table_frame, text=row["key"], width=200, font=ctk.CTkFont(weight=font_weight)).grid(row=idx, column=0, padx=5, pady=3)
            total_text = f"{row['total']:.2f}%" if isinstance(row['total'], (int, float)) else ""
            ctk.CTkLabel(var_table_frame, text=total_text, width=150, font=ctk.CTkFont(weight=font_weight)).grid(row=idx, column=1, padx=5, pady=3)
    
    def display_nested_table(self, parent, result_data: Dict):
        """Display variance table for nested analysis"""
        var_frame = ctk.CTkFrame(parent)
        var_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(
            var_frame,
            text="Componentes de Variância (Nested)",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # Warning for negative variances
        if result_data.get("has_negative"):
            warning_label = ctk.CTkLabel(
                var_frame,
                text="⚠ Atenção: Variâncias negativas detectadas (exibidas como 0)",
                text_color="orange",
                font=ctk.CTkFont(size=11)
            )
            warning_label.pack(pady=(0, 10))
        
        var_table_frame = ctk.CTkFrame(var_frame)
        var_table_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        # Headers
        headers = ["Componente", "Variância", "Desvio Padrão", "% Total"]
        for col, header in enumerate(headers):
            ctk.CTkLabel(
                var_table_frame,
                text=header,
                font=ctk.CTkFont(weight="bold"),
                width=120
            ).grid(row=0, column=col, padx=5, pady=5)
        
        # Data
        for idx, row in enumerate(result_data["variances"], start=1):
            ctk.CTkLabel(var_table_frame, text=row["key"], width=120).grid(row=idx, column=0, padx=5, pady=3)
            
            var_text = f"{max(0, row['variance']):.4e}" if isinstance(row.get('variance'), (int, float)) else ""
            ctk.CTkLabel(var_table_frame, text=var_text, width=120).grid(row=idx, column=1, padx=5, pady=3)
            
            std_text = f"{row['desvpad']:.4e}" if isinstance(row.get('desvpad'), (int, float)) else ""
            ctk.CTkLabel(var_table_frame, text=std_text, width=120).grid(row=idx, column=2, padx=5, pady=3)
            
            total_text = f"{row['total']:.2f}%" if isinstance(row.get('total'), (int, float)) else ""
            ctk.CTkLabel(var_table_frame, text=total_text, width=120).grid(row=idx, column=3, padx=5, pady=3)
    
    def display_variance_chart(self, parent, result_data: Dict, response_name: str):
        """Display variance components bar chart"""
        chart_frame = ctk.CTkFrame(parent)
        chart_frame.pack(fill="both", expand=True)
        
        ctk.CTkLabel(
            chart_frame,
            text=f"Componentes de Variância - {response_name}",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)
        
        # Prepare data
        labels = []
        values = []
        total_row = None
        
        for row in result_data["variances"]:
            if row["key"] == "Total":
                total_row = row
            else:
                labels.append(row["key"])
                values.append(row["total"] if isinstance(row["total"], (int, float)) else 0)
        
        # Sort by value
        sorted_data = sorted(zip(labels, values), key=lambda x: x[1])
        labels, values = zip(*sorted_data) if sorted_data else ([], [])
        
        # Add Total at the end
        if total_row:
            labels = list(labels) + ["Total"]
            values = list(values) + [total_row["total"] if isinstance(total_row["total"], (int, float)) else 0]
        
        # Create chart
        fig = self.Figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(111)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']
        bar_colors = [colors[i % len(colors)] for i in range(len(values))]
        
        # Highlight Total with different color
        if labels and labels[-1] == "Total":
            bar_colors[-1] = '#2ECC71'  # Green for Total
        
        bars = ax.barh(labels, values, color=bar_colors, edgecolor='black', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height() / 2.,
                    f'{width:.1f}%',
                    ha='left', va='center', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Contribuição (%)', fontsize=10, fontweight='bold')
        ax.set_title('Componentes de Variância', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        fig.tight_layout()
        
        canvas = self.FigureCanvasTkAgg(fig, chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    def clear_results(self):
        """Clear all results"""
        for widget in self.results_container.winfo_children():
            widget.destroy()
        
        self.results = None
