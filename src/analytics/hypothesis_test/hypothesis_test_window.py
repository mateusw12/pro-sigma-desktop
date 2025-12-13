"""
Hypothesis Test Window
Statistical hypothesis testing with multiple test types
"""
import customtkinter as ctk
from tkinter import messagebox
import pandas as pd

from src.analytics.hypothesis_test.hypothesis_test_utils import (
    calculate_mean_difference,
    calculate_one_way_anova,
    calculate_t_test_expected_mean,
    calculate_t_test_sample
)
from src.analytics.hypothesis_test.mean_difference.mean_difference_display import display_mean_difference_results
from src.analytics.hypothesis_test.one_way_anova.one_way_anova_display import display_anova_results
from src.analytics.hypothesis_test.test_t.t_test_expected_display import display_t_test_expected_results
from src.analytics.hypothesis_test.test_t.t_test_sample_display import display_t_test_sample_results


class HypothesisTestWindow(ctk.CTkToplevel):
    def __init__(self, parent, df: pd.DataFrame):
        super().__init__(parent)
        
        self.df = df
        self.results = None
        self.test_type = "mean_difference"  # mean_difference, one_way_anova, t_test_expected
        
        # Window configuration
        self.title("Teste de Hipótese")
        
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
            text="📊 Teste de Hipótese Estatística",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=(0, 20))
        
        # Configuration Frame
        config_frame = ctk.CTkFrame(self.main_container)
        config_frame.pack(fill="x", pady=(0, 20))
        
        # Test Type Section
        type_section = ctk.CTkFrame(config_frame)
        type_section.pack(fill="x", padx=20, pady=20)
        
        ctk.CTkLabel(
            type_section,
            text="Tipo de Teste",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", pady=(0, 10))
        
        self.test_type_var = ctk.StringVar(value="mean_difference")
        
        test_types = [
            ("mean_difference", "Test t - Diferença de Média (2+ grupos)"),
            ("one_way_anova", "One-Way ANOVA (2+ grupos)"),
            ("t_test_expected", "t-Test - Valor Esperado (1 amostra vs. média)"),
            ("t_test_sample", "t-Test - Amostras Pareadas (2+ amostras)")
        ]
        
        for value, text in test_types:
            ctk.CTkRadioButton(
                type_section,
                text=text,
                variable=self.test_type_var,
                value=value,
                command=self.on_test_type_change
            ).pack(anchor="w", padx=10, pady=5)
        
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
            text="Selecione 1 coluna X (grupos) e 1 ou mais colunas Y (respostas):",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.info_label.pack(anchor="w", pady=(0, 10))
        
        # X Column (Groups) - Single selection
        x_frame = ctk.CTkFrame(col_section, fg_color="transparent")
        x_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(
            x_frame,
            text="Coluna X (grupos) - Opcional:",
            width=250,
            font=ctk.CTkFont(weight="bold")
        ).pack(side="left", padx=(0, 10))
        
        columns_with_none = ["Nenhuma"] + list(self.df.columns)
        self.x_column_var = ctk.StringVar(value="Nenhuma")
        self.x_column_dropdown = ctk.CTkComboBox(
            x_frame,
            values=columns_with_none,
            variable=self.x_column_var,
            width=300
        )
        self.x_column_dropdown.pack(side="left")
        
        # Y Columns (Responses) - Multiple selection
        y_frame = ctk.CTkFrame(col_section, fg_color="transparent")
        y_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(
            y_frame,
            text="Respostas Y (obrigatório):",
            width=250,
            font=ctk.CTkFont(weight="bold")
        ).pack(side="left", padx=(0, 10))
        
        # Scrollable frame for Y checkboxes
        self.y_scroll_frame = ctk.CTkScrollableFrame(y_frame, height=120)
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
        
        # Expected Mean Entry (only for t-test expected) - create but don't pack yet
        self.expected_mean_frame = ctk.CTkFrame(col_section, fg_color="transparent")
        
        ctk.CTkLabel(
            self.expected_mean_frame,
            text="Média Esperada:",
            width=250,
            font=ctk.CTkFont(weight="bold")
        ).pack(side="left", padx=(0, 10))
        
        self.expected_mean_entry = ctk.CTkEntry(
            self.expected_mean_frame,
            width=300,
            placeholder_text="Ex: 100.0"
        )
        self.expected_mean_entry.pack(side="left")
        
        # Store reference to col_section for later repacking
        self.col_section = col_section
        
        # Action Buttons
        self.button_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        self.button_frame.pack(fill="x", padx=20, pady=20)
        
        self.calculate_btn = ctk.CTkButton(
            self.button_frame,
            text="🔍 Calcular Teste",
            command=self.calculate_test,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#2E86DE"
        )
        self.calculate_btn.pack(side="left", padx=10)
        
        self.clear_btn = ctk.CTkButton(
            self.button_frame,
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
    
    def on_test_type_change(self):
        """Update UI when test type changes"""
        test_type = self.test_type_var.get()
        
        # Always unpack first
        self.expected_mean_frame.pack_forget()
        
        if test_type == "mean_difference":
            self.info_label.configure(
                text="Test t - Diferença de Média: Selecione 1 coluna X (grupos) e respostas Y"
            )
            self.x_column_dropdown.configure(state="normal")
        elif test_type == "one_way_anova":
            self.info_label.configure(
                text="One-Way ANOVA: Selecione 1 coluna X (grupos) e respostas Y"
            )
            self.x_column_dropdown.configure(state="normal")
        elif test_type == "t_test_expected":
            self.info_label.configure(
                text="t-Test: Selecione 1 resposta Y e informe a média esperada"
            )
            self.x_column_dropdown.configure(state="disabled")
            self.x_column_var.set("Nenhuma")
            # Pack the expected mean frame
            self.expected_mean_frame.pack(fill="x", pady=5, in_=self.col_section)
        elif test_type == "t_test_sample":
            self.info_label.configure(
                text="Test t - Amostra Pareada: Selecione 2+ respostas Y (comparação pareada)"
            )
            self.x_column_dropdown.configure(state="disabled")
            self.x_column_var.set("Nenhuma")
    
    def validate_inputs(self) -> bool:
        """Validate user inputs"""
        test_type = self.test_type_var.get()
        x_col = self.x_column_var.get()
        y_cols = [col for col, var in self.y_vars if var.get() == "on"]
        
        # Check Y columns
        if len(y_cols) < 1:
            messagebox.showerror("Erro", "Selecione pelo menos uma resposta Y!")
            return False
        
        # Check test-specific requirements
        if test_type in ["mean_difference", "one_way_anova"]:
            if x_col == "Nenhuma":
                messagebox.showerror("Erro", "Selecione uma coluna X (grupos) para este teste!")
                return False
            
            # Check that X is not in Y
            if x_col in y_cols:
                messagebox.showerror("Erro", "A coluna X não pode estar nas respostas Y!")
                return False
        
        elif test_type == "t_test_expected":
            if len(y_cols) > 1:
                messagebox.showwarning(
                    "Atenção",
                    "t-Test com valor esperado usa apenas a primeira resposta selecionada."
                )
            
            # Check expected mean
            try:
                expected_mean = float(self.expected_mean_entry.get())
            except ValueError:
                messagebox.showerror("Erro", "Informe um valor numérico válido para a média esperada!")
                return False
        
        elif test_type == "t_test_sample":
            if len(y_cols) < 2:
                messagebox.showerror("Erro", "Test t - Amostra Pareada requer pelo menos 2 respostas Y!")
                return False
        
        return True
    
    def calculate_test(self):
        """Main calculation function"""
        if not self.validate_inputs():
            return
        
        self.calculate_btn.configure(state="disabled", text="⏳ Calculando...")
        self.update()
        
        try:
            test_type = self.test_type_var.get()
            x_col = self.x_column_var.get()
            y_cols = [col for col, var in self.y_vars if var.get() == "on"]
            
            # Prepare dataframe
            if test_type in ["mean_difference", "one_way_anova"]:
                if x_col != "Nenhuma":
                    work_df = self.df[[x_col] + y_cols].copy()
                else:
                    work_df = self.df[y_cols].copy()
            else:  # t_test_expected and t_test_sample
                work_df = self.df[y_cols].copy()
            
            work_df = work_df.dropna()
            
            if len(work_df) < 2:
                messagebox.showerror("Erro", "Dados insuficientes para análise!")
                return
            
            # Calculate based on test type
            if test_type == "mean_difference":
                # For mean difference, reorganize DataFrame with X column first
                if x_col != "Nenhuma" and x_col in work_df.columns:
                    # Put X column first, then response columns
                    cols = [x_col] + [col for col in work_df.columns if col != x_col]
                    work_df = work_df[cols]
                
                self.results = calculate_mean_difference(work_df, y_cols)
                self.results["test_type"] = "mean_difference"
                
            elif test_type == "one_way_anova":
                # For ANOVA, reorganize DataFrame with X column first
                if x_col != "Nenhuma" and x_col in work_df.columns:
                    # Put X column first, then response columns
                    cols = [x_col] + [col for col in work_df.columns if col != x_col]
                    work_df = work_df[cols]
                
                self.results = calculate_one_way_anova(work_df, y_cols)
                self.results["test_type"] = "one_way_anova"
                
            elif test_type == "t_test_expected":
                expected_mean = float(self.expected_mean_entry.get())
                result = calculate_t_test_expected_mean(work_df, y_cols[0], expected_mean)
                self.results = {"result": result, "response": y_cols[0], "test_type": "t_test_expected"}
                
            elif test_type == "t_test_sample":
                # Para paired-sample, não precisa de expected_mean
                self.results = calculate_t_test_sample(work_df, y_cols, 0)
                self.results["test_type"] = "t_test_sample"
            
            # Display results
            self.display_results()
            
        except Exception as e:
            messagebox.showerror("Erro no Cálculo", f"Erro ao calcular teste:\n{str(e)}")
            print(f"Hypothesis test error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.calculate_btn.configure(state="normal", text="🔍 Calcular Teste")
    
    def display_results(self):
        """Display calculation results"""
        # Clear previous results
        for widget in self.results_container.winfo_children():
            widget.destroy()
        
        if not self.results:
            return
        
        # Results title
        title = ctk.CTkLabel(
            self.results_container,
            text="📈 Resultados do Teste de Hipótese",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title.pack(pady=(10, 20))
        
        test_type = self.results.get("test_type")
        
        if test_type == "mean_difference":
            display_mean_difference_results(self.results_container, self.results)
        elif test_type == "one_way_anova":
            display_anova_results(self.results_container, self.results)
        elif test_type == "t_test_expected":
            display_t_test_expected_results(self.results_container, self.results)
        elif test_type == "t_test_sample":
            display_t_test_sample_results(self.results_container, self.results)
    
    def clear_results(self):
        """Clear all results"""
        for widget in self.results_container.winfo_children():
            widget.destroy()
        
        self.results = None
