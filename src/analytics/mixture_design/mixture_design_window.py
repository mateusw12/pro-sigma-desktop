"""
Mixture Design Window
Interface para Design de Mistura
"""
import customtkinter as ctk
from tkinter import messagebox
import pandas as pd
from src.utils.lazy_imports import (
    get_numpy, get_pandas, get_matplotlib, 
    get_matplotlib_figure, get_matplotlib_backend
)
from src.utils.ui_components import create_action_button, create_horizontal_stats_table
from typing import Dict

from src.analytics.mixture_design.mixture_design_utils import (
    generate_mixture_design,
    calculate_mixture_model,
    interpret_mixture_results
)

# Lazy-loaded libraries
_pd = None
_np = None
_plt = None
_Figure = None
_FigureCanvasTkAgg = None

def _ensure_libs():
    """Carrega bibliotecas pesadas apenas quando necessário"""
    global _pd, _np, _plt, _Figure, _FigureCanvasTkAgg
    if _pd is None:
        _pd = get_pandas()
        _np = get_numpy()
        _plt = get_matplotlib()
        _Figure = get_matplotlib_figure()
        _FigureCanvasTkAgg = get_matplotlib_backend()
    return _pd, _np, _plt, _Figure, _FigureCanvasTkAgg


class MixtureDesignWindow(ctk.CTkToplevel):
    def __init__(self, parent, df):
        super().__init__(parent)

        # Carrega bibliotecas
        pd, np, plt, Figure, FigureCanvasTkAgg = _ensure_libs()
        self.pd = pd
        self.np = np
        self.plt = plt
        self.Figure = Figure
        self.FigureCanvasTkAgg = FigureCanvasTkAgg
        
        self.df = df
        self.results = {}
        self.current_y_col = None
        
        # Window configuration
        self.title("Design de Mistura (Mixture Design)")
        self.resizable(True, True)
        self.minsize(1400, 900)
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
            text="🧪 Design de Mistura (Mixture Design)",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=(0, 20))
        
        # Configuration Frame
        config_frame = ctk.CTkFrame(self.main_container)
        config_frame.pack(fill="x", pady=(0, 20))
        
        # Response variables (Y)
        y_section = ctk.CTkFrame(config_frame)
        y_section.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            y_section,
            text="📊 Variáveis Resposta (Y)",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", pady=(0, 5))
        
        ctk.CTkLabel(
            y_section,
            text="Selecione uma ou mais colunas Y para análise separada",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        ).pack(anchor="w", padx=10)
        
        # Scrollable frame for Y checkboxes
        self.y_scroll_frame = ctk.CTkScrollableFrame(y_section, height=100)
        self.y_scroll_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
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
        
        # Mixture components (X)
        x_section = ctk.CTkFrame(config_frame)
        x_section.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            x_section,
            text="⚗️ Componentes da Mistura (X)",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", pady=(0, 5))
        
        ctk.CTkLabel(
            x_section,
            text="Componentes devem somar 1 (ou 100%)",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        ).pack(anchor="w", padx=10)
        
        # Scrollable frame for X checkboxes
        self.x_scroll_frame = ctk.CTkScrollableFrame(x_section, height=150)
        self.x_scroll_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
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
        
        # Interactions section
        interaction_section = ctk.CTkFrame(config_frame)
        interaction_section.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            interaction_section,
            text="🔗 Termos de Interação",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", pady=(0, 5))
        
        # Include all interactions checkbox
        self.include_interactions_var = ctk.StringVar(value="off")
        ctk.CTkCheckBox(
            interaction_section,
            text="Incluir todas as interações de 2ª ordem (X1*X2, X1*X3, ...)",
            variable=self.include_interactions_var,
            onvalue="on",
            offvalue="off"
        ).pack(anchor="w", padx=10, pady=5)
        
        # Custom interactions
        custom_frame = ctk.CTkFrame(interaction_section, fg_color="transparent")
        custom_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(
            custom_frame,
            text="Ou especifique interações customizadas (ex: X1*X2, X2*X3):",
            font=ctk.CTkFont(size=11)
        ).pack(anchor="w")
        
        self.custom_interactions_entry = ctk.CTkEntry(
            custom_frame,
            placeholder_text="X1*X2, X1*X3",
            width=400
        )
        self.custom_interactions_entry.pack(anchor="w", pady=5)
        
        # Action Buttons
        button_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        button_frame.pack(fill="x", padx=20, pady=20)
        
        self.calculate_btn = create_action_button(
            button_frame,
            text="Calcular Modelo de Mistura",
            command=self.calculate_model,
            icon="🔬"
        )
        self.calculate_btn.pack(side="left", padx=10)
        
        self.generate_exp_btn = ctk.CTkButton(
            button_frame,
            text="🧪 Gerar Experimento",
            command=self.generate_experiment,
            height=40,
            font=ctk.CTkFont(size=14),
            fg_color="#27AE60"
        )
        self.generate_exp_btn.pack(side="left", padx=10)
        
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
        self.results_container = ctk.CTkScrollableFrame(self.main_container, height=600)
        self.results_container.pack(fill="both", expand=True, pady=(0, 20))
    
    def validate_inputs(self) -> bool:
        """Valida entradas"""
        # Variáveis Y
        y_cols = [col for col, var in self.y_vars if var.get() == "on"]
        print(f"DEBUG: Y columns selected: {y_cols}")
        print(f"DEBUG: Y vars check: {[(col, var.get()) for col, var in self.y_vars]}")
        
        if len(y_cols) < 1:
            messagebox.showerror("Erro", "Selecione pelo menos uma variável resposta Y!")
            return False
        
        # Componentes X
        x_cols = [col for col, var in self.x_vars if var.get() == "on"]
        print(f"DEBUG: X columns selected: {x_cols}")
        
        if len(x_cols) < 3:
            messagebox.showerror("Erro", "Selecione pelo menos 3 componentes da mistura!")
            return False
        
        # Verificar se Y e X não se sobrepõem
        overlap = set(y_cols) & set(x_cols)
        if overlap:
            messagebox.showerror("Erro", f"Colunas não podem ser Y e X ao mesmo tempo: {overlap}")
            return False
        
        return True
    
    def calculate_model(self):
        """Calcula modelo de mistura para cada Y"""
        if not self.validate_inputs():
            return
        
        self.calculate_btn.configure(state="disabled", text="⏳ Calculando...")
        self.update()
        
        try:
            # Get selected columns
            y_cols = [col for col, var in self.y_vars if var.get() == "on"]
            x_cols = [col for col, var in self.x_vars if var.get() == "on"]
            
            print(f"DEBUG: Starting calculation with Y={y_cols}, X={x_cols}")
            
            # Get interaction settings
            include_all_interactions = self.include_interactions_var.get() == "on"
            custom_interactions_text = self.custom_interactions_entry.get().strip()
            
            custom_terms = None
            if custom_interactions_text:
                custom_terms = [t.strip() for t in custom_interactions_text.split(',')]
            
            print(f"DEBUG: Include all interactions: {include_all_interactions}")
            print(f"DEBUG: Custom terms: {custom_terms}")
            
            # Calculate for each Y
            self.results = {}
            failed_y = []
            
            for y_col in y_cols:
                try:
                    # Prepare data
                    work_df = self.df[[y_col] + x_cols].copy()
                    work_df = work_df.dropna()
                    
                    print(f"Processing {y_col}: {len(work_df)} rows after dropna")
                    
                    if len(work_df) < 10:
                        failed_y.append(f"{y_col} (dados insuficientes: {len(work_df)} linhas)")
                        continue
                    
                    # Calculate model
                    results = calculate_mixture_model(
                        work_df,
                        y_col,
                        x_cols,
                        include_interactions=include_all_interactions,
                        custom_terms=custom_terms
                    )
                    
                    # Add interpretations
                    results['interpretations'] = interpret_mixture_results(results)
                    
                    self.results[y_col] = results
                    print(f"Successfully calculated model for {y_col}")
                    
                except Exception as e:
                    failed_y.append(f"{y_col} (erro: {str(e)})")
                    print(f"Error calculating {y_col}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Check if any models succeeded
            print(f"DEBUG: Results calculated: {len(self.results)} models")
            print(f"DEBUG: Results keys: {list(self.results.keys())}")
            
            if len(self.results) == 0:
                error_msg = f"Nenhum modelo foi calculado!\n\nVariáveis Y selecionadas: {len(y_cols)}\n"
                if failed_y:
                    error_msg += f"\nProblemas encontrados:\n" + "\n".join(f"• {f}" for f in failed_y)
                messagebox.showerror("Erro", error_msg)
                return
            
            # Display results
            print("DEBUG: Calling display_results()")
            self.display_results()
            print("DEBUG: display_results() completed")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao calcular modelo:\n{str(e)}")
            print(f"Mixture design error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.calculate_btn.configure(state="normal", text="🔬 Calcular Modelo de Mistura")
    
    def generate_experiment(self):
        """Gera design de experimento de mistura"""
        # Create popup window
        exp_window = ctk.CTkToplevel(self)
        exp_window.title("Gerar Experimento de Mistura")
        exp_window.geometry("700x700")
        exp_window.transient(self)
        exp_window.grab_set()
        
        # Main scrollable frame
        main_frame = ctk.CTkScrollableFrame(exp_window)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        ctk.CTkLabel(
            main_frame,
            text="🧪 Configuração do Experimento",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(0, 20))
        
        # Number of factors
        factors_frame = ctk.CTkFrame(main_frame)
        factors_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            factors_frame,
            text="Número de Fatores (Componentes):",
            font=ctk.CTkFont(size=13)
        ).pack(side="left", padx=10)
        
        n_factors_var = ctk.StringVar(value="3")
        n_factors_entry = ctk.CTkEntry(factors_frame, textvariable=n_factors_var, width=100)
        n_factors_entry.pack(side="left", padx=10)
        
        # Button to update constraints
        update_btn = ctk.CTkButton(
            factors_frame,
            text="↻ Atualizar Restrições",
            command=lambda: update_constraints_ui(),
            width=150,
            height=30,
            font=ctk.CTkFont(size=12)
        )
        update_btn.pack(side="left", padx=10)
        
        # Constraints section
        constraints_section = ctk.CTkFrame(main_frame)
        constraints_section.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            constraints_section,
            text="⚙️ Restrições dos Componentes (Opcional)",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        ctk.CTkLabel(
            constraints_section,
            text="Defina limites mín/máx para cada componente (deixe vazio para sem restrição)",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        ).pack(pady=(0, 10))
        
        # Container for constraint entries
        constraints_container = ctk.CTkFrame(constraints_section)
        constraints_container.pack(fill="x", padx=10, pady=5)
        
        constraint_entries = []
        
        def update_constraints_ui():
            """Atualiza UI de restrições baseado no número de fatores"""
            # Clear existing
            for widget in constraints_container.winfo_children():
                widget.destroy()
            constraint_entries.clear()
            
            try:
                n_factors = int(n_factors_var.get())
                if n_factors < 3:
                    n_factors = 3
                
                # Header
                header_frame = ctk.CTkFrame(constraints_container, fg_color="transparent")
                header_frame.pack(fill="x", pady=5)
                
                ctk.CTkLabel(header_frame, text="Componente", width=100, font=ctk.CTkFont(size=11, weight="bold")).pack(side="left", padx=5)
                ctk.CTkLabel(header_frame, text="Mínimo", width=80, font=ctk.CTkFont(size=11, weight="bold")).pack(side="left", padx=5)
                ctk.CTkLabel(header_frame, text="Máximo", width=80, font=ctk.CTkFont(size=11, weight="bold")).pack(side="left", padx=5)
                
                # Create entry for each factor
                for i in range(n_factors):
                    entry_frame = ctk.CTkFrame(constraints_container, fg_color="transparent")
                    entry_frame.pack(fill="x", pady=3)
                    
                    ctk.CTkLabel(
                        entry_frame,
                        text=f"X{i+1}",
                        width=100,
                        font=ctk.CTkFont(size=12)
                    ).pack(side="left", padx=5)
                    
                    min_var = ctk.StringVar(value="")
                    min_entry = ctk.CTkEntry(entry_frame, textvariable=min_var, width=80, placeholder_text="0.0")
                    min_entry.pack(side="left", padx=5)
                    
                    max_var = ctk.StringVar(value="")
                    max_entry = ctk.CTkEntry(entry_frame, textvariable=max_var, width=80, placeholder_text="1.0")
                    max_entry.pack(side="left", padx=5)
                    
                    constraint_entries.append({
                        'component': f'X{i+1}',
                        'min_var': min_var,
                        'max_var': max_var
                    })
                    
            except ValueError:
                pass
        
        # Initialize with 3 factors
        update_constraints_ui()
        
        # Number of runs
        runs_frame = ctk.CTkFrame(main_frame)
        runs_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            runs_frame,
            text="Número de Experimentos:",
            font=ctk.CTkFont(size=13)
        ).pack(side="left", padx=10)
        
        n_runs_var = ctk.StringVar(value="10")
        n_runs_entry = ctk.CTkEntry(runs_frame, textvariable=n_runs_var, width=100)
        n_runs_entry.pack(side="left", padx=10)
        
        # Design type
        design_frame = ctk.CTkFrame(main_frame)
        design_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            design_frame,
            text="Tipo de Design:",
            font=ctk.CTkFont(size=13)
        ).pack(side="left", padx=10)
        
        design_type_var = ctk.StringVar(value="space_filling")
        design_menu = ctk.CTkOptionMenu(
            design_frame,
            variable=design_type_var,
            values=["space_filling", "optimal"],
            width=150
        )
        design_menu.pack(side="left", padx=10)
        
        # Info labels
        info_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        info_frame.pack(fill="x", pady=10, padx=10)
        
        ctk.CTkLabel(
            info_frame,
            text="💡 Space Filling: Latin Hypercube (distribuição uniforme)",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        ).pack(anchor="w", pady=2)
        
        ctk.CTkLabel(
            info_frame,
            text="💡 Optimal: D-optimal (maximiza informação)",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        ).pack(anchor="w", pady=2)
        
        # Result text
        result_text = ctk.CTkTextbox(main_frame, height=150)
        result_text.pack(fill="both", expand=True, pady=10)
        
        def perform_generation():
            try:
                n_factors = int(n_factors_var.get())
                n_runs = int(n_runs_var.get())
                design_type = design_type_var.get()
                
                if n_factors < 3:
                    messagebox.showerror("Erro", "Mínimo de 3 fatores!")
                    return
                
                if n_runs < 5:
                    messagebox.showerror("Erro", "Mínimo de 5 experimentos!")
                    return
                
                # Parse constraints
                constraints_df = None
                constraints_list = []
                
                for entry in constraint_entries:
                    min_val = entry['min_var'].get().strip()
                    max_val = entry['max_var'].get().strip()
                    
                    if min_val or max_val:
                        try:
                            min_level = float(min_val) if min_val else 0.0
                            max_level = float(max_val) if max_val else 1.0
                            
                            if min_level < 0 or max_level > 1 or min_level >= max_level:
                                messagebox.showerror("Erro", f"Restrições inválidas para {entry['component']}: min={min_level}, max={max_level}")
                                return
                            
                            constraints_list.append({
                                'component': entry['component'],
                                'minLevelValue': min_level,
                                'maxLevelValue': max_level
                            })
                        except ValueError:
                            messagebox.showerror("Erro", f"Valor inválido nas restrições de {entry['component']}")
                            return
                
                if constraints_list:
                    constraints_df = self.pd.DataFrame(constraints_list)
                    print(f"DEBUG: Constraints applied:\n{constraints_df}")
                
                # Generate design
                design_df = generate_mixture_design(
                    n_factors=n_factors,
                    n_runs=n_runs,
                    design_type=design_type,
                    constraints=constraints_df
                )
                
                # Display result
                result_text.delete("1.0", "end")
                result_text.insert("1.0", f"Experimento Gerado: {n_runs} runs x {n_factors} fatores\n")
                if constraints_df is not None:
                    result_text.insert("end", f"Com restrições aplicadas\n")
                result_text.insert("end", "\n")
                result_text.insert("end", design_df.to_string(index=True))
                
                # Ask to export
                if messagebox.askyesno("Exportar", "Deseja exportar o experimento para Excel?"):
                    from tkinter import filedialog
                    filename = filedialog.asksaveasfilename(
                        defaultextension=".xlsx",
                        filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
                    )
                    if filename:
                        design_df.to_excel(filename, index=False)
                        messagebox.showinfo("Sucesso", f"Experimento exportado para:\n{filename}")
                        messagebox.showinfo("Sucesso", f"Experimento exportado para:\\n{filename}")
                
            except ValueError as e:
                messagebox.showerror("Erro", f"Valores inválidos:\\n{str(e)}")
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao gerar experimento:\\n{str(e)}")
                import traceback
                traceback.print_exc()
        
        # Button frame
        btn_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        btn_frame.pack(fill="x", pady=10)
        
        generate_btn = ctk.CTkButton(
            btn_frame,
            text="✨ Gerar Experimento",
            command=perform_generation,
            height=40,
            font=ctk.CTkFont(size=14),
            fg_color="#27AE60"
        )
        generate_btn.pack(side="left", padx=10)
        
        close_btn = ctk.CTkButton(
            btn_frame,
            text="Fechar",
            command=exp_window.destroy,
            height=40,
            font=ctk.CTkFont(size=14),
            fg_color="#95A5A6"
        )
        close_btn.pack(side="left", padx=10)
    
    def clear_results(self):
        """Limpa resultados"""
        print(f"DEBUG: clear_results() called - clearing {len(self.results_container.winfo_children())} widgets")
        for widget in self.results_container.winfo_children():
            widget.destroy()
        # Não limpar self.results aqui! Apenas os widgets visuais
    
    def display_results(self):
        """Exibe resultados"""
        print(f"DEBUG: display_results() called with {len(self.results)} results")
        self.clear_results()
        
        if not self.results:
            print("DEBUG: No results to display (self.results is empty)")
            return
        
        print(f"DEBUG: Displaying results for: {list(self.results.keys())}")
        
        # Title
        title = ctk.CTkLabel(
            self.results_container,
            text="📊 Resultados do Modelo de Mistura",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title.pack(pady=(10, 20))
        
        # Display for each Y
        for y_col, result in self.results.items():
            print(f"DEBUG: Displaying results for Y={y_col}")
            self.display_y_results(y_col, result)
    
    def display_y_results(self, y_col: str, result: Dict):
        """Exibe resultados para uma variável Y"""
        # Header para esta Y
        y_header = ctk.CTkFrame(self.results_container)
        y_header.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            y_header,
            text=f"📈 Análise para: {y_col}",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#2563eb"
        ).pack(pady=10)
        
        # Equation
        self.display_equation(y_col, result)
        
        # Summary of Fit
        self.display_summary_of_fit(y_col, result)
        
        # ANOVA
        self.display_anova(y_col, result)
        
        # Parameter Estimates
        self.display_parameter_estimates(y_col, result)
        
        # Plots
        self.display_plots(y_col, result)
        
        # Interpretations
        self.display_interpretations(y_col, result)
        
        # Separator
        ctk.CTkFrame(self.results_container, height=2, fg_color="#e0e0e0").pack(fill="x", pady=20)
    
    def display_equation(self, y_col: str, result: Dict):
        """Exibe equação do modelo"""
        eq_frame = ctk.CTkFrame(self.results_container)
        eq_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            eq_frame,
            text="📝 Equação do Modelo",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        eq_text = ctk.CTkTextbox(eq_frame, height=80, wrap="word")
        eq_text.pack(fill="x", padx=10, pady=10)
        eq_text.insert("1.0", result['equation'])
        eq_text.configure(state="disabled")
    
    def display_summary_of_fit(self, y_col: str, result: Dict):
        """Exibe resumo do ajuste"""
        summary_frame = ctk.CTkFrame(self.results_container)
        summary_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            summary_frame,
            text="📊 Resumo do Ajuste",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        summary = result['summaryOfFit']
        rows_data = [
            {"Métrica": "R²", "Valor": f"{summary['r2']:.4f}"},
            {"Métrica": "R² Ajustado", "Valor": f"{summary['r2_adj']:.4f}"},
            {"Métrica": "RMSE", "Valor": f"{summary['rmse']:.4f}"},
            {"Métrica": "Média de Y", "Valor": f"{summary['mean']:.4f}"},
            {"Métrica": "Observações", "Valor": f"{summary['observations']}"}
        ]
        
        create_horizontal_stats_table(
            summary_frame,
            columns=["Métrica", "Valor"],
            rows_data=rows_data,
            title="Estatísticas do Modelo"
        )
    
    def display_anova(self, y_col: str, result: Dict):
        """Exibe tabela ANOVA"""
        anova_frame = ctk.CTkFrame(self.results_container)
        anova_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            anova_frame,
            text="📋 ANOVA - Análise de Variância",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        rows_data = []
        for row in result['anovaTable']:
            rows_data.append({
                "Fonte": row['source'],
                "GL": f"{row['df']}" if row['df'] is not None else "",
                "SQ": f"{row['sumSquares']:.4f}" if row['sumSquares'] is not None else "",
                "QM": f"{row['meanSquares']:.4f}" if row['meanSquares'] is not None else "",
                "F": f"{row['fValue']:.4f}" if row['fValue'] is not None else "",
                "P-value": f"{row['prob']:.4f}" if row['prob'] is not None else ""
            })
        
        create_horizontal_stats_table(
            anova_frame,
            columns=["Fonte", "GL", "SQ", "QM", "F", "P-value"],
            rows_data=rows_data,
            title="Tabela ANOVA"
        )
    
    def display_parameter_estimates(self, y_col: str, result: Dict):
        """Exibe estimativas dos parâmetros"""
        param_frame = ctk.CTkFrame(self.results_container)
        param_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            param_frame,
            text="📐 Estimativas dos Parâmetros",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        rows_data = []
        for param in result['parameterEstimates']:
            rows_data.append({
                "Termo": param['term'],
                "Estimativa": f"{param['estimate']:.6f}",
                "Erro Padrão": f"{param['stdError']:.6f}",
                "t-value": f"{param['tValue']:.4f}",
                "P-value": f"{param['prob']:.4f}"
            })
        
        create_horizontal_stats_table(
            param_frame,
            columns=["Termo", "Estimativa", "Erro Padrão", "t-value", "P-value"],
            rows_data=rows_data,
            title="Coeficientes do Modelo"
        )
    
    def display_plots(self, y_col: str, result: Dict):
        """Exibe gráficos"""
        plot_frame = ctk.CTkFrame(self.results_container)
        plot_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            plot_frame,
            text="📈 Gráficos de Predição",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        # Create figure with 3 subplots
        fig = self.Figure(figsize=(15, 5))
        
        # 1. Real vs Predicted - Line plot ordered
        ax1 = fig.add_subplot(131)
        y_actual = self.np.array(result['yActual'])
        y_pred = self.np.array(result['predictions'])
        
        # Sort by actual values (descending)
        sorted_indices = self.np.argsort(y_actual)[::-1]
        y_actual_sorted = y_actual[sorted_indices]
        y_pred_sorted = y_pred[sorted_indices]
        x_positions = self.np.arange(len(y_actual_sorted))
        
        ax1.plot(x_positions, y_actual_sorted, 'o-', label='Real', linewidth=2, markersize=5)
        ax1.plot(x_positions, y_pred_sorted, 's-', label='Predito', linewidth=2, markersize=5)
        ax1.set_xlabel('Observações (ordenadas)')
        ax1.set_ylabel('Valores')
        ax1.set_title('Real vs Predito')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Residuals
        ax2 = fig.add_subplot(132)
        residuals = result['residuals']
        ax2.scatter(y_pred, residuals, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Valores Preditos')
        ax2.set_ylabel('Resíduos')
        ax2.set_title('Resíduos vs Preditos')
        ax2.grid(True, alpha=0.3)
        
        # 3. Parameter Estimates (absolute values) - Bar plot sorted
        ax3 = fig.add_subplot(133)
        params = result['parameterEstimates']
        
        # Create list of (term, abs_estimate) and sort by absolute value
        param_data = [(p['term'], abs(p['estimate'])) for p in params]
        param_data.sort(key=lambda x: x[1])  # Sort ascending (menor para maior)
        
        terms = [p[0] for p in param_data]
        estimates = [p[1] for p in param_data]
        
        bars = ax3.barh(terms, estimates, color='steelblue', alpha=0.7)
        ax3.set_xlabel('|Estimativa|')
        ax3.set_title('Estimativas (Valor Absoluto)')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, estimates)):
            ax3.text(val, i, f' {val:.4f}', va='center', fontsize=9)
        
        fig.tight_layout()
        
        canvas = self.FigureCanvasTkAgg(fig, plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    def display_interpretations(self, y_col: str, result: Dict):
        """Exibe interpretações"""
        interp_frame = ctk.CTkFrame(self.results_container)
        interp_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            interp_frame,
            text="💡 Interpretações",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        for interp in result['interpretations']:
            ctk.CTkLabel(
                interp_frame,
                text=f"• {interp}",
                font=ctk.CTkFont(size=12),
                wraplength=1200,
                justify="left"
            ).pack(anchor="w", padx=20, pady=3)
