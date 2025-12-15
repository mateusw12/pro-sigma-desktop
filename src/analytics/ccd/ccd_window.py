"""
Interface gráfica para Central Composite Design (CCD)
Permite gerar experimentos e analisar dados com ANOVA completo
"""

import customtkinter as ctk
from tkinter import messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk

from src.utils.lazy_imports import get_pandas, get_numpy
from src.utils.ui_components import add_chart_export_button
from src.analytics.ccd.ccd_utils import (
    generate_ccd_design,
    calculate_ccd_analysis,
    split_dataframes_by_response
)


class CCDWindow(ctk.CTkToplevel):
    """Janela de Central Composite Design"""
    
    def __init__(self, parent, data=None):
        super().__init__(parent)
        
        self.title("Central Composite Design (CCD)")
        self.geometry("1400x900")
        self.state('zoomed')
        
        self.data = data
        self.generated_design = None
        self.analysis_results = {}
        self.current_response = None
        self.quadratic_vars = {}  # Store quadratic term checkboxes
        self.interaction_vars = {}  # Store interaction term checkboxes
        
        self.configure(fg_color="#2b2b2b")
        
        # Determina se inicia em modo geração ou análise
        self.mode = "generate" if data is None else "analyze"
        
        self._create_widgets()
        
        if self.mode == "analyze":
            self._populate_analysis_columns()
    
    def _create_widgets(self):
        """Cria os widgets da interface"""
        
        # Container principal com abas
        self.tabview = ctk.CTkTabview(self, fg_color="#2b2b2b")
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Aba de Geração
        self.tab_generate = self.tabview.add("🧪 Gerar Experimento")
        self._create_generation_tab()
        
        # Aba de Análise
        self.tab_analyze = self.tabview.add("📊 Analisar Dados")
        self._create_analysis_tab()
        
        # Seleciona aba inicial baseada no modo
        if self.mode == "analyze":
            self.tabview.set("📊 Analisar Dados")
    
    def _create_generation_tab(self):
        """Cria aba de geração de experimentos"""
        
        # Container com scroll
        scroll_container = ctk.CTkScrollableFrame(
            self.tab_generate,
            scrollbar_button_color="gray30",
            scrollbar_button_hover_color="gray40"
        )
        scroll_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Título
        title = ctk.CTkLabel(
            scroll_container,
            text="🧪 Geração de Experimento Central Composite Design",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=(0, 10))
        
        # Descrição
        desc = ctk.CTkLabel(
            scroll_container,
            text="Configure e gere designs CCD (Rotatable/Orthogonal) ou Box-Behnken para planejamento de experimentos",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        desc.pack(pady=(0, 20))
        
        # Frame de configuração
        config_frame = ctk.CTkFrame(scroll_container)
        config_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(
            config_frame,
            text="⚙️ Configuração do Experimento",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(20, 15), padx=20, anchor="w")
        
        # Grid de configurações
        grid = ctk.CTkFrame(config_frame, fg_color="transparent")
        grid.pack(fill="x", padx=20, pady=(0, 10))
        
        # Número de fatores
        ctk.CTkLabel(
            grid,
            text="Número de Fatores:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=5, pady=10)
        self.n_factors_entry = ctk.CTkEntry(grid, width=150)
        self.n_factors_entry.insert(0, "3")
        self.n_factors_entry.grid(row=0, column=1, padx=5, pady=10, sticky="w")
        
        # Pontos centrais
        ctk.CTkLabel(
            grid,
            text="Pontos Centrais:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).grid(row=1, column=0, sticky="w", padx=5, pady=10)
        self.n_center_entry = ctk.CTkEntry(grid, width=150)
        self.n_center_entry.insert(0, "6")
        self.n_center_entry.grid(row=1, column=1, padx=5, pady=10, sticky="w")
        
        # Tipo de design
        ctk.CTkLabel(
            grid,
            text="Tipo de Design:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).grid(row=2, column=0, sticky="w", padx=5, pady=10)
        self.design_type_combo = ctk.CTkComboBox(
            grid,
            width=200,
            values=["rotatable", "orthogonal", "bbd"],
            state="readonly"
        )
        self.design_type_combo.set("rotatable")
        self.design_type_combo.grid(row=2, column=1, padx=5, pady=10, sticky="w")
        
        # Informação sobre tipos
        info_text = "• Rotatable: Alpha rotacional (k^0.25)\n• Orthogonal: Alpha ortogonal\n• BBD: Box-Behnken Design (mín. 3 fatores)"
        ctk.CTkLabel(
            grid,
            text=info_text,
            font=ctk.CTkFont(size=10),
            text_color="gray",
            justify="left"
        ).grid(row=3, column=1, padx=5, pady=5, sticky="w")
        
        # Número de respostas
        ctk.CTkLabel(
            grid,
            text="Número de Respostas (Y):",
            font=ctk.CTkFont(size=13, weight="bold")
        ).grid(row=0, column=2, sticky="w", padx=(30, 5), pady=10)
        self.n_responses_entry = ctk.CTkEntry(grid, width=150)
        self.n_responses_entry.insert(0, "1")
        self.n_responses_entry.grid(row=0, column=3, padx=5, pady=10, sticky="w")
        
        # Info sobre respostas
        ctk.CTkLabel(
            grid,
            text="Valor padrão: 99999 para cada resposta",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        ).grid(row=1, column=2, columnspan=2, padx=(30, 5), pady=5, sticky="w")
        
        # Botão gerar
        generate_btn = ctk.CTkButton(
            config_frame,
            text="🚀 Gerar Design",
            command=self._generate_design,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#1f6aa5",
            hover_color="#144870",
            height=45,
            width=250
        )
        generate_btn.pack(pady=20)
        
        # Frame para preview do design gerado
        preview_frame = ctk.CTkFrame(scroll_container)
        preview_frame.pack(fill="both", expand=True, pady=(0, 20))
        
        ctk.CTkLabel(
            preview_frame,
            text="📋 Preview do Experimento Gerado",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(20, 10), padx=20, anchor="w")
        
        self.design_preview_frame = ctk.CTkScrollableFrame(
            preview_frame,
            height=400
        )
        self.design_preview_frame.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        
        # Mensagem inicial
        ctk.CTkLabel(
            self.design_preview_frame,
            text="Clique em 'Gerar Design' para criar um novo experimento",
            text_color="gray",
            font=ctk.CTkFont(size=12)
        ).pack(pady=50)
        
        # Botões de ação
        action_frame = ctk.CTkFrame(preview_frame, fg_color="transparent")
        action_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        ctk.CTkButton(
            action_frame,
            text="💾 Exportar para CSV",
            command=self._export_design,
            height=40,
            width=200,
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            action_frame,
            text="📋 Copiar para Área de Transferência",
            command=self._copy_design,
            height=40,
            width=250,
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=5)
    
    def _create_analysis_tab(self):
        """Cria aba de análise de dados"""
        
        # Container com scroll
        scroll_container = ctk.CTkScrollableFrame(
            self.tab_analyze,
            scrollbar_button_color="gray30",
            scrollbar_button_hover_color="gray40"
        )
        scroll_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Título
        title = ctk.CTkLabel(
            scroll_container,
            text="📊 Análise Central Composite Design",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=(0, 5))
        
        # Descrição
        desc = ctk.CTkLabel(
            scroll_container,
            text="Análise completa com ANOVA, Parameter Estimates, Summary of Fit e Lack of Fit Test",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        desc.pack(pady=(0, 10))
        
        # Frame de seleção
        selection_frame = ctk.CTkFrame(scroll_container)
        selection_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(
            selection_frame,
            text="🔧 Seleção de Variáveis",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(15, 10), padx=20, anchor="w")
        
        # Container para X e Y lado a lado
        vars_container = ctk.CTkFrame(selection_frame, fg_color="transparent")
        vars_container.pack(fill="x", padx=20, pady=(0, 10))
        
        # Colunas X (esquerda)
        x_frame = ctk.CTkFrame(vars_container)
        x_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        ctk.CTkLabel(
            x_frame,
            text="Variáveis Independentes (X) - Fatores:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", pady=(5, 3), padx=10)
        
        self.x_columns_frame = ctk.CTkScrollableFrame(x_frame, height=100)
        self.x_columns_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Frame para termos de X (lado a lado) - dentro do x_frame
        terms_x_container = ctk.CTkFrame(x_frame, fg_color="transparent")
        terms_x_container.pack(fill="x", padx=10, pady=(0, 10))
        
        # === Termos Quadráticos (esquerda) ===
        quadratic_frame = ctk.CTkFrame(terms_x_container)
        quadratic_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        ctk.CTkLabel(
            quadratic_frame,
            text="Termos Quadráticos (X²):",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", pady=(5, 3), padx=5)
        
        self.quadratic_scroll = ctk.CTkScrollableFrame(quadratic_frame, height=80)
        self.quadratic_scroll.pack(fill="both", expand=True, padx=5, pady=(0, 5))
        
        ctk.CTkLabel(
            self.quadratic_scroll,
            text="Selecione X para habilitar",
            text_color="gray",
            font=ctk.CTkFont(size=10)
        ).pack(pady=10)
        
        # === Termos de Interação (direita) ===
        interaction_frame = ctk.CTkFrame(terms_x_container)
        interaction_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        ctk.CTkLabel(
            interaction_frame,
            text="Termos de Interação (X₁ × X₂):",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", pady=(5, 3), padx=5)
        
        self.interaction_scroll = ctk.CTkScrollableFrame(interaction_frame, height=80)
        self.interaction_scroll.pack(fill="both", expand=True, padx=5, pady=(0, 5))
        
        ctk.CTkLabel(
            self.interaction_scroll,
            text="Selecione 2+ X para habilitar",
            text_color="gray",
            font=ctk.CTkFont(size=10)
        ).pack(pady=10)
        
        # Colunas Y (direita)
        y_frame = ctk.CTkFrame(vars_container)
        y_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        ctk.CTkLabel(
            y_frame,
            text="Variáveis Dependentes (Y) - Respostas:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", pady=(5, 3), padx=10)
        
        self.y_columns_frame = ctk.CTkScrollableFrame(y_frame, height=120)
        self.y_columns_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Botão calcular
        calculate_btn = ctk.CTkButton(
            selection_frame,
            text="🔍 Calcular Análise",
            command=self._calculate_analysis,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#1f6aa5",
            hover_color="#144870",
            height=40,
            width=200
        )
        calculate_btn.pack(pady=10)
        
        # Frame de resultados
        self.results_frame = ctk.CTkFrame(scroll_container, fg_color="transparent")
        self.results_frame.pack(fill="both", expand=True, pady=(0, 10))
    
    def _generate_design(self):
        """Gera o design CCD"""
        try:
            n_factors = int(self.n_factors_entry.get())
            n_center = int(self.n_center_entry.get())
            design_type = self.design_type_combo.get()
            n_responses = int(self.n_responses_entry.get())
            
            if n_factors < 2:
                messagebox.showerror("Erro", "Número de fatores deve ser no mínimo 2")
                return
            
            if design_type == "bbd" and n_factors < 3:
                messagebox.showerror("Erro", "Box-Behnken requer no mínimo 3 fatores")
                return
            
            # Gera design
            self.generated_design = generate_ccd_design(
                n_factors, n_center, design_type, n_responses
            )
            
            # Exibe preview
            self._display_design_preview()
            
            messagebox.showinfo(
                "Sucesso",
                f"Design gerado com sucesso!\n\n"
                f"Total de experimentos: {len(self.generated_design)}"
            )
            
        except ValueError as e:
            messagebox.showerror("Erro", f"Valores inválidos: {str(e)}")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao gerar design: {str(e)}")
    
    def _display_design_preview(self):
        """Exibe preview do design gerado"""
        
        # Limpa preview anterior
        for widget in self.design_preview_frame.winfo_children():
            widget.destroy()
        
        if self.generated_design is None:
            return
        
        pd = get_pandas()
        
        # Cria tabela
        df = self.generated_design
        
        # Info sobre o design
        info_frame = ctk.CTkFrame(self.design_preview_frame, fg_color="#1f6aa5", corner_radius=5)
        info_frame.pack(fill="x", pady=(0, 10))
        
        info_text = f"✓ Design gerado com sucesso! Total de experimentos: {len(df)}  |  Fatores: {len([c for c in df.columns if c.startswith('x')])}  |  Respostas: {len([c for c in df.columns if c.startswith('Y')])}"
        ctk.CTkLabel(
            info_frame,
            text=info_text,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="white"
        ).pack(pady=10)
        
        # Header
        header_frame = ctk.CTkFrame(self.design_preview_frame, fg_color="#144870")
        header_frame.pack(fill="x", pady=(0, 1))
        
        for i, col in enumerate(df.columns):
            label = ctk.CTkLabel(
                header_frame,
                text=col,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color="white",
                width=80
            )
            label.grid(row=0, column=i, padx=2, pady=8, sticky="ew")
        
        # Linhas (limita a 100 para performance)
        max_rows = min(100, len(df))
        for idx in range(max_rows):
            row_frame = ctk.CTkFrame(
                self.design_preview_frame,
                fg_color="#2b2b2b" if idx % 2 == 0 else "#1e1e1e"
            )
            row_frame.pack(fill="x", pady=1)
            
            for i, col in enumerate(df.columns):
                value = df.iloc[idx][col]
                if isinstance(value, (int, float)):
                    text = f"{value:.4f}" if abs(value) < 1000 else f"{value:.2f}"
                else:
                    text = str(value)
                
                label = ctk.CTkLabel(
                    row_frame,
                    text=text,
                    font=ctk.CTkFont(size=10),
                    width=80
                )
                label.grid(row=0, column=i, padx=2, pady=5, sticky="ew")
        
        if len(df) > max_rows:
            info = ctk.CTkLabel(
                self.design_preview_frame,
                text=f"... mostrando {max_rows} de {len(df)} linhas (use Exportar para ver todos)",
                font=ctk.CTkFont(size=10),
                text_color="gray"
            )
            info.pack(pady=10)
    
    def _export_design(self):
        """Exporta design para CSV"""
        if self.generated_design is None:
            messagebox.showwarning("Aviso", "Gere um design primeiro")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="ccd_design.csv"
        )
        
        if file_path:
            try:
                self.generated_design.to_csv(file_path, index=False)
                messagebox.showinfo("Sucesso", f"Design exportado para:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao exportar: {str(e)}")
    
    def _copy_design(self):
        """Copia design para área de transferência"""
        if self.generated_design is None:
            messagebox.showwarning("Aviso", "Gere um design primeiro")
            return
        
        try:
            self.generated_design.to_clipboard(index=False, sep='\t')
            messagebox.showinfo("Sucesso", "Design copiado para área de transferência")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao copiar: {str(e)}")
    
    def _populate_analysis_columns(self):
        """Popula checkboxes para seleção de colunas"""
        if self.data is None:
            return
        
        pd = get_pandas()
        
        # Converte datetime
        for col in self.data.columns:
            if pd.api.types.is_datetime64_any_dtype(self.data[col]):
                self.data[col] = self.data[col].astype(str)
        
        # Identifica colunas numéricas
        numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) < 3:
            messagebox.showerror(
                "Erro",
                "É necessário ter pelo menos 3 colunas numéricas (2 X + 1 Y) para análise CCD."
            )
            self.destroy()
            return
        
        # Checkboxes para X
        self.x_column_vars = {}
        for col in numeric_cols:
            var = ctk.BooleanVar(value=True)
            check = ctk.CTkCheckBox(
                self.x_columns_frame,
                text=col,
                variable=var,
                command=self.update_terms
            )
            check.pack(anchor="w", padx=5, pady=2)
            self.x_column_vars[col] = var
        
        # Atualiza termos inicialmente
        self.update_terms()
        
        # Checkboxes para Y
        self.y_column_vars = {}
        for col in numeric_cols:
            var = ctk.BooleanVar(value=False)
            check = ctk.CTkCheckBox(
                self.y_columns_frame,
                text=col,
                variable=var
            )
            check.pack(anchor="w", padx=5, pady=2)
            self.y_column_vars[col] = var
    
    def update_terms(self):
        """Atualiza termos quadráticos e de interação baseado nas variáveis X selecionadas"""
        # Limpa termos quadráticos anteriores
        for widget in self.quadratic_scroll.winfo_children():
            widget.destroy()
        
        # Limpa termos de interação anteriores
        for widget in self.interaction_scroll.winfo_children():
            widget.destroy()
        
        self.quadratic_vars = {}
        self.interaction_vars = {}
        
        # Obtém variáveis X selecionadas
        selected_x = [col for col, var in self.x_column_vars.items() if var.get()]
        
        # === Termos Quadráticos ===
        if len(selected_x) < 1:
            ctk.CTkLabel(
                self.quadratic_scroll,
                text="Selecione pelo menos 1 variável X para habilitar termos quadráticos",
                text_color="gray",
                font=ctk.CTkFont(size=11)
            ).pack(pady=10)
        else:
            ctk.CTkLabel(
                self.quadratic_scroll,
                text=f"Selecione os termos quadráticos desejados:",
                font=ctk.CTkFont(size=11, weight="bold")
            ).pack(anchor="w", padx=5, pady=(5, 10))
            
            for x in selected_x:
                term_name = f"{x}²"
                var = ctk.BooleanVar(value=True)  # Marcado por padrão
                cb = ctk.CTkCheckBox(
                    self.quadratic_scroll,
                    text=term_name,
                    variable=var,
                    font=ctk.CTkFont(size=11)
                )
                cb.pack(anchor="w", padx=20, pady=2)
                self.quadratic_vars[x] = var
        
        # === Termos de Interação ===
        if len(selected_x) < 2:
            ctk.CTkLabel(
                self.interaction_scroll,
                text="Selecione pelo menos 2 variáveis X para habilitar interações",
                text_color="gray",
                font=ctk.CTkFont(size=11)
            ).pack(pady=10)
        else:
            ctk.CTkLabel(
                self.interaction_scroll,
                text=f"Selecione os termos de interação desejados:",
                font=ctk.CTkFont(size=11, weight="bold")
            ).pack(anchor="w", padx=5, pady=(5, 10))
            
            from itertools import combinations
            for x1, x2 in combinations(selected_x, 2):
                term_name = f"{x1} × {x2}"
                var = ctk.BooleanVar(value=True)  # Marcado por padrão
                cb = ctk.CTkCheckBox(
                    self.interaction_scroll,
                    text=term_name,
                    variable=var,
                    font=ctk.CTkFont(size=11)
                )
                cb.pack(anchor="w", padx=20, pady=2)
                self.interaction_vars[(x1, x2)] = var
    
    def _calculate_analysis(self):
        """Calcula análise CCD"""
        
        # Obtém colunas selecionadas
        x_columns = [col for col, var in self.x_column_vars.items() if var.get()]
        y_columns = [col for col, var in self.y_column_vars.items() if var.get()]
        
        if len(x_columns) < 2:
            messagebox.showwarning("Aviso", "Selecione pelo menos 2 colunas X")
            return
        
        if len(y_columns) < 1:
            messagebox.showwarning("Aviso", "Selecione pelo menos 1 coluna Y")
            return
        
        # Verifica sobreposição
        overlap = set(x_columns) & set(y_columns)
        if overlap:
            messagebox.showwarning("Aviso", "Uma coluna não pode ser X e Y ao mesmo tempo")
            return
        
        try:
            # Obtém termos selecionados
            selected_quadratic = [col for col, var in self.quadratic_vars.items() if var.get()]
            selected_interactions = [(x1, x2) for (x1, x2), var in self.interaction_vars.items() if var.get()]
            
            # Prepara dados
            analysis_data = self.data[x_columns + y_columns].copy()
            
            # Divide por coluna de resposta se múltiplos Y
            if len(y_columns) > 1:
                dataframes = split_dataframes_by_response(analysis_data, y_columns)
            else:
                dataframes = {y_columns[0]: analysis_data}
            
            # Calcula análise para cada Y
            self.analysis_results = {}
            for y_col, df in dataframes.items():
                result = calculate_ccd_analysis(
                    df,
                    y_col,
                    quadratic_terms=selected_quadratic,
                    interaction_terms=selected_interactions
                )
                self.analysis_results[y_col] = result
            
            # Exibe resultados
            self.current_response = list(self.analysis_results.keys())[0]
            self._display_results()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao calcular análise:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def _display_results(self):
        """Exibe resultados da análise"""
        
        # Limpa resultados anteriores
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        if not self.analysis_results:
            return
        
        # Se múltiplas respostas, cria tabview
        if len(self.analysis_results) > 1:
            response_tabs = ctk.CTkTabview(self.results_frame, fg_color="#2b2b2b")
            response_tabs.pack(fill="both", expand=True)
            
            for y_col in self.analysis_results.keys():
                tab = response_tabs.add(y_col)
                self._create_result_tab(tab, y_col)
        else:
            # Uma única resposta
            self._create_result_tab(self.results_frame, self.current_response)
    
    def _create_result_tab(self, parent, response_col):
        """Cria aba de resultados para uma resposta"""
        
        result = self.analysis_results[response_col]
        
        # Container direto sem scroll interno (usa o scroll da aba principal)
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.pack(fill="both", expand=True)
        
        # Equação
        eq_frame = ctk.CTkFrame(container, fg_color="#1e1e1e", corner_radius=10)
        eq_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            eq_frame,
            text="📐 Equação do Modelo",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        ctk.CTkLabel(
            eq_frame,
            text=result['equation'],
            font=ctk.CTkFont(size=11),
            wraplength=1200
        ).pack(pady=(5, 10), padx=10)
        
        # Grid de tabelas e gráficos
        content = ctk.CTkFrame(container, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Coluna esquerda - Tabelas
        left_col = ctk.CTkFrame(content, fg_color="transparent")
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        self._create_table(left_col, "Parameter Estimates", result['parameter_estimates'])
        self._create_table(left_col, "ANOVA", result['anova_table'])
        self._create_table(left_col, "Summary of Fit", result['summary_of_fit'])
        self._create_table(left_col, "Lack of Fit", self._format_lack_of_fit(result['lack_of_fit']))
        
        # Coluna direita - Gráficos
        right_col = ctk.CTkFrame(content, fg_color="transparent")
        right_col.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        self._create_charts(right_col, result, response_col)
    
    def _create_table(self, parent, title, df):
        """Cria uma tabela formatada"""
        
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", pady=10)
        
        # Título
        ctk.CTkLabel(
            frame,
            text=f"📊 {title}",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(15, 10), padx=15, anchor="w")
        
        # Tabela
        table_frame = ctk.CTkScrollableFrame(frame, height=180)
        table_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        # Header
        header_frame = ctk.CTkFrame(table_frame, fg_color="#144870")
        header_frame.pack(fill="x", pady=(0, 1))
        
        for i, col in enumerate(df.columns):
            label = ctk.CTkLabel(
                header_frame,
                text=col,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color="white",
                width=120
            )
            label.grid(row=0, column=i, padx=2, pady=8, sticky="ew")
        
        # Linhas
        for idx in range(len(df)):
            row_frame = ctk.CTkFrame(
                table_frame,
                fg_color="#2b2b2b" if idx % 2 == 0 else "#1e1e1e"
            )
            row_frame.pack(fill="x", pady=1)
            
            for i, col in enumerate(df.columns):
                value = df.iloc[idx][col]
                if isinstance(value, (int, float)):
                    # Formatação melhorada de números
                    if abs(value) < 0.0001 and value != 0:
                        text = f"{value:.4e}"
                    elif abs(value) < 1:
                        text = f"{value:.6f}".rstrip('0').rstrip('.')
                    elif abs(value) < 100:
                        text = f"{value:.4f}".rstrip('0').rstrip('.')
                    else:
                        text = f"{value:.2f}"
                else:
                    text = str(value)
                
                label = ctk.CTkLabel(
                    row_frame,
                    text=text,
                    font=ctk.CTkFont(size=10),
                    width=120
                )
                label.grid(row=0, column=i, padx=2, pady=5, sticky="ew")
    
    def _format_lack_of_fit(self, lof_dict):
        """Formata dicionário de Lack of Fit como DataFrame"""
        pd = get_pandas()
        
        data = {
            'Source': ['Lack of Fit', 'Pure Error', 'Total'],
            'DF': [
                lof_dict['grausLiberdade']['lackOfFit'],
                lof_dict['grausLiberdade']['erroPuro'],
                lof_dict['grausLiberdade']['total']
            ],
            'SS': [
                lof_dict['sQuadrados']['lackOfFit'],
                lof_dict['sQuadrados']['erroPuro'],
                lof_dict['sQuadrados']['total']
            ],
            'MS': [
                lof_dict['mQuadrados']['lackOfFit'],
                lof_dict['mQuadrados']['erroPuro'],
                0
            ],
            'F-Ratio': [lof_dict['fRatio'], 0, 0],
            'Prob > F': [lof_dict['probF'], 0, 0]
        }
        
        return pd.DataFrame(data)
    
    def _create_charts(self, parent, result, response_col):
        """Cria gráficos de predito vs observado e superfície 3D"""
        
        # === Gráfico 1: Scatter Predito vs Observado ===
        chart_frame1 = ctk.CTkFrame(parent)
        chart_frame1.pack(fill="both", expand=True, pady=10)
        
        ctk.CTkLabel(
            chart_frame1,
            text="📈 Predito vs Observado (Scatter)",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(15, 10), padx=15, anchor="w")
        
        # Cria figura scatter
        fig1 = Figure(figsize=(7, 5), dpi=100, facecolor='white')
        ax1 = fig1.add_subplot(111)
        
        y = result['y']
        y_pred = result['y_predicted']
        
        # Scatter plot
        ax1.scatter(y, y_pred, alpha=0.7, s=60, color='#1f77b4', edgecolors='black', linewidth=0.5)
        
        # Linha de referência (y = x)
        min_val = min(min(y), min(y_pred))
        max_val = max(max(y), max(y_pred))
        margin = (max_val - min_val) * 0.05
        ax1.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], 
                'r--', linewidth=2, label='Linha Ideal (y = x)', alpha=0.8)
        
        ax1.set_xlabel('Valores Observados', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Valores Preditos', fontsize=12, fontweight='bold')
        ax1.set_title(f'Predito vs Observado - {response_col}', fontsize=14, fontweight='bold', pad=15)
        ax1.legend(loc='best', framealpha=0.9, fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Adiciona R² no gráfico
        r2 = result.get('summary_of_fit', None)
        if r2 is not None:
            r2_val = r2[r2['metric'] == 'r2']['value'].values[0]
            ax1.text(0.05, 0.95, f'R² = {r2_val:.4f}', 
                   transform=ax1.transAxes, fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig1.tight_layout()
        
        # Canvas scatter
        canvas_frame1 = ctk.CTkFrame(chart_frame1, fg_color="#ffffff", corner_radius=5)
        canvas_frame1.pack(fill="both", expand=True, padx=15, pady=(0, 10))
        
        canvas1 = FigureCanvasTkAgg(fig1, canvas_frame1)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
        # Botão exportar scatter
        export_frame1 = ctk.CTkFrame(chart_frame1, fg_color="transparent")
        export_frame1.pack(pady=(0, 10))
        add_chart_export_button(export_frame1, fig1, f"ccd_scatter_{response_col}")
        
        # === Gráfico 2: Linha Y vs Y Predito ===
        chart_frame2 = ctk.CTkFrame(parent)
        chart_frame2.pack(fill="both", expand=True, pady=10)
        
        ctk.CTkLabel(
            chart_frame2,
            text="📉 Comparação Real vs Predito (Linha)",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(15, 10), padx=15, anchor="w")
        
        # Cria figura linha
        fig2 = Figure(figsize=(7, 5), dpi=100, facecolor='white')
        ax2 = fig2.add_subplot(111)
        
        # Ordena por Y (valores reais do menor para o maior)
        np = get_numpy()
        y_array = np.array(y)
        y_pred_array = np.array(y_pred)
        sorted_indices = np.argsort(y_array)
        y_sorted = y_array[sorted_indices]
        y_pred_sorted = y_pred_array[sorted_indices]
        
        indices = range(len(y_sorted))
        ax2.plot(indices, y_sorted, 'o-', color='#2ca02c', linewidth=2, markersize=6, 
                label='Valores Reais', alpha=0.8)
        ax2.plot(indices, y_pred_sorted, 's--', color='#ff7f0e', linewidth=2, markersize=6, 
                label='Valores Preditos', alpha=0.8)
        
        ax2.set_xlabel('Observação (ordenada por Y)', fontsize=12, fontweight='bold')
        ax2.set_ylabel(response_col, fontsize=12, fontweight='bold')
        ax2.set_title(f'Real vs Predito - {response_col}', fontsize=14, fontweight='bold', pad=15)
        ax2.legend(loc='best', framealpha=0.9, fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        fig2.tight_layout()
        
        # Canvas linha
        canvas_frame2 = ctk.CTkFrame(chart_frame2, fg_color="#ffffff", corner_radius=5)
        canvas_frame2.pack(fill="both", expand=True, padx=15, pady=(0, 10))
        
        canvas2 = FigureCanvasTkAgg(fig2, canvas_frame2)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
        # Botão exportar linha
        export_frame2 = ctk.CTkFrame(chart_frame2, fg_color="transparent")
        export_frame2.pack(pady=(0, 10))
        add_chart_export_button(export_frame2, fig2, f"ccd_linha_{response_col}")
        
        # === Gráfico 3: Superfície 3D ===
        self._create_3d_surface_chart(parent, result, response_col)
    
    def _create_3d_surface_chart(self, parent, result, response_col):
        """Cria gráfico de superfície 3D"""
        
        surface_frame = ctk.CTkFrame(parent)
        surface_frame.pack(fill="both", expand=True, pady=10)
        
        ctk.CTkLabel(
            surface_frame,
            text="🌐 Superfície de Resposta 3D",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(15, 10), padx=15, anchor="w")
        
        # Obtém fatores (colunas X originais, não os termos calculados)
        param_estimates = result['parameter_estimates']
        all_terms = param_estimates['term'].tolist()
        
        # Filtra apenas os fatores originais (não quadráticos nem interações)
        factors = [term for term in all_terms 
                  if term != 'Intercept' 
                  and '_squared' not in term 
                  and '_interaction_' not in term]
        
        if len(factors) < 2:
            ctk.CTkLabel(
                surface_frame,
                text="⚠️ É necessário pelo menos 2 fatores para gerar superfície 3D",
                text_color="orange",
                font=ctk.CTkFont(size=11)
            ).pack(pady=20)
            return
        
        # Controles de seleção de fatores
        controls_frame = ctk.CTkFrame(surface_frame, fg_color="transparent")
        controls_frame.pack(fill="x", padx=15, pady=10)
        
        # Fator X1
        ctk.CTkLabel(
            controls_frame,
            text="Fator X1 (eixo X):",
            font=ctk.CTkFont(size=11, weight="bold")
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        factor_x1_var = tk.StringVar(value=factors[0])
        factor_x1_combo = ctk.CTkComboBox(
            controls_frame,
            variable=factor_x1_var,
            values=factors,
            width=150,
            state="readonly"
        )
        factor_x1_combo.grid(row=0, column=1, padx=5, pady=5)
        
        # Fator X2
        ctk.CTkLabel(
            controls_frame,
            text="Fator X2 (eixo Y):",
            font=ctk.CTkFont(size=11, weight="bold")
        ).grid(row=0, column=2, padx=5, pady=5, sticky="w")
        
        factor_x2_var = tk.StringVar(value=factors[1] if len(factors) > 1 else factors[0])
        factor_x2_combo = ctk.CTkComboBox(
            controls_frame,
            variable=factor_x2_var,
            values=factors,
            width=150,
            state="readonly"
        )
        factor_x2_combo.grid(row=0, column=3, padx=5, pady=5)
        
        # Frame para o gráfico 3D
        graph_3d_container = ctk.CTkFrame(surface_frame, fg_color="transparent")
        graph_3d_container.pack(fill="both", expand=True, padx=15, pady=10)
        
        # Botão para gerar
        def generate_surface():
            self._generate_3d_surface(
                graph_3d_container, 
                result, 
                response_col, 
                factor_x1_var.get(), 
                factor_x2_var.get()
            )
        
        generate_btn = ctk.CTkButton(
            controls_frame,
            text="🔮 Gerar Superfície 3D",
            command=generate_surface,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#1f6aa5",
            hover_color="#144870",
            height=35,
            width=180
        )
        generate_btn.grid(row=0, column=4, padx=15, pady=5)
        
        # Gera automaticamente na primeira vez
        generate_surface()
    
    def _generate_3d_surface(self, container, result, response_col, factor_x1, factor_x2):
        """Gera o gráfico 3D de superfície"""
        
        if factor_x1 == factor_x2:
            messagebox.showwarning("Aviso", "Selecione fatores diferentes para X1 e X2")
            return
        
        # Limpa container
        for widget in container.winfo_children():
            widget.destroy()
        
        pd = get_pandas()
        np = get_numpy()
        
        # Obtém coeficientes
        param_estimates = result['parameter_estimates']
        
        # Cria dicionário de coeficientes
        coef_dict = {}
        for idx, row in param_estimates.iterrows():
            coef_dict[row['term']] = row['estimate']
        
        # Prepara dados para grid
        # Usa os dados originais se disponíveis
        if hasattr(self, 'data') and self.data is not None:
            x1_data = self.data[factor_x1].values if factor_x1 in self.data.columns else np.array([0])
            x2_data = self.data[factor_x2].values if factor_x2 in self.data.columns else np.array([0])
            y_data = self.data[response_col].values if response_col in self.data.columns else result['y']
        else:
            # Usa valores padrão
            x1_data = np.linspace(-1, 1, 20)
            x2_data = np.linspace(-1, 1, 20)
            y_data = result['y']
        
        # Cria grid
        x1_min, x1_max = x1_data.min(), x1_data.max()
        x2_min, x2_max = x2_data.min(), x2_data.max()
        
        x1_range = x1_max - x1_min
        x2_range = x2_max - x2_min
        x1_min -= x1_range * 0.1
        x1_max += x1_range * 0.1
        x2_min -= x2_range * 0.1
        x2_max += x2_range * 0.1
        
        x1_grid = np.linspace(x1_min, x1_max, 30)
        x2_grid = np.linspace(x2_min, x2_max, 30)
        X1_mesh, X2_mesh = np.meshgrid(x1_grid, x2_grid)
        
        # Calcula predições para a superfície
        Z_mesh = np.zeros_like(X1_mesh)
        
        for i in range(X1_mesh.shape[0]):
            for j in range(X1_mesh.shape[1]):
                x1_val = X1_mesh[i, j]
                x2_val = X2_mesh[i, j]
                
                # Calcula predição
                pred = coef_dict.get('Intercept', 0)
                pred += coef_dict.get(factor_x1, 0) * x1_val
                pred += coef_dict.get(factor_x2, 0) * x2_val
                pred += coef_dict.get(f'{factor_x1}_squared', 0) * (x1_val ** 2)
                pred += coef_dict.get(f'{factor_x2}_squared', 0) * (x2_val ** 2)
                pred += coef_dict.get(f'{factor_x1}_interaction_{factor_x2}', 0) * x1_val * x2_val
                pred += coef_dict.get(f'{factor_x2}_interaction_{factor_x1}', 0) * x1_val * x2_val
                
                # Adiciona outros fatores com médias (se houver)
                for term in coef_dict.keys():
                    if (term not in ['Intercept', factor_x1, factor_x2] and
                        '_squared' not in term and '_interaction_' not in term):
                        if hasattr(self, 'data') and term in self.data.columns:
                            pred += coef_dict[term] * self.data[term].mean()
                
                Z_mesh[i, j] = pred
        
        # Cria figura 3D
        fig = Figure(figsize=(10, 7), facecolor='#f0f0f0')
        ax = fig.add_subplot(111, projection='3d')
        
        # Superfície
        surf = ax.plot_surface(X1_mesh, X2_mesh, Z_mesh, cmap='viridis', alpha=0.8,
                              edgecolor='none', antialiased=True)
        
        # Pontos originais
        ax.scatter(x1_data, x2_data, y_data, c='red', marker='o', s=50,
                  label='Dados Originais', alpha=0.6, edgecolors='darkred', linewidths=1)
        
        # Labels
        ax.set_xlabel(factor_x1, fontsize=11, fontweight='bold', labelpad=10)
        ax.set_ylabel(factor_x2, fontsize=11, fontweight='bold', labelpad=10)
        ax.set_zlabel(response_col, fontsize=11, fontweight='bold', labelpad=10)
        ax.set_title(f'Superfície de Resposta: {response_col} vs {factor_x1} e {factor_x2}',
                    fontsize=13, fontweight='bold', pad=20)
        
        # Colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label=response_col)
        
        ax.legend(loc='upper left', frameon=True, shadow=True)
        ax.view_init(elev=25, azim=45)
        
        fig.tight_layout()
        
        # Canvas
        canvas_frame = ctk.CTkFrame(container, fg_color="#ffffff", corner_radius=5)
        canvas_frame.pack(fill="both", expand=True)
        
        canvas = FigureCanvasTkAgg(fig, canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
        # Botão exportar
        export_frame = ctk.CTkFrame(container, fg_color="transparent")
        export_frame.pack(pady=(5, 0))
        add_chart_export_button(
            export_frame, fig,
            f"ccd_superficie_3d_{response_col}_{factor_x1}_{factor_x2}"
        )
