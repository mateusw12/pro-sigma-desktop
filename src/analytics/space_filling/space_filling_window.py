"""
Space Filling Design Analysis Window
Interface para análise Space Filling com múltiplas respostas
"""

import customtkinter as ctk
from tkinter import ttk, messagebox

from src.utils.lazy_imports import (
    get_pandas, 
    get_numpy, 
    get_matplotlib_figure, 
    get_matplotlib_backend
)
from src.utils.ui_components import (
    create_minitab_style_table,
    create_vertical_stats_table
)
from src.analytics.space_filling.space_filling_utils import (
    calculate_space_filling_analysis,
    validate_space_filling_data,
    generate_equation
)
from src.analytics.space_filling.generate_experiment_window import GenerateExperimentWindow


class SpaceFillingWindow(ctk.CTkToplevel):
    """Janela para Análise Space Filling Design"""
    
    def __init__(self, parent, data=None):
        super().__init__(parent)
        self.title("Space Filling Design")
        self.geometry("1600x900")
        self.minsize(1400, 800)
        
        # Maximizar janela
        try:
            self.state("zoomed")
        except Exception:
            pass
        
        # Configurar como modal
        self.transient(parent)
        self.grab_set()
        
        self.parent = parent
        self.data = data
        self.results = {}
        self.current_response = None
        self.selected_x_cols = []
        self.selected_y_cols = []
        self.interaction_terms = []
        
        self._build_ui()
        
        if data is not None:
            self._populate_columns()
    
    def _build_ui(self):
        """Constrói a interface"""
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=16, pady=16)
        
        # === PAINEL SUPERIOR: Configuração ===
        config_frame = ctk.CTkFrame(main_frame)
        config_frame.pack(fill="x", pady=(0, 12))
        
        # Título
        title_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        title_frame.pack(fill="x", padx=12, pady=(8, 0))
        
        ctk.CTkLabel(
            title_frame,
            text="Space Filling Design",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(side="left")
        
        # Botão Gerar Experimento
        ctk.CTkButton(
            title_frame,
            text="📋 Gerar Experimento",
            command=self._open_generate_experiment,
            height=32,
            fg_color="#9B59B6",
            hover_color="#7D3C98"
        ).pack(side="right", padx=5)
        
        # Seleção de colunas
        selection_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        selection_frame.pack(fill="x", padx=12, pady=8)
        
        # Colunas X
        x_frame = ctk.CTkFrame(selection_frame, fg_color="transparent")
        x_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        ctk.CTkLabel(
            x_frame,
            text="Variáveis X (Fatores):",
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(anchor="w")
        
        self.x_columns_frame = ctk.CTkScrollableFrame(x_frame, height=100)
        self.x_columns_frame.pack(fill="both", expand=True, pady=(4, 0))
        
        # Colunas Y
        y_frame = ctk.CTkFrame(selection_frame, fg_color="transparent")
        y_frame.pack(side="left", fill="both", expand=True, padx=5)
        
        ctk.CTkLabel(
            y_frame,
            text="Variáveis Y (Respostas):",
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(anchor="w")
        
        self.y_columns_frame = ctk.CTkScrollableFrame(y_frame, height=100)
        self.y_columns_frame.pack(fill="both", expand=True, pady=(4, 0))
        
        # Interações
        interaction_frame = ctk.CTkFrame(selection_frame, fg_color="transparent")
        interaction_frame.pack(side="left", fill="both", expand=True, padx=(5, 0))
        
        ctk.CTkLabel(
            interaction_frame,
            text="Interações (opcional):",
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(anchor="w")
        
        inter_input_frame = ctk.CTkFrame(interaction_frame, fg_color="transparent")
        inter_input_frame.pack(fill="x", pady=(4, 0))
        
        self.interaction_entry = ctk.CTkEntry(inter_input_frame, placeholder_text="Ex: X1*X2")
        self.interaction_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        ctk.CTkButton(
            inter_input_frame,
            text="➕",
            command=self._add_interaction,
            width=30
        ).pack(side="left")
        
        self.interactions_listbox = ctk.CTkTextbox(interaction_frame, height=60)
        self.interactions_listbox.pack(fill="both", expand=True, pady=(4, 0))
        
        # Opções de análise
        options_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        options_frame.pack(fill="x", padx=12, pady=(8, 8))
        
        self.full_model_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            options_frame,
            text="Modelo Completo (2ª ordem + quadráticos)",
            variable=self.full_model_var,
            command=self._toggle_full_model
        ).pack(side="left", padx=5)
        
        self.recalculate_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            options_frame,
            text="Modelo Reduzido (recalculate)",
            variable=self.recalculate_var
        ).pack(side="left", padx=5)
        
        # Botão Calcular
        ctk.CTkButton(
            options_frame,
            text="🔍 Calcular Análise",
            command=self._calculate_analysis,
            height=36,
            fg_color="#27AE60",
            hover_color="#1E8449"
        ).pack(side="right", padx=5)
        
        # === PAINEL INFERIOR: Resultados ===
        results_notebook = ctk.CTkTabview(main_frame)
        results_notebook.pack(fill="both", expand=True)
        
        # Guarda referência para adicionar tabs dinamicamente
        self.results_notebook = results_notebook
    
    def _populate_columns(self):
        """Popula checkboxes com colunas disponíveis"""
        if self.data is None:
            return
        
        # Limpa frames
        for widget in self.x_columns_frame.winfo_children():
            widget.destroy()
        for widget in self.y_columns_frame.winfo_children():
            widget.destroy()
        
        self.x_checkboxes = {}
        self.y_checkboxes = {}
        
        # Identifica colunas numéricas
        numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()
        
        # Cria checkboxes para X
        for col in numeric_cols:
            var = ctk.BooleanVar(value=False)
            checkbox = ctk.CTkCheckBox(
                self.x_columns_frame,
                text=col,
                variable=var
            )
            checkbox.pack(anchor="w", pady=2)
            self.x_checkboxes[col] = var
        
        # Cria checkboxes para Y
        for col in numeric_cols:
            var = ctk.BooleanVar(value=False)
            checkbox = ctk.CTkCheckBox(
                self.y_columns_frame,
                text=col,
                variable=var
            )
            checkbox.pack(anchor="w", pady=2)
            self.y_checkboxes[col] = var
    
    def _add_interaction(self):
        """Adiciona termo de interação"""
        interaction = self.interaction_entry.get().strip()
        if not interaction:
            return
        
        if interaction not in self.interaction_terms:
            self.interaction_terms.append(interaction)
            self._update_interactions_display()
        
        self.interaction_entry.delete(0, "end")
    
    def _update_interactions_display(self):
        """Atualiza exibição das interações"""
        self.interactions_listbox.delete("1.0", "end")
        for term in self.interaction_terms:
            self.interactions_listbox.insert("end", f"{term}\n")
    
    def _toggle_full_model(self):
        """Ativa/desativa modo de modelo completo"""
        if self.full_model_var.get():
            # Obtém colunas X selecionadas
            selected_x = [col for col, var in self.x_checkboxes.items() if var.get()]
            
            if len(selected_x) < 1:
                messagebox.showwarning("Aviso", "Selecione pelo menos 1 variável X antes de ativar o modelo completo")
                self.full_model_var.set(False)
                return
            
            # Gera termos automaticamente
            self.interaction_terms = self._generate_full_model_terms(selected_x)
            self._update_interactions_display()
            
            # Desabilita entrada manual de interações
            self.interaction_entry.configure(state="disabled")
            self.interactions_listbox.configure(state="disabled")
        else:
            # Limpa termos gerados automaticamente
            self.interaction_terms = []
            self._update_interactions_display()
            
            # Habilita entrada manual
            self.interaction_entry.configure(state="normal")
            self.interactions_listbox.configure(state="normal")
    
    def _generate_full_model_terms(self, x_cols):
        """Gera automaticamente termos de 2ª ordem e quadráticos"""
        terms = []
        
        # Termos quadráticos: X1*X1, X2*X2, etc
        for col in x_cols:
            terms.append(f"{col}*{col}")
        
        # Interações de 2ª ordem: X1*X2, X1*X3, X2*X3, etc
        for i in range(len(x_cols)):
            for j in range(i + 1, len(x_cols)):
                terms.append(f"{x_cols[i]}*{x_cols[j]}")
        
        return terms
    
    def _calculate_analysis(self):
        """Calcula a análise Space Filling"""
        if self.data is None:
            messagebox.showerror("Erro", "Nenhum dado carregado")
            return
        
        # Obtém colunas selecionadas
        self.selected_x_cols = [col for col, var in self.x_checkboxes.items() if var.get()]
        self.selected_y_cols = [col for col, var in self.y_checkboxes.items() if var.get()]
        
        if len(self.selected_x_cols) < 1:
            messagebox.showwarning("Aviso", "Selecione pelo menos 1 variável X")
            return
        
        if len(self.selected_y_cols) < 1:
            messagebox.showwarning("Aviso", "Selecione pelo menos 1 variável Y")
            return
        
        # Gera termos de interação
        interaction_terms = None
        if self.full_model_var.get():
            # Modelo completo: gera automaticamente termos de 2ª ordem e quadráticos
            interaction_terms = self._generate_full_model_terms(self.selected_x_cols)
            # Atualiza display
            self.interaction_terms = interaction_terms
            self._update_interactions_display()
        elif len(self.interaction_terms) > 0:
            # Usa termos manuais
            interaction_terms = self.interaction_terms
        
        # Valida dados
        all_cols = self.selected_x_cols + self.selected_y_cols
        data_subset = self.data[all_cols]
        
        is_valid, error_msg = validate_space_filling_data(data_subset, self.selected_y_cols)
        if not is_valid:
            messagebox.showerror("Erro de Validação", error_msg)
            return
        
        try:
            # Calcula análise
            self.results = calculate_space_filling_analysis(
                data_subset,
                self.selected_y_cols,
                interaction_terms,
                self.recalculate_var.get()
            )
            
            # Exibe resultados
            self._display_results()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao calcular análise: {str(e)}")
    
    def _display_results(self):
        """Exibe resultados em tabs para cada Y"""
        # Limpa tabs existentes
        for tab_name in self.results_notebook.winfo_children():
            tab_name.destroy()
        
        # Cria tab para cada resposta
        for response_col, result in self.results.items():
            if 'error' in result:
                continue
            
            tab = self.results_notebook.add(response_col)
            self._create_results_tab(tab, response_col, result)
        
        # Seleciona primeira tab
        if len(self.results) > 0:
            first_response = list(self.results.keys())[0]
            self.results_notebook.set(first_response)
    
    def _create_results_tab(self, tab, response_col, result):
        """Cria conteúdo da tab de resultados"""
        # Cria scroll frame
        scroll_frame = ctk.CTkScrollableFrame(tab)
        scroll_frame.pack(fill="both", expand=True, padx=15, pady=15)
        
        # === CABEÇALHO COM INFORMAÇÕES DA RESPOSTA ===
        header_frame = ctk.CTkFrame(scroll_frame, fg_color="#1f538d", corner_radius=8)
        header_frame.pack(fill="x", pady=(0, 15))
        
        ctk.CTkLabel(
            header_frame,
            text=f"📊 Análise de Regressão - Variável Resposta: {response_col}",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="white"
        ).pack(pady=12)
        
        # === Equação do Modelo ===
        equation_frame = ctk.CTkFrame(scroll_frame, corner_radius=8)
        equation_frame.pack(fill="x", pady=(0, 15))
        
        ctk.CTkLabel(
            equation_frame,
            text="📐 Equação do Modelo:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=15, pady=(12, 6))
        
        param_names = ['Intercept'] + self.selected_x_cols
        if self.interaction_terms:
            param_names += self.interaction_terms
        
        equation = generate_equation(result['betas'], param_names, result['mean'])
        
        equation_text = ctk.CTkTextbox(equation_frame, height=70, font=ctk.CTkFont(size=11))
        equation_text.pack(fill="x", padx=15, pady=(0, 12))
        equation_text.insert("1.0", equation)
        equation_text.configure(state="disabled", fg_color="#2b2b2b")
        
        # === ANOVA Table ===
        anova_frame = ctk.CTkFrame(scroll_frame, corner_radius=8)
        anova_frame.pack(fill="x", pady=(0, 15))
        
        ctk.CTkLabel(
            anova_frame,
            text="📊 Análise de Variância (ANOVA):",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=15, pady=(12, 6))
        
        anova_tree = self._create_anova_table(anova_frame, result['anovaTable'])
        
        # === Summary of Fit ===
        summary_frame = ctk.CTkFrame(scroll_frame, corner_radius=8)
        summary_frame.pack(fill="x", pady=(0, 15))
        
        ctk.CTkLabel(
            summary_frame,
            text="📈 Resumo do Ajuste:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=15, pady=(12, 6))
        
        self._create_summary_table(summary_frame, result['summarOfFit'])
        
        # === Parameter Estimates ===
        param_frame = ctk.CTkFrame(scroll_frame, corner_radius=8)
        param_frame.pack(fill="x", pady=(0, 15))
        
        ctk.CTkLabel(
            param_frame,
            text="🔢 Estimativas dos Parâmetros:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=15, pady=(12, 6))
        
        self._create_parameter_estimates_table(param_frame, result['parameterEstimates'])
        
        # === Gráficos ===
        charts_frame = ctk.CTkFrame(scroll_frame, corner_radius=8)
        charts_frame.pack(fill="both", expand=True, pady=(0, 15))
        
        ctk.CTkLabel(
            charts_frame,
            text="📉 Visualizações:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=15, pady=(12, 6))
        
        self._create_charts(charts_frame, result, response_col)
        
        # === Gráfico 3D de Superfície de Resposta ===
        surface_frame = ctk.CTkFrame(scroll_frame, corner_radius=8)
        surface_frame.pack(fill="both", expand=True, pady=(0, 15))
        
        ctk.CTkLabel(
            surface_frame,
            text="🌐 Superfície de Resposta 3D:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=15, pady=(12, 6))
        
        self._create_3d_surface_controls(surface_frame, result, response_col)
    
    def _create_anova_table(self, parent, anova_data):
        """Cria tabela ANOVA"""
        headers = ["Fonte", "GL", "SQ", "MQ", "F", "Prob > F"]
        
        # Formata Prob > F
        prob_f_str = f"{anova_data['probF']:.4f}" if anova_data['probF'] >= 0.0001 else "< 0.0001"
        
        data_rows = [
            [
                "Modelo",
                f"{anova_data['grausLiberdade']['modelo']}",
                f"{anova_data['sQuadrados']['modelo']:.4f}",
                f"{anova_data['mQuadrados']['modelo']:.4f}",
                f"{anova_data['fRatio']:.4f}",
                prob_f_str
            ],
            [
                "Erro",
                f"{anova_data['grausLiberdade']['erro']}",
                f"{anova_data['sQuadrados']['erro']:.4f}",
                f"{anova_data['mQuadrados']['erro']:.4f}",
                "",
                ""
            ],
            [
                "Total",
                f"{anova_data['grausLiberdade']['total']}",
                f"{anova_data['sQuadrados']['total']:.4f}",
                "",
                "",
                ""
            ]
        ]
        
        return create_minitab_style_table(
            parent,
            headers=headers,
            data_rows=data_rows,
            title="",
            column_widths=[120, 80, 120, 120, 100, 120]
        )
    
    def _create_summary_table(self, parent, summary_data):
        """Cria tabela Summary of Fit"""
        stats_dict = {
            "R²": f"{summary_data['rQuadrado']:.4f}",
            "R² Ajustado": f"{summary_data['rQuadradoAjustado']:.4f}",
            "RMSE": f"{summary_data['rmse']:.4f}",
            "Média da Resposta": f"{summary_data['media']:.4f}",
            "Observações": f"{summary_data['observacoes']}"
        }
        
        return create_vertical_stats_table(
            parent,
            stats_dict=stats_dict,
            title=""
        )
    
    def _create_parameter_estimates_table(self, parent, param_data):
        """Cria tabela Parameter Estimates"""
        headers = ["Termo", "Estimativa", "Erro Padrão", "t Ratio", "Prob > |t|"]
        
        data_rows = []
        for term, values in param_data.items():
            prob_t_str = f"{values['pValue']:.4f}" if values['pValue'] >= 0.0001 else "< 0.0001"
            data_rows.append([
                term,
                f"{values['estimates']:.4f}",
                f"{values['stdError']:.4f}",
                f"{values['tRatio']:.4f}",
                prob_t_str
            ])
        
        return create_minitab_style_table(
            parent,
            headers=headers,
            data_rows=data_rows,
            title="",
            column_widths=[150, 120, 120, 100, 120]
        )
    
    def _create_charts(self, parent, result, response_col):
        """Cria gráficos"""
        Figure = get_matplotlib_figure()
        FigureCanvasTkAgg = get_matplotlib_backend()
        
        charts_container = ctk.CTkFrame(parent, fg_color="white", corner_radius=8)
        charts_container.pack(fill="both", expand=True, padx=15, pady=(0, 12))
        
        # Figura com 2 subplots
        fig = Figure(figsize=(14, 5.5), facecolor='#f0f0f0')
        
        # Gráfico 1: Overlay (Y vs Y Predicted)
        ax1 = fig.add_subplot(121, facecolor='white')
        ax1.plot(result['y'], 'o-', label='Y Real', markersize=5, linewidth=2, color='#2E86DE', alpha=0.8)
        ax1.plot(result['yPredictedsOdered'], 's-', label='Y Predito', markersize=5, linewidth=2, color='#E67E22', alpha=0.8)
        ax1.set_xlabel('Observações (ordenadas)', fontsize=11, fontweight='bold')
        ax1.set_ylabel(response_col, fontsize=11, fontweight='bold')
        ax1.set_title('Overlay Plot - Real vs Predito', fontsize=12, fontweight='bold', pad=12)
        ax1.legend(frameon=True, shadow=True, loc='best')
        ax1.grid(True, alpha=0.25, linestyle='--')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Gráfico 2: Estimativas dos parâmetros
        ax2 = fig.add_subplot(122, facecolor='white')
        params = list(result['parameterEstimates'].keys())
        estimates = [abs(v['estimates']) for v in result['parameterEstimates'].values()]
        
        # Remove Intercept do gráfico
        if 'Intercept' in params:
            idx = params.index('Intercept')
            params.pop(idx)
            estimates.pop(idx)
        
        if len(params) > 0:
            # Ordena do menor para o maior
            sorted_indices = sorted(range(len(estimates)), key=lambda i: estimates[i])
            params = [params[i] for i in sorted_indices]
            estimates = [estimates[i] for i in sorted_indices]
            
            y_pos = range(len(params))
            colors = ['#27AE60' if e > 0 else '#E74C3C' for e in estimates]
            bars = ax2.barh(y_pos, estimates, color='#2E86DE', alpha=0.7, edgecolor='#1f538d', linewidth=1.5)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(params, fontsize=10)
            ax2.set_xlabel('|Estimativa|', fontsize=11, fontweight='bold')
            ax2.set_title('Importância dos Parâmetros', fontsize=12, fontweight='bold', pad=12)
            ax2.grid(True, alpha=0.25, axis='x', linestyle='--')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            
            # Adiciona valores nas barras
            for i, (bar, val) in enumerate(zip(bars, estimates)):
                width = bar.get_width()
                ax2.text(width, bar.get_y() + bar.get_height()/2, 
                        f' {val:.3f}', 
                        ha='left', va='center', fontsize=9, fontweight='bold')
        
        fig.tight_layout(pad=2.0)
        
        # Adiciona ao frame
        canvas = FigureCanvasTkAgg(fig, charts_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def _create_3d_surface_controls(self, parent, result, response_col):
        """Cria controles para gráfico 3D"""
        controls_frame = ctk.CTkFrame(parent, fg_color="transparent")
        controls_frame.pack(fill="x", padx=15, pady=(0, 10))
        
        # Obtém lista de fatores (exceto Intercept)
        factors = [k for k in result['parameterEstimates'].keys() if k != 'Intercept' and '*' not in k]
        
        if len(factors) < 2:
            ctk.CTkLabel(
                controls_frame,
                text="⚠️ É necessário pelo menos 2 fatores para gerar superfície 3D",
                text_color="#E67E22"
            ).pack(pady=10)
            return
        
        # Seleção de fatores
        ctk.CTkLabel(controls_frame, text="Selecione 2 fatores:", font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w")
        
        factors_selection_frame = ctk.CTkFrame(controls_frame, fg_color="transparent")
        factors_selection_frame.pack(fill="x", pady=5)
        
        # Fator X1
        x1_frame = ctk.CTkFrame(factors_selection_frame, fg_color="transparent")
        x1_frame.pack(side="left", padx=(0, 10))
        ctk.CTkLabel(x1_frame, text="Eixo X:").pack(anchor="w")
        x1_combo = ctk.CTkComboBox(x1_frame, values=factors, width=150)
        x1_combo.set(factors[0])
        x1_combo.pack()
        
        # Fator X2
        x2_frame = ctk.CTkFrame(factors_selection_frame, fg_color="transparent")
        x2_frame.pack(side="left", padx=(0, 10))
        ctk.CTkLabel(x2_frame, text="Eixo Y:").pack(anchor="w")
        x2_combo = ctk.CTkComboBox(x2_frame, values=factors, width=150)
        x2_combo.set(factors[1] if len(factors) > 1 else factors[0])
        x2_combo.pack()
        
        # Botão gerar
        ctk.CTkButton(
            factors_selection_frame,
            text="🔮 Gerar Superfície 3D",
            command=lambda: self._generate_3d_surface(
                parent, result, response_col, x1_combo.get(), x2_combo.get()
            ),
            fg_color="#9B59B6",
            hover_color="#7D3C98"
        ).pack(side="left", padx=10)
        
        # Frame para o gráfico 3D
        self.surface_3d_frame = ctk.CTkFrame(parent, fg_color="white", corner_radius=8)
        self.surface_3d_frame.pack(fill="both", expand=True, padx=15, pady=(0, 12))
    
    def _generate_3d_surface(self, parent, result, response_col, factor_x1, factor_x2):
        """Gera gráfico 3D de superfície de resposta"""
        if factor_x1 == factor_x2:
            messagebox.showwarning("Aviso", "Selecione fatores diferentes para os eixos X e Y")
            return
        
        # Limpa frame anterior
        for widget in self.surface_3d_frame.winfo_children():
            widget.destroy()
        
        Figure = get_matplotlib_figure()
        FigureCanvasTkAgg = get_matplotlib_backend()
        np = get_numpy()
        
        # Obtém os dados originais
        x1_data = self.data[factor_x1].values
        x2_data = self.data[factor_x2].values
        
        # Cria grid para superfície
        x1_min, x1_max = x1_data.min(), x1_data.max()
        x2_min, x2_max = x2_data.min(), x2_data.max()
        
        # Expande um pouco os limites
        x1_range = x1_max - x1_min
        x2_range = x2_max - x2_min
        x1_min -= x1_range * 0.1
        x1_max += x1_range * 0.1
        x2_min -= x2_range * 0.1
        x2_max += x2_range * 0.1
        
        x1_grid = np.linspace(x1_min, x1_max, 50)
        x2_grid = np.linspace(x2_min, x2_max, 50)
        X1_mesh, X2_mesh = np.meshgrid(x1_grid, x2_grid)
        
        # Prepara matriz de predição
        # Precisa ter todas as colunas X na ordem correta
        param_names = list(result['parameterEstimates'].keys())
        betas = [result['parameterEstimates'][p]['estimates'] for p in param_names]
        
        # Cria matriz com valores médios para outros fatores
        n_points = len(x1_grid) * len(x2_grid)
        X_pred = np.ones((n_points, len(param_names)))  # Começa com 1 para intercept
        
        # Preenche com médias dos outros fatores
        col_idx = 1  # Pula intercept
        for param in param_names[1:]:  # Pula Intercept
            if '*' in param:  # É interação ou quadrático
                terms = param.split('*')
                if terms[0] == factor_x1 and terms[1] == factor_x1:  # X1*X1
                    X_pred[:, col_idx] = X1_mesh.flatten() ** 2
                elif terms[0] == factor_x2 and terms[1] == factor_x2:  # X2*X2
                    X_pred[:, col_idx] = X2_mesh.flatten() ** 2
                elif (terms[0] == factor_x1 and terms[1] == factor_x2) or \
                     (terms[0] == factor_x2 and terms[1] == factor_x1):  # X1*X2
                    X_pred[:, col_idx] = X1_mesh.flatten() * X2_mesh.flatten()
                else:  # Interação com outro fator - usa média
                    mean_vals = []
                    for term in terms:
                        if term in self.data.columns:
                            mean_vals.append(self.data[term].mean())
                    if len(mean_vals) == 2:
                        X_pred[:, col_idx] = mean_vals[0] * mean_vals[1]
            elif param == factor_x1:
                X_pred[:, col_idx] = X1_mesh.flatten()
            elif param == factor_x2:
                X_pred[:, col_idx] = X2_mesh.flatten()
            else:  # Outro fator - usa média
                if param in self.data.columns:
                    X_pred[:, col_idx] = self.data[param].mean()
            col_idx += 1
        
        # Calcula predições
        Z = X_pred @ np.array(betas)
        Z_mesh = Z.reshape(X1_mesh.shape)
        
        # Cria figura 3D
        fig = Figure(figsize=(12, 8), facecolor='#f0f0f0')
        ax = fig.add_subplot(111, projection='3d')
        
        # Superfície
        surf = ax.plot_surface(X1_mesh, X2_mesh, Z_mesh, cmap='viridis', alpha=0.8, 
                              edgecolor='none', antialiased=True)
        
        # Pontos originais
        y_original = self.data[response_col].values
        ax.scatter(x1_data, x2_data, y_original, c='red', marker='o', s=50, 
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
        
        # Ajusta visualização
        ax.view_init(elev=25, azim=45)
        
        fig.tight_layout()
        
        # Adiciona ao frame
        canvas = FigureCanvasTkAgg(fig, self.surface_3d_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def _open_generate_experiment(self):
        """Abre janela para gerar experimento"""
        GenerateExperimentWindow(self)
