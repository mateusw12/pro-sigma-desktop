"""
Logistic Regression Window
Binary classification with logistic regression
"""
import customtkinter as ctk
from tkinter import messagebox
from src.utils.lazy_imports import (
    get_numpy, get_pandas, get_matplotlib, 
    get_matplotlib_figure, get_matplotlib_backend
)
from src.utils.ui_components import create_action_button, create_horizontal_stats_table

from src.analytics.logistic_regression.logistic_regression_utils import (
    calculate_logistic_regression,
    interpret_logistic_results
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


class LogisticRegressionWindow(ctk.CTkToplevel):
    def __init__(self, parent, df):
        super().__init__(parent)

        # Carrega bibliotecas pesadas (lazy)
        pd, np, plt, Figure, FigureCanvasTkAgg = _ensure_libs()
        self.pd = pd
        self.np = np
        self.plt = plt
        self.Figure = Figure
        self.FigureCanvasTkAgg = FigureCanvasTkAgg
        
        self.df = df
        self.results = None
        
        # Window configuration
        self.title("Regressão Logística Binária")
        
        # Allow resizing
        self.resizable(True, True)
        self.minsize(1200, 800)
        
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
            text="📊 Regressão Logística Binária",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=(0, 20))
        
        # Configuration Frame
        config_frame = ctk.CTkFrame(self.main_container)
        config_frame.pack(fill="x", pady=(0, 20))
        
        # Response variable (Y)
        y_section = ctk.CTkFrame(config_frame)
        y_section.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            y_section,
            text="Variável Resposta Y (binária - aceita categórica ou 0/1)",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", pady=(0, 5))
        
        ctk.CTkLabel(
            y_section,
            text="Selecione a coluna target (deve ter exatamente 2 valores únicos)",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        ).pack(anchor="w", padx=10)
        
        self.y_combo = ctk.CTkComboBox(
            y_section,
            values=list(self.df.columns),
            width=300
        )
        self.y_combo.pack(anchor="w", padx=10)
        self.y_combo.set(self.df.columns[-1])  # Última coluna como padrão
        
        # Predictor variables (X)
        x_section = ctk.CTkFrame(config_frame)
        x_section.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            x_section,
            text="Variáveis Preditoras X",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", pady=(0, 5))
        
        # Scrollable frame for X checkboxes
        self.x_scroll_frame = ctk.CTkScrollableFrame(x_section, height=150)
        self.x_scroll_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.x_checkboxes = []
        self.x_vars = []
        
        for col in self.df.columns:
            var = ctk.StringVar(value="on" if col != self.df.columns[-1] else "off")
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
        
        # Categorical variables
        cat_section = ctk.CTkFrame(config_frame)
        cat_section.pack(fill="x", padx=20, pady=10)
        
        cat_header = ctk.CTkFrame(cat_section, fg_color="transparent")
        cat_header.pack(fill="x", pady=(0, 5))
        
        ctk.CTkLabel(
            cat_header,
            text="🏷️ Variáveis Categóricas",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(side="left")
        
        # Botões de seleção rápida
        ctk.CTkButton(
            cat_header,
            text="Marcar Todas",
            command=self.select_all_categorical,
            width=100,
            height=25,
            font=ctk.CTkFont(size=11),
            fg_color="#3b82f6"
        ).pack(side="right", padx=5)
        
        ctk.CTkButton(
            cat_header,
            text="Desmarcar Todas",
            command=self.deselect_all_categorical,
            width=100,
            height=25,
            font=ctk.CTkFont(size=11),
            fg_color="#6b7280"
        ).pack(side="right")
        
        ctk.CTkLabel(
            cat_section,
            text="Selecione as variáveis preditoras que são categóricas (texto/níveis discretos).\nSerão convertidas em variáveis dummy para o modelo.",
            font=ctk.CTkFont(size=11),
            text_color="gray",
            justify="left"
        ).pack(anchor="w", padx=10, pady=(0, 5))
        
        # Scrollable frame for categorical checkboxes
        self.cat_scroll_frame = ctk.CTkScrollableFrame(cat_section, height=120)
        self.cat_scroll_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.cat_checkboxes = []
        self.cat_vars = []
        
        for col in self.df.columns:
            # Auto-detectar se é categórica (object, string ou poucos valores únicos)
            is_categorical = False
            if self.df[col].dtype == 'object':
                is_categorical = True
            elif self.df[col].dtype in ['int64', 'float64']:
                # Se tem poucos valores únicos (<=10), provavelmente é categórica
                n_unique = self.df[col].nunique()
                if n_unique <= 10 and n_unique > 2:  # Mais de 2 mas até 10 valores
                    is_categorical = True
            
            var = ctk.StringVar(value="on" if is_categorical else "off")
            
            # Criar label com indicador visual
            label_text = col
            if is_categorical:
                label_text = f"{col} 🏷️"  # Emoji para indicar auto-detectada
            
            cb = ctk.CTkCheckBox(
                self.cat_scroll_frame,
                text=label_text,
                variable=var,
                onvalue="on",
                offvalue="off"
            )
            cb.pack(anchor="w", pady=2)
            self.cat_checkboxes.append(cb)
            self.cat_vars.append((col, var))
        
        # Action Buttons
        button_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        button_frame.pack(fill="x", padx=20, pady=20)
        
        self.calculate_btn = create_action_button(
            button_frame,
            text="Calcular Regressão Logística",
            command=self.calculate_regression,
            icon="🔍"
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
        
        # Results Container (scrollable)
        self.results_container = ctk.CTkScrollableFrame(self.main_container, height=600)
        self.results_container.pack(fill="both", expand=True, pady=(0, 20))
    
    def select_all_categorical(self):
        """Marca todas as variáveis categóricas"""
        for col, var in self.cat_vars:
            var.set("on")
    
    def deselect_all_categorical(self):
        """Desmarca todas as variáveis categóricas"""
        for col, var in self.cat_vars:
            var.set("off")
    
    def validate_inputs(self) -> bool:
        """Valida entradas do usuário"""
        # Variável resposta
        y_col = self.y_combo.get()
        if not y_col:
            messagebox.showerror("Erro", "Selecione a variável resposta Y!")
            return False
        
        # Verificar se Y tem exatamente 2 valores únicos
        unique_values = self.df[y_col].dropna().unique()
        if len(unique_values) != 2:
            messagebox.showerror(
                "Erro", 
                f"A variável resposta deve ter exatamente 2 valores únicos!\\nValores encontrados ({len(unique_values)}): {list(unique_values)[:5]}"
            )
            return False
        
        # Preditores
        x_cols = [col for col, var in self.x_vars if var.get() == "on"]
        if len(x_cols) < 1:
            messagebox.showerror("Erro", "Selecione pelo menos um preditor X!")
            return False
        
        # Verificar se Y não está nos preditores
        if y_col in x_cols:
            messagebox.showerror("Erro", "A variável resposta não pode ser um preditor!")
            return False
        
        return True
    
    def calculate_regression(self):
        """Calcula regressão logística"""
        if not self.validate_inputs():
            return
        
        self.calculate_btn.configure(state="disabled", text="⏳ Calculando...")
        self.update()
        
        try:
            # Get selected columns
            y_col = self.y_combo.get()
            x_cols = [col for col, var in self.x_vars if var.get() == "on"]
            cat_cols = [col for col, var in self.cat_vars if var.get() == "on" and col in x_cols]
            
            print(f"DEBUG - Y column: {y_col}")
            print(f"DEBUG - X columns: {x_cols}")
            print(f"DEBUG - Categorical columns: {cat_cols}")
            
            # Prepare data
            work_df = self.df[[y_col] + x_cols].copy()
            work_df = work_df.dropna()
            
            print(f"DEBUG - Working dataframe shape: {work_df.shape}")
            print(f"DEBUG - Y unique values: {work_df[y_col].unique()}")
            
            # Verificar se há dados suficientes
            if len(work_df) < 10:
                messagebox.showerror("Erro", "Dados insuficientes após remover valores faltantes!")
                return
            
            # Calcular regressão
            print("DEBUG - Calling calculate_logistic_regression...")
            self.results = calculate_logistic_regression(
                work_df, 
                y_col, 
                x_cols,
                cat_cols
            )
            print("DEBUG - Regression calculated successfully!")
            print(f"DEBUG - self.results type: {type(self.results)}")
            print(f"DEBUG - self.results is None: {self.results is None}")
            if self.results:
                print(f"DEBUG - self.results keys: {self.results.keys()}")
            
            # Adicionar interpretações
            interpretations = interpret_logistic_results(
                self.results['metrics'],
                self.results['parameterEstimates']
            )
            self.results['interpretations'] = interpretations
            print(f"DEBUG - Interpretations added")
            
            # Display results
            print(f"DEBUG - Before display_results, self.results is None: {self.results is None}")
            self.display_results()
            print(f"DEBUG - After display_results")
            
            messagebox.showinfo("Sucesso", "Regressão logística calculada com sucesso!")
            
        except ValueError as e:
            messagebox.showerror("Erro de Validação", f"Erro nos dados:\\n{str(e)}")
            print(f"Logistic regression validation error: {e}")
        except Exception as e:
            messagebox.showerror("Erro no Cálculo", f"Erro ao calcular regressão:\\n{str(e)}\\n\\nVerifique se os dados estão corretos.")
            print(f"Logistic regression error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.calculate_btn.configure(state="normal", text="🔍 Calcular Regressão Logística")
    
    def clear_results(self):
        """Limpa resultados da interface (não apaga self.results)"""
        for widget in self.results_container.winfo_children():
            widget.destroy()
    
    def display_results(self):
        """Exibe resultados da regressão"""
        print("DEBUG - display_results() called")
        
        if not self.results:
            print("DEBUG - No results to display!")
            return
        
        # Limpar widgets anteriores (mas manter self.results)
        self.clear_results()
        
        print(f"DEBUG - Results keys: {self.results.keys()}")
        
        # Title
        title = ctk.CTkLabel(
            self.results_container,
            text="📈 Resultados da Regressão Logística",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title.pack(pady=(10, 20))
        print("DEBUG - Title created")
        
        # Display sections in sequence (like other tools)
        self.display_response_mapping()
        self.display_summary()
        self.display_anova()
        self.display_coefficients()
        self.display_confusion_matrix()
        self.display_sigmoid()
        self.display_equations()
        
        print("DEBUG - All displays complete!")
        
        # Update scroll region
        self.main_container.update_idletasks()
    
    def display_response_mapping(self):
        """Exibe mapeamento da variável resposta"""
        reverse_mapping = self.results.get('reverseMappingResponse', {0: '0', 1: '1'})
        
        info_frame = ctk.CTkFrame(self.results_container)
        info_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            info_frame,
            text=f"📋 Mapeamento da Variável Resposta: 0 = '{reverse_mapping.get(0, '0')}' | 1 = '{reverse_mapping.get(1, '1')}'",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#2563eb"
        ).pack(pady=10)
    
    def display_summary(self):
        """Exibe resumo do modelo"""
        try:
            print("DEBUG - display_summary started")
            metrics = self.results['metrics']
            print(f"DEBUG - Metrics: {metrics}")
        
            # Informação sobre mapeamento da resposta
            reverse_mapping = self.results.get('reverseMappingResponse', {0: '0', 1: '1'})
            
            info_frame = ctk.CTkFrame(self.results_container)
            info_frame.pack(fill="x", padx=10, pady=10)
            
            ctk.CTkLabel(
                info_frame,
                text=f"📋 Mapeamento da Variável Resposta: 0 = '{reverse_mapping.get(0, '0')}' | 1 = '{reverse_mapping.get(1, '1')}'",
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color="#2563eb"
            ).pack(pady=10)
            
            # Whole Model Test
            whole_model_frame = ctk.CTkFrame(self.results_container)
            whole_model_frame.pack(fill="x", padx=10, pady=10)
            
            rows_data = [
                {"Métrica": "Observações", "Valor": f"{metrics['observations']}"},
                {"Métrica": "Graus de Liberdade", "Valor": f"{metrics['df']}"},
                {"Métrica": "Log-Likelihood (Full)", "Valor": f"{metrics['logLikelihoodFull']:.4f}"},
                {"Métrica": "Log-Likelihood (Reduced)", "Valor": f"{metrics['logLikelihoodReduced']:.4f}"},
                {"Métrica": "Chi-Square", "Valor": f"{metrics['chiSquare']:.4f}"},
                {"Métrica": "P-value (Chi-Square)", "Valor": f"{metrics['probChisq']:.4f}"},
                {"Métrica": "R² (McFadden)", "Valor": f"{metrics['rsquare']:.4f}"},
                {"Métrica": "AIC", "Valor": f"{metrics['aic']:.4f}"},
                {"Métrica": "BIC", "Valor": f"{metrics['bic']:.4f}"}
            ]
            
            create_horizontal_stats_table(
                whole_model_frame,
                columns=["Métrica", "Valor"],
                rows_data=rows_data,
                title="Teste do Modelo Completo (Whole Model Test)"
            )
            
            # Classification Metrics
            class_frame = ctk.CTkFrame(self.results_container)
            class_frame.pack(fill="x", padx=10, pady=10)
            
            ctk.CTkLabel(
                class_frame,
                text="🎯 Métricas de Classificação",
                font=ctk.CTkFont(size=16, weight="bold")
            ).pack(pady=(10, 5))
            
            class_data = [
                {"Métrica": "Acurácia", "Valor": f"{metrics['accuracy']*100:.2f}%"},
                {"Métrica": "Precisão", "Valor": f"{metrics['precision']*100:.2f}%"},
                {"Métrica": "Recall (Sensibilidade)", "Valor": f"{metrics['recall']*100:.2f}%"},
                {"Métrica": "F1-Score", "Valor": f"{metrics['f1_score']:.4f}"}
            ]
            
            create_horizontal_stats_table(
                class_frame,
                columns=["Métrica", "Valor"],
                rows_data=class_data,
                title="Métricas de Classificação"
            )
            
            # Interpretações
            interp_frame = ctk.CTkFrame(self.results_container)
            interp_frame.pack(fill="x", padx=10, pady=10)
            
            ctk.CTkLabel(
                interp_frame,
                text="💡 Interpretações",
                font=ctk.CTkFont(size=16, weight="bold")
            ).pack(pady=(10, 5))
            
            for interp in self.results['interpretations']['interpretations']:
                ctk.CTkLabel(
                    interp_frame,
                    text=f"• {interp}",
                    font=ctk.CTkFont(size=12),
                    wraplength=900,
                    justify="left"
                ).pack(anchor="w", padx=20, pady=3)
            
                print("DEBUG - display_summary completed")
        except Exception as e:
            print(f"ERROR in display_summary: {e}")
            import traceback
            traceback.print_exc()
    
    def display_coefficients(self):
        """Exibe coeficientes e estatísticas"""
        coef_frame = ctk.CTkFrame(self.results_container)
        coef_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            coef_frame,
            text="📐 Coeficientes da Regressão",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        param_estimates = self.results['parameterEstimates']
        
        # Convert to dict format for table
        rows_data = []
        for param in param_estimates:
            rows_data.append({
                "Termo": param['term'],
                "Estimativa": f"{param['estimate']:.6f}",
                "Erro Padrão": f"{param['stdError']:.6f}" if not self.np.isnan(param['stdError']) else "N/A",
                "Z-value": f"{param['zValue']:.4f}" if not self.np.isnan(param['zValue']) else "N/A",
                "P-value": f"{param['prob']:.4f}" if not self.np.isnan(param['prob']) else "N/A",
                "LogWorth": f"{param['logWorth']:.4f}" if not self.np.isnan(param['logWorth']) else "N/A"
            })
        
        create_horizontal_stats_table(
            coef_frame,
            columns=["Termo", "Estimativa", "Erro Padrão", "Z-value", "P-value", "LogWorth"],
            rows_data=rows_data,
            title="Estimativas dos Parâmetros"
        )
    
    def display_anova(self):
        """Exibe tabela ANOVA (Likelihood Ratio Tests)"""
        if not self.results.get('anovaTable'):
            return
        
        anova_frame = ctk.CTkFrame(self.results_container)
        anova_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            anova_frame,
            text="📋 ANOVA - Likelihood Ratio Tests",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        rows_data = []
        for row in self.results['anovaTable']:
            rows_data.append({
                "Termo": row['term'],
                "DF": f"{row['df']}",
                "Deviance": f"{row['deviance']:.4f}",
                "P-value": f"{row['prob']:.4f}"
            })
        
        anova_frame = ctk.CTkFrame(self.results_container)
        anova_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            anova_frame,
            text="📋 ANOVA - Likelihood Ratio Tests",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        create_horizontal_stats_table(
            anova_frame,
            columns=["Termo", "DF", "Deviance", "P-value"],
            rows_data=rows_data,
            title="Análise de Variância"
        )
    
    def display_confusion_matrix(self):
        """Exibe matriz de confusão"""
        conf_matrix = self.np.array(self.results['confusionMatrix'])
        reverse_mapping = self.results.get('reverseMappingResponse', {0: '0', 1: '1'})
        
        # Create frame for matrix
        matrix_frame = ctk.CTkFrame(self.results_container)
        matrix_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            matrix_frame,
            text="🎯 Matriz de Confusão",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        # Create table
        table_frame = ctk.CTkFrame(matrix_frame)
        table_frame.pack(padx=20, pady=10)
        
        # Headers
        ctk.CTkLabel(table_frame, text="", width=120).grid(row=0, column=0, padx=5, pady=5)
        ctk.CTkLabel(table_frame, text=f"Predito: {reverse_mapping[0]}", font=ctk.CTkFont(weight="bold"), width=120).grid(row=0, column=1, padx=5, pady=5)
        ctk.CTkLabel(table_frame, text=f"Predito: {reverse_mapping[1]}", font=ctk.CTkFont(weight="bold"), width=120).grid(row=0, column=2, padx=5, pady=5)
        
        ctk.CTkLabel(table_frame, text=f"Real: {reverse_mapping[0]}", font=ctk.CTkFont(weight="bold"), width=120).grid(row=1, column=0, padx=5, pady=5)
        ctk.CTkLabel(table_frame, text=f"Real: {reverse_mapping[1]}", font=ctk.CTkFont(weight="bold"), width=120).grid(row=2, column=0, padx=5, pady=5)
        
        # Values
        for i in range(2):
            for j in range(2):
                ctk.CTkLabel(
                    table_frame, 
                    text=str(conf_matrix[i][j]),
                    font=ctk.CTkFont(size=14),
                    width=100
                ).grid(row=i+1, column=j+1, padx=5, pady=5)
    
    def display_sigmoid(self):
        """Exibe curva sigmoide"""
        sigmoid_frame = ctk.CTkFrame(self.results_container)
        sigmoid_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            sigmoid_frame,
            text="📊 Curva Sigmoide (Logit vs Probabilidade)",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        sigmoid = self.results['sigmoid']
        
        # Create matplotlib figure
        fig = self.Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        ax.plot(sigmoid['x'], sigmoid['y'], 'b-', linewidth=2)
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold = 0.5')
        ax.set_xlabel('Logit (Linear Predictor)', fontsize=12)
        ax.set_ylabel('Probabilidade P(Y=1)', fontsize=12)
        ax.set_title('Curva Sigmoide', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        
        # Embed plot
        canvas = self.FigureCanvasTkAgg(fig, sigmoid_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    def display_equations(self):
        """Exibe equações de probabilidade"""
        equations_frame = ctk.CTkFrame(self.results_container)
        equations_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            equations_frame,
            text="📝 Equações do Modelo",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        equations = self.results['equations']
        
        # Frame for equations
        eq_frame = ctk.CTkScrollableFrame(equations_frame, height=200)
        eq_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Main equation
        ctk.CTkLabel(
            eq_frame,
            text="Equação Principal",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", pady=(10, 5))
        
        eq_text = ctk.CTkTextbox(eq_frame, height=80, wrap="word")
        eq_text.pack(fill="x", padx=10, pady=5)
        eq_text.insert("1.0", equations['equationMain'])
        eq_text.configure(state="disabled")
        
        # Logit calculation
        ctk.CTkLabel(
            eq_frame,
            text="Cálculo do Logit",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", pady=(20, 5))
        
        logit_text = ctk.CTkTextbox(eq_frame, height=60, wrap="word")
        logit_text.pack(fill="x", padx=10, pady=5)
        logit_text.insert("1.0", equations['equationCalc'])
        logit_text.configure(state="disabled")
        
        # Individual equations
        ctk.CTkLabel(
            eq_frame,
            text="Equações por Variável",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", pady=(20, 5))
        
        for var, eq in equations['equations'].items():
            var_frame = ctk.CTkFrame(eq_frame)
            var_frame.pack(fill="x", padx=10, pady=5)
            
            ctk.CTkLabel(
                var_frame,
                text=f"{var}:",
                font=ctk.CTkFont(weight="bold"),
                width=200
            ).pack(side="left", padx=5)
            
            eq_label = ctk.CTkLabel(
                var_frame,
                text=eq,
                wraplength=700,
                justify="left"
            )
            eq_label.pack(side="left", fill="x", expand=True, padx=5)
