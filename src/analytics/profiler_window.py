"""
Profiler Window - Interactive prediction profiler
Similar ao JMP Profiler
"""
import customtkinter as ctk
from tkinter import messagebox
import statsmodels.api as sm
from typing import Dict
from src.utils.lazy_imports import (
    get_numpy, get_pandas, get_matplotlib, 
    get_matplotlib_figure, get_matplotlib_backend
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


class ProfilerWindow(ctk.CTkToplevel):
    def __init__(
        self, 
        parent,
        model_data: Dict,
        title: str = "Profiler Interativo"
    ):
        """
        Profiler interativo para análise de sensibilidade
        
        Args:
            parent: Janela pai
            model_data: Dict contendo:
                - 'model': modelo ajustado (statsmodels)
                - 'X': DataFrame com variáveis independentes
                - 'y': Series com variável dependente
                - 'x_cols': lista de nomes das colunas X
                - 'y_col': nome da coluna Y
                - 'model_type': 'linear', 'logistic', etc.
            title: Título da janela
        """
        super().__init__(parent)
        
        # Carrega bibliotecas
        pd, np, plt, Figure, FigureCanvasTkAgg = _ensure_libs()
        self.pd = pd
        self.np = np
        self.plt = plt
        self.Figure = Figure
        self.FigureCanvasTkAgg = FigureCanvasTkAgg
        
        # Store model data
        self.model = model_data['model']
        self.X = model_data['X']
        self.y = model_data['y']
        self.x_cols = model_data['x_cols']
        self.y_col = model_data['y_col']
        self.model_type = model_data.get('model_type', 'linear')
        
        # Separate main factors from interactions
        # Interactions typically have "×" or "*" in their names
        self.main_factors = [col for col in self.x_cols if '×' not in col and '*' not in col]
        self.interactions = [col for col in self.x_cols if '×' in col or '*' in col]
        
        # Current values for each factor (initialize to mean)
        self.current_values = {}
        for col in self.x_cols:
            self.current_values[col] = float(self.X[col].mean())
        
        # Sliders and value labels
        self.sliders = {}
        self.value_labels = {}
        
        # Calculate RMSE for confidence intervals (use provided or calculate)
        self.rmse = model_data.get('rmse', None)
        if self.rmse is None:
            self.rmse = self._calculate_rmse()
        
        # Optimization settings for large datasets
        self.n_plot_points = 30 if len(self.X) > 1000 else 50
        
        # Calculate global Y range for consistent scales
        self.y_min_global, self.y_max_global = self._calculate_global_y_range()
        
        # Window configuration
        self.title(title)
        self.resizable(True, True)
        self.minsize(1200, 800)
        self.state('zoomed')
        
        # Make modal
        self.transient(parent)
        self.grab_set()
        
        self.create_widgets()
        self.update_prediction()
    
    def _calculate_rmse(self):
        """Calcula RMSE do modelo para intervalo de confiança"""
        try:
            # Predict on training data
            X_const = sm.add_constant(self.X[self.x_cols], has_constant='add')
            y_pred = self.model.predict(X_const)
            
            # Calculate RMSE: sqrt(MSE)
            residuals = self.y - y_pred
            mse = self.np.mean(residuals ** 2)
            rmse = mse ** 0.5
            return float(rmse)
        except:
            return 0.0
    
    def _calculate_global_y_range(self):
        """Calcula range global de Y considerando todas as variações de todos os fatores"""
        all_predictions = []
        
        # Only vary main factors, not interactions
        for col in self.main_factors:
            min_val = float(self.X[col].min())
            max_val = float(self.X[col].max())
            x_range = self.np.linspace(min_val, max_val, self.n_plot_points)
            
            for x_val in x_range:
                pred_input = self.current_values.copy()
                pred_input[col] = x_val
                
                # Update interaction terms if any
                self._update_interactions(pred_input)
                
                pred_df = self.pd.DataFrame([pred_input])[self.x_cols]
                pred_df_const = sm.add_constant(pred_df, has_constant='add')
                
                if self.model_type == 'logistic':
                    y_pred = self.model.predict(pred_df_const)[0]
                else:
                    y_pred = self.model.predict(pred_df_const)[0]
                
                all_predictions.append(y_pred)
        
        y_min = min(all_predictions)
        y_max = max(all_predictions)
        
        # Add 10% margin
        margin = (y_max - y_min) * 0.1
        return y_min - margin, y_max + margin
    
    def _update_interactions(self, values_dict: dict):
        """Atualiza valores dos termos de interação com base nos fatores principais"""
        for interaction in self.interactions:
            # Parse interaction name (e.g., "A × B" or "A*B")
            if '×' in interaction:
                factors = interaction.split(' × ')
            elif '*' in interaction:
                factors = interaction.split('*')
            else:
                continue
            
            # Calculate interaction value
            if len(factors) == 2:
                factor1, factor2 = factors[0].strip(), factors[1].strip()
                if factor1 in values_dict and factor2 in values_dict:
                    values_dict[interaction] = values_dict[factor1] * values_dict[factor2]
    
    def create_widgets(self):
        """Cria interface"""
        # Main container
        self.main_container = ctk.CTkFrame(self)
        self.main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title = ctk.CTkLabel(
            self.main_container,
            text=f"📊 Profiler Interativo - {self.y_col}",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=(0, 20))
        
        # Top section: Prediction display
        self.create_prediction_display()
        
        # Middle section: Factor controls and plots
        self.create_factor_section()
        
        # Bottom buttons
        self.create_buttons()
    
    def create_prediction_display(self):
        """Cria display da predição atual"""
        pred_frame = ctk.CTkFrame(self.main_container)
        pred_frame.pack(fill="x", pady=(0, 20))
        
        # Title
        ctk.CTkLabel(
            pred_frame,
            text="🎯 Predição Atual",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(10, 5))
        
        # Prediction value
        self.pred_label = ctk.CTkLabel(
            pred_frame,
            text="",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color="#2563eb"
        )
        self.pred_label.pack(pady=10)
        
        # Confidence intervals
        self.ci_label = ctk.CTkLabel(
            pred_frame,
            text="",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.ci_label.pack(pady=(0, 10))
    
    def create_factor_section(self):
        """Cria seção de fatores com sliders e gráficos"""
        factor_frame = ctk.CTkFrame(self.main_container)
        factor_frame.pack(fill="both", expand=True, pady=(0, 20))
        
        # Scrollable frame for factors
        scroll_frame = ctk.CTkScrollableFrame(factor_frame)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create row for each MAIN factor only (no interactions)
        n_factors = len(self.main_factors)
        
        # Adjust columns based on number of factors
        if n_factors == 1:
            n_cols = 1  # Single column for 1 factor (centered)
        elif n_factors == 2:
            n_cols = 2  # Two columns for 2 factors
        else:
            n_cols = min(3, n_factors)  # Max 3 columns for 3+ factors
        
        n_rows = int(self.np.ceil(n_factors / n_cols))
        
        # Iterate only over main factors (not interactions)
        for idx, col in enumerate(self.main_factors):
            row = idx // n_cols
            col_idx = idx % n_cols
            
            self.create_factor_control(scroll_frame, col, row, col_idx, n_cols)
    
    def create_factor_control(self, parent, col_name: str, row: int, col: int, max_cols: int):
        """Cria controle de um fator individual"""
        # Container for this factor
        factor_container = ctk.CTkFrame(parent, width=400 if max_cols == 1 else 300)
        factor_container.grid(row=row, column=col, padx=20, pady=10, sticky="nsew")
        
        # Configure grid weight
        parent.grid_columnconfigure(col, weight=1)
        
        # Factor name
        ctk.CTkLabel(
            factor_container,
            text=col_name,
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        # Mini plot showing effect (moved to top)
        self.create_factor_plot(factor_container, col_name)
        
        # Get min/max values
        min_val = float(self.X[col_name].min())
        max_val = float(self.X[col_name].max())
        
        # Value input section - now below plot
        input_section = ctk.CTkFrame(factor_container, fg_color="transparent")
        input_section.pack(fill="x", padx=15, pady=(10, 5))
        
        # Label and Entry in same line
        ctk.CTkLabel(
            input_section,
            text="Valor:",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", pady=(0, 5))
        
        value_entry = ctk.CTkEntry(
            input_section,
            width=200 if max_cols == 1 else 180,
            placeholder_text=f"{self.current_values[col_name]:.4f}"
        )
        value_entry.insert(0, f"{self.current_values[col_name]:.4f}")
        value_entry.pack(anchor="w", pady=(0, 10))
        
        # Bind entry change
        def on_entry_change(event):
            try:
                val = float(value_entry.get())
                if min_val <= val <= max_val:
                    self.current_values[col_name] = val
                    self.sliders[col_name].set(val)
                    self.update_prediction()
                    self.update_plots()
                else:
                    messagebox.showwarning("Aviso", f"Valor deve estar entre {min_val:.4f} e {max_val:.4f}")
                    value_entry.delete(0, "end")
                    value_entry.insert(0, f"{self.current_values[col_name]:.4f}")
            except ValueError:
                messagebox.showerror("Erro", "Digite um valor numérico válido")
                value_entry.delete(0, "end")
                value_entry.insert(0, f"{self.current_values[col_name]:.4f}")
        
        value_entry.bind("<Return>", on_entry_change)
        value_entry.bind("<FocusOut>", on_entry_change)
        
        # Store entry reference
        if not hasattr(self, 'value_entries'):
            self.value_entries = {}
        self.value_entries[col_name] = value_entry
        
        # Slider section
        slider_section = ctk.CTkFrame(factor_container, fg_color="transparent")
        slider_section.pack(fill="x", padx=15, pady=5)
        
        ctk.CTkLabel(
            slider_section,
            text="Ajuste:",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", pady=(0, 5))
        
        slider = ctk.CTkSlider(
            slider_section,
            from_=min_val,
            to=max_val,
            command=lambda val, c=col_name: self.on_slider_change(c, val),
            width=250 if max_cols == 1 else 200
        )
        slider.set(self.current_values[col_name])
        slider.pack(anchor="w", pady=(0, 10))
        self.sliders[col_name] = slider
        
        # Range labels section
        range_section = ctk.CTkFrame(factor_container, fg_color="transparent")
        range_section.pack(fill="x", padx=15, pady=(0, 10))
        
        ctk.CTkLabel(
            range_section,
            text="Limites:",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", pady=(0, 5))
        
        range_frame = ctk.CTkFrame(range_section, fg_color="transparent")
        range_frame.pack(fill="x")
        
        ctk.CTkLabel(
            range_frame,
            text=f"Min: {min_val:.3f}",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        ).pack(side="left")
        
        ctk.CTkLabel(
            range_frame,
            text=f"Max: {max_val:.3f}",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        ).pack(side="right")
    
    def create_factor_plot(self, parent, col_name: str):
        """Cria mini gráfico mostrando efeito do fator"""
        # Create figure with better size
        fig = self.Figure(figsize=(4, 2.5), dpi=80)
        ax = fig.add_subplot(111)
        
        # Get range for this factor (optimized for large datasets)
        min_val = float(self.X[col_name].min())
        max_val = float(self.X[col_name].max())
        x_range = self.np.linspace(min_val, max_val, self.n_plot_points)
        
        # Vectorized prediction for performance
        predictions = []
        for x_val in x_range:
            pred_input = self.current_values.copy()
            pred_input[col_name] = x_val
            self._update_interactions(pred_input)
            
            pred_df = self.pd.DataFrame([pred_input])[self.x_cols]
            pred_df_const = sm.add_constant(pred_df, has_constant='add')
            
            if self.model_type == 'logistic':
                y_pred = self.model.predict(pred_df_const)[0]
            else:
                y_pred = self.model.predict(pred_df_const)[0]
            
            predictions.append(y_pred)
        
        predictions = self.np.array(predictions)
        
        # Plot confidence band (RMSE)
        ax.fill_between(x_range, predictions - self.rmse, predictions + self.rmse, 
                        alpha=0.2, color='#2563eb', label='IC (RMSE)')
        
        # Plot prediction line
        ax.plot(x_range, predictions, 'b-', linewidth=2.5, color='#2563eb', label='Predição')
        ax.axvline(self.current_values[col_name], color='#ef4444', linestyle='--', 
                  linewidth=2, alpha=0.8, label='Valor Atual')
        
        ax.set_xlabel(col_name, fontsize=10, fontweight='bold')
        ax.set_ylabel(self.y_col, fontsize=10, fontweight='bold')
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Apply global Y limits for consistent scale across all plots
        ax.set_ylim(self.y_min_global, self.y_max_global)
        
        # Add background color to make it stand out
        ax.set_facecolor('#f8f9fa')
        
        fig.tight_layout()
        
        # Embed in tkinter with padding
        canvas = self.FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=(5, 10), padx=10)
        
        # Store for updates
        if not hasattr(self, 'factor_plots'):
            self.factor_plots = {}
        self.factor_plots[col_name] = (fig, ax, canvas)
    
    def on_slider_change(self, col_name: str, value: float):
        """Callback quando slider muda"""
        self.current_values[col_name] = float(value)
        
        # Update interaction terms
        self._update_interactions(self.current_values)
        
        # Update entry field
        if hasattr(self, 'value_entries') and col_name in self.value_entries:
            entry = self.value_entries[col_name]
            entry.delete(0, "end")
            entry.insert(0, f"{value:.4f}")
        self.update_prediction()
        self.update_plots()
    
    def update_prediction(self):
        """Atualiza predição com valores atuais"""
        try:
            # Create prediction DataFrame
            pred_df = self.pd.DataFrame([self.current_values])[self.x_cols]
            
            # Add constant for statsmodels (if needed)
            pred_df_const = sm.add_constant(pred_df, has_constant='add')
            
            # Predict
            if self.model_type == 'logistic':
                y_pred = self.model.predict(pred_df_const)[0]
                pred_text = f"{y_pred:.4f}"
            else:
                y_pred = self.model.predict(pred_df_const)[0]
                pred_text = f"{y_pred:.4f}"
            
            self.pred_label.configure(text=pred_text)
            
            # Calculate confidence intervals using RMSE
            ci_lower = y_pred - self.rmse
            ci_upper = y_pred + self.rmse
            
            self.ci_label.configure(
                text=f"IC (RMSE): [{ci_lower:.4f}, {ci_upper:.4f}]"
            )
                
        except Exception as e:
            print(f"Error updating prediction: {e}")
            import traceback
            traceback.print_exc()
            self.pred_label.configure(text="Erro")
    
    def update_plots(self):
        """Atualiza todos os mini gráficos (otimizado)"""
        for col_name, (fig, ax, canvas) in self.factor_plots.items():
            # Clear and redraw
            ax.clear()
            
            # Get range (optimized for large datasets)
            min_val = float(self.X[col_name].min())
            max_val = float(self.X[col_name].max())
            x_range = self.np.linspace(min_val, max_val, self.n_plot_points)
            
            # Vectorized prediction
            predictions = []
            for x_val in x_range:
                pred_input = self.current_values.copy()
                pred_input[col_name] = x_val
                self._update_interactions(pred_input)
                
                pred_df = self.pd.DataFrame([pred_input])[self.x_cols]
                pred_df_const = sm.add_constant(pred_df, has_constant='add')
                
                if self.model_type == 'logistic':
                    y_pred = self.model.predict(pred_df_const)[0]
                else:
                    y_pred = self.model.predict(pred_df_const)[0]
                
                predictions.append(y_pred)
            
            predictions = self.np.array(predictions)
            
            # Plot confidence band (RMSE)
            ax.fill_between(x_range, predictions - self.rmse, predictions + self.rmse, 
                           alpha=0.2, color='#2563eb')
            
            # Plot prediction line
            ax.plot(x_range, predictions, 'b-', linewidth=2.5, color='#2563eb')
            ax.axvline(self.current_values[col_name], color='#ef4444', linestyle='--', 
                      linewidth=2, alpha=0.8)
            
            ax.set_xlabel(col_name, fontsize=10, fontweight='bold')
            ax.set_ylabel(self.y_col, fontsize=10, fontweight='bold')
            ax.tick_params(labelsize=9)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Apply global Y limits for consistent scale
            ax.set_ylim(self.y_min_global, self.y_max_global)
            
            # Add background color
            ax.set_facecolor('#f8f9fa')
            
            canvas.draw()
    
    def create_buttons(self):
        """Cria botões de ação"""
        button_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        button_frame.pack(fill="x", pady=10)
        
        # Reset button
        reset_btn = ctk.CTkButton(
            button_frame,
            text="🔄 Reset para Médias",
            command=self.reset_to_means,
            height=40,
            font=ctk.CTkFont(size=14),
            fg_color="#3498DB"
        )
        reset_btn.pack(side="left", padx=10)
        
        # Close button
        close_btn = ctk.CTkButton(
            button_frame,
            text="Fechar",
            command=self.destroy,
            height=40,
            font=ctk.CTkFont(size=14),
            fg_color="#95A5A6"
        )
        close_btn.pack(side="left", padx=10)
    
    def reset_to_means(self):
        """Reset todos os valores para as médias"""
        # Reset main factors
        for col in self.main_factors:
            mean_val = float(self.X[col].mean())
            self.current_values[col] = mean_val
            self.sliders[col].set(mean_val)
            # Update entry field
            if hasattr(self, 'value_entries') and col in self.value_entries:
                entry = self.value_entries[col]
                entry.delete(0, "end")
                entry.insert(0, f"{mean_val:.4f}")
        
        # Update interactions
        self._update_interactions(self.current_values)
        
        self.update_prediction()
        self.update_plots()
