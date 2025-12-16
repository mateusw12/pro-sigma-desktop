"""
Gaussian Process Regression Window
Interface para regressão com Processos Gaussianos
"""
import customtkinter as ctk
from tkinter import messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class GaussianProcessWindow(ctk.CTkToplevel):
    """Janela para análise Gaussian Process"""
    
    def __init__(self, parent, data=None):
        super().__init__(parent)
        
        self.title("Gaussian Process Regression")
        self.geometry("1400x900")
        
        # Maximizar janela
        self.state('zoomed')
        self.lift()
        self.focus_force()
        
        self.data = data
        self.results = None
        
        # Importar aqui para evitar import no início
        import pandas as pd
        import numpy as np
        
        self._setup_ui()
        
        # Carregar dados se fornecidos
        if self.data is not None:
            self._load_from_dataframe(self.data)
    
    def _setup_ui(self):
        """Configura interface do usuário"""
        # Frame principal com scroll
        main_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Título
        title = ctk.CTkLabel(
            main_frame,
            text="Gaussian Process Regression",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=(0, 10))
        
        # Descrição
        desc = ctk.CTkLabel(
            main_frame,
            text="Regressão probabilística com intervalos de confiança - Ideal para dados com ruído e pequenas amostras",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        desc.pack(pady=(0, 20))
        
        # Frame de configuração
        config_frame = ctk.CTkFrame(main_frame)
        config_frame.pack(fill="x", pady=(0, 20))
        
        config_title = ctk.CTkLabel(
            config_frame,
            text="Configurações do Modelo",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        config_title.grid(row=0, column=0, columnspan=4, sticky="w", padx=15, pady=(15, 10))
        
        # Seleção de X
        x_label = ctk.CTkLabel(config_frame, text="Features (X):", anchor="w")
        x_label.grid(row=1, column=0, sticky="w", padx=15, pady=5)
        
        self._create_x_listbox(config_frame)
        
        # Seleção de Y
        y_label = ctk.CTkLabel(config_frame, text="Target (Y):", anchor="w")
        y_label.grid(row=1, column=1, sticky="w", padx=15, pady=5)
        
        self.y_combobox = ctk.CTkComboBox(config_frame, values=[], width=200)
        self.y_combobox.grid(row=2, column=1, sticky="w", padx=15, pady=5)
        
        # Kernel
        kernel_label = ctk.CTkLabel(config_frame, text="Kernel:", anchor="w")
        kernel_label.grid(row=1, column=2, sticky="w", padx=15, pady=5)
        
        from src.analytics.gaussian_process.gaussian_process_utils import get_available_kernels
        kernel_names = list(get_available_kernels().keys())
        
        self.kernel_var = ctk.StringVar(value='RBF')
        self.kernel_combobox = ctk.CTkComboBox(
            config_frame, 
            values=kernel_names,
            variable=self.kernel_var,
            width=200
        )
        self.kernel_combobox.grid(row=2, column=2, sticky="w", padx=15, pady=5)
        
        # Tamanho do teste
        test_size_label = ctk.CTkLabel(config_frame, text="Test Size (%):", anchor="w")
        test_size_label.grid(row=3, column=1, sticky="w", padx=15, pady=5)
        
        self.test_size_var = ctk.IntVar(value=20)
        test_size_spinbox = ctk.CTkEntry(
            config_frame,
            width=100,
            textvariable=self.test_size_var
        )
        test_size_spinbox.grid(row=4, column=1, sticky="w", padx=15, pady=5)
        
        # Normalizar
        self.scale_var = ctk.BooleanVar(value=True)
        scale_check = ctk.CTkCheckBox(
            config_frame,
            text="Normalizar dados",
            variable=self.scale_var
        )
        scale_check.grid(row=3, column=2, sticky="w", padx=15, pady=5)
        
        # Alpha (regularização)
        alpha_label = ctk.CTkLabel(config_frame, text="Alpha (ruído):", anchor="w")
        alpha_label.grid(row=5, column=1, sticky="w", padx=15, pady=5)
        
        self.alpha_var = ctk.StringVar(value="1e-10")
        alpha_entry = ctk.CTkEntry(
            config_frame,
            width=100,
            textvariable=self.alpha_var
        )
        alpha_entry.grid(row=6, column=1, sticky="w", padx=15, pady=5)
        
        alpha_info = ctk.CTkLabel(
            config_frame,
            text="Valores maiores = mais suavização",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        alpha_info.grid(row=6, column=2, sticky="w", padx=15, pady=5)
        
        # Botões de ação
        action_frame = ctk.CTkFrame(main_frame)
        action_frame.pack(fill="x", pady=(0, 20))
        
        train_btn = ctk.CTkButton(
            action_frame,
            text="▶ Treinar Modelo",
            command=self._train_model,
            width=200,
            height=40,
            fg_color="#2E86DE",
            hover_color="#1c5fa8"
        )
        train_btn.pack(side="left", padx=15, pady=15)
        
        compare_btn = ctk.CTkButton(
            action_frame,
            text="📊 Comparar Kernels",
            command=self._compare_kernels,
            width=200,
            height=40
        )
        compare_btn.pack(side="left", padx=5, pady=15)
        
        clear_btn = ctk.CTkButton(
            action_frame,
            text="🗑 Limpar Resultados",
            command=self._clear_results,
            width=200,
            height=40,
            fg_color="#6c757d",
            hover_color="#5a6268"
        )
        clear_btn.pack(side="left", padx=5, pady=15)
        
        # Frame de resultados
        self.results_frame = ctk.CTkFrame(main_frame)
        self.results_frame.pack(fill="both", expand=True)
        
        results_title = ctk.CTkLabel(
            self.results_frame,
            text="Resultados da Análise",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        results_title.pack(pady=15)
    
    def _create_x_listbox(self, parent):
        """Cria listbox para seleção de features"""
        listbox_frame = ctk.CTkFrame(parent)
        listbox_frame.grid(row=2, column=0, sticky="nsew", padx=15, pady=5, rowspan=5)
        
        self.x_scroll_frame = ctk.CTkScrollableFrame(listbox_frame, height=150)
        self.x_scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.x_vars = {}
    
    def _load_from_dataframe(self, df):
        """Carrega dados do DataFrame"""
        # Limpar anterior
        for widget in self.x_scroll_frame.winfo_children():
            widget.destroy()
        
        self.x_vars.clear()
        
        # Colunas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Features X
        for col in numeric_cols:
            var = ctk.BooleanVar()
            check = ctk.CTkCheckBox(
                self.x_scroll_frame,
                text=col,
                variable=var
            )
            check.pack(anchor="w", pady=2)
            self.x_vars[col] = var
        
        # Target Y
        self.y_combobox.configure(values=numeric_cols)
        if numeric_cols:
            self.y_combobox.set(numeric_cols[-1])
    
    def _get_selected_x(self):
        """Retorna lista de features X selecionadas"""
        return [col for col, var in self.x_vars.items() if var.get()]
    
    def _train_model(self):
        """Treina modelo Gaussian Process"""
        x_cols = self._get_selected_x()
        y_col = self.y_combobox.get()
        
        if not x_cols:
            messagebox.showwarning("Features", "Selecione pelo menos 1 feature (X).")
            return
        
        if not y_col:
            messagebox.showwarning("Target", "Selecione a variável target (Y).")
            return
        
        try:
            from src.analytics.gaussian_process.gaussian_process_utils import train_gaussian_process
            
            # Parâmetros
            kernel_name = self.kernel_var.get()
            test_size = self.test_size_var.get() / 100.0
            scale = self.scale_var.get()
            
            try:
                alpha = float(self.alpha_var.get())
            except:
                alpha = 1e-10
            
            # Treinar
            self.results = train_gaussian_process(
                self.data,
                x_cols,
                y_col,
                kernel_name=kernel_name,
                test_size=test_size,
                scale=scale,
                alpha=alpha
            )
            
            # Mostrar resultados
            self._display_results()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao treinar modelo:\n{str(e)}")
    
    def _display_results(self):
        """Mostra resultados do modelo"""
        # Limpar anterior
        for widget in self.results_frame.winfo_children():
            if widget.winfo_class() != 'CTkLabel' or widget.cget('text') != 'Resultados da Análise':
                widget.destroy()
        
        # Métricas
        self._display_metrics()
        
        # Kernel otimizado
        self._display_kernel_info()
        
        # Gráficos
        self._display_plots()
    
    def _display_metrics(self):
        """Mostra métricas do modelo"""
        metrics_frame = ctk.CTkFrame(self.results_frame)
        metrics_frame.pack(fill="x", padx=15, pady=10)
        
        title = ctk.CTkLabel(
            metrics_frame,
            text="Métricas de Performance",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        title.pack(pady=10)
        
        interpretations = self.results['interpretations']
        
        grid_frame = ctk.CTkFrame(metrics_frame)
        grid_frame.pack(padx=15, pady=10)
        
        color_map = {'green': '#28a745', 'blue': '#2E86DE', 'yellow': '#ffc107', 'red': '#dc3545'}
        
        row = 0
        
        # R²
        if 'r2' in interpretations:
            interp = interpretations['r2']
            label = ctk.CTkLabel(
                grid_frame,
                text=f"• {interp['message']}",
                text_color=color_map.get(interp['color'], 'white'),
                font=ctk.CTkFont(size=12)
            )
            label.grid(row=row, column=0, sticky="w", padx=10, pady=5)
            row += 1
        
        # Overfitting
        if 'overfitting' in interpretations:
            interp = interpretations['overfitting']
            label = ctk.CTkLabel(
                grid_frame,
                text=f"• {interp['message']}",
                text_color=color_map.get(interp['color'], 'white'),
                font=ctk.CTkFont(size=12)
            )
            label.grid(row=row, column=0, sticky="w", padx=10, pady=5)
            row += 1
        
        # Cobertura IC
        if 'confidence' in interpretations:
            interp = interpretations['confidence']
            label = ctk.CTkLabel(
                grid_frame,
                text=f"• {interp['message']}",
                text_color=color_map.get(interp['color'], 'white'),
                font=ctk.CTkFont(size=12)
            )
            label.grid(row=row, column=0, sticky="w", padx=10, pady=5)
            row += 1
        
        # Métricas numéricas
        test_metrics = self.results['test_metrics']
        label = ctk.CTkLabel(
            grid_frame,
            text=f"• RMSE (Teste): {test_metrics['rmse']:.4f}  |  MAE: {test_metrics['mae']:.4f}",
            font=ctk.CTkFont(size=12)
        )
        label.grid(row=row, column=0, sticky="w", padx=10, pady=5)
    
    def _display_kernel_info(self):
        """Mostra informações do kernel"""
        kernel_frame = ctk.CTkFrame(self.results_frame)
        kernel_frame.pack(fill="x", padx=15, pady=10)
        
        title = ctk.CTkLabel(
            kernel_frame,
            text="Kernel Otimizado",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        title.pack(pady=10)
        
        kernel_text = self.results['optimized_kernel']
        log_likelihood = self.results['log_marginal_likelihood']
        
        info_label = ctk.CTkLabel(
            kernel_frame,
            text=f"{kernel_text}\n\nLog-Likelihood Marginal: {log_likelihood:.2f}",
            font=ctk.CTkFont(size=11),
            justify="left"
        )
        info_label.pack(padx=15, pady=10)
    
    def _display_plots(self):
        """Mostra gráficos do modelo"""
        plots_frame = ctk.CTkFrame(self.results_frame)
        plots_frame.pack(fill="both", expand=True, padx=15, pady=10)
        
        title = ctk.CTkLabel(
            plots_frame,
            text="Visualizações",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        title.pack(pady=10)
        
        # Número de features
        n_features = len(self.results['x_columns'])
        
        if n_features == 1:
            self._plot_1d_regression(plots_frame)
        else:
            self._plot_multi_dimensional(plots_frame)
    
    def _plot_1d_regression(self, parent):
        """Plota regressão 1D com intervalo de confiança"""
        fig = Figure(figsize=(14, 10))
        
        X_train = self.results['X_train']
        X_test = self.results['X_test']
        y_train = self.results['y_train']
        y_test = self.results['y_test']
        y_train_pred = self.results['y_train_pred']
        y_test_pred = self.results['y_test_pred']
        y_train_std = self.results['y_train_std']
        y_test_std = self.results['y_test_std']
        
        # Plot 1: Fit com intervalos de confiança
        ax1 = fig.add_subplot(221)
        
        # Criar pontos suaves para a curva
        X_all = np.vstack([X_train, X_test])
        X_min, X_max = X_all.min(), X_all.max()
        X_smooth = np.linspace(X_min, X_max, 300).reshape(-1, 1)
        
        # Predizer
        from src.analytics.gaussian_process.gaussian_process_utils import predict_with_uncertainty
        y_smooth, y_smooth_std = predict_with_uncertainty(self.results, X_smooth)
        
        # Plotar
        ax1.plot(X_smooth, y_smooth, 'b-', linewidth=2, label='Predição GP')
        ax1.fill_between(
            X_smooth.ravel(),
            y_smooth - 1.96 * y_smooth_std,
            y_smooth + 1.96 * y_smooth_std,
            alpha=0.3, color='blue', label='IC 95%'
        )
        ax1.fill_between(
            X_smooth.ravel(),
            y_smooth - y_smooth_std,
            y_smooth + y_smooth_std,
            alpha=0.5, color='blue', label='IC 68%'
        )
        ax1.scatter(X_train, y_train, c='green', s=50, alpha=0.6, label='Train', edgecolors='black')
        ax1.scatter(X_test, y_test, c='red', s=50, alpha=0.6, label='Test', edgecolors='black')
        
        ax1.set_xlabel(self.results['x_columns'][0], fontsize=11)
        ax1.set_ylabel(self.results['y_column'], fontsize=11)
        ax1.set_title('Gaussian Process Regression com Intervalos de Confiança', fontsize=12, weight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Predito vs Real
        ax2 = fig.add_subplot(222)
        
        all_y = np.concatenate([y_train, y_test])
        all_pred = np.concatenate([y_train_pred, y_test_pred])
        
        ax2.scatter(y_train, y_train_pred, c='green', s=50, alpha=0.6, label='Train', edgecolors='black')
        ax2.scatter(y_test, y_test_pred, c='red', s=50, alpha=0.6, label='Test', edgecolors='black')
        
        # Linha diagonal
        min_val = min(all_y.min(), all_pred.min())
        max_val = max(all_y.max(), all_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfeito')
        
        ax2.set_xlabel('Valores Reais', fontsize=11)
        ax2.set_ylabel('Valores Preditos', fontsize=11)
        ax2.set_title('Predito vs Real', fontsize=12, weight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Resíduos
        ax3 = fig.add_subplot(223)
        
        residuals_train = y_train - y_train_pred
        residuals_test = y_test - y_test_pred
        
        ax3.scatter(y_train_pred, residuals_train, c='green', s=50, alpha=0.6, label='Train', edgecolors='black')
        ax3.scatter(y_test_pred, residuals_test, c='red', s=50, alpha=0.6, label='Test', edgecolors='black')
        ax3.axhline(y=0, color='k', linestyle='--', linewidth=2)
        
        ax3.set_xlabel('Valores Preditos', fontsize=11)
        ax3.set_ylabel('Resíduos', fontsize=11)
        ax3.set_title('Análise de Resíduos', fontsize=12, weight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Incerteza vs X
        ax4 = fig.add_subplot(224)
        
        ax4.scatter(X_train, y_train_std, c='green', s=50, alpha=0.6, label='Train', edgecolors='black')
        ax4.scatter(X_test, y_test_std, c='red', s=50, alpha=0.6, label='Test', edgecolors='black')
        ax4.plot(X_smooth, y_smooth_std, 'b-', linewidth=2, label='Incerteza GP')
        
        ax4.set_xlabel(self.results['x_columns'][0], fontsize=11)
        ax4.set_ylabel('Desvio Padrão (Incerteza)', fontsize=11)
        ax4.set_title('Incerteza das Predições', fontsize=12, weight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def _plot_multi_dimensional(self, parent):
        """Plota análise multidimensional"""
        fig = Figure(figsize=(14, 10))
        
        y_train = self.results['y_train']
        y_test = self.results['y_test']
        y_train_pred = self.results['y_train_pred']
        y_test_pred = self.results['y_test_pred']
        y_train_std = self.results['y_train_std']
        y_test_std = self.results['y_test_std']
        
        # Plot 1: Predito vs Real
        ax1 = fig.add_subplot(221)
        
        all_y = np.concatenate([y_train, y_test])
        all_pred = np.concatenate([y_train_pred, y_test_pred])
        
        ax1.scatter(y_train, y_train_pred, c='green', s=50, alpha=0.6, label='Train', edgecolors='black')
        ax1.scatter(y_test, y_test_pred, c='red', s=50, alpha=0.6, label='Test', edgecolors='black')
        
        min_val = min(all_y.min(), all_pred.min())
        max_val = max(all_y.max(), all_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfeito')
        
        ax1.set_xlabel('Valores Reais', fontsize=11)
        ax1.set_ylabel('Valores Preditos', fontsize=11)
        ax1.set_title('Predito vs Real', fontsize=12, weight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Resíduos
        ax2 = fig.add_subplot(222)
        
        residuals_train = y_train - y_train_pred
        residuals_test = y_test - y_test_pred
        
        ax2.scatter(y_train_pred, residuals_train, c='green', s=50, alpha=0.6, label='Train', edgecolors='black')
        ax2.scatter(y_test_pred, residuals_test, c='red', s=50, alpha=0.6, label='Test', edgecolors='black')
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=2)
        
        ax2.set_xlabel('Valores Preditos', fontsize=11)
        ax2.set_ylabel('Resíduos', fontsize=11)
        ax2.set_title('Análise de Resíduos', fontsize=12, weight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Incerteza
        ax3 = fig.add_subplot(223)
        
        all_std = np.concatenate([y_train_std, y_test_std])
        indices = np.arange(len(all_std))
        
        train_indices = np.arange(len(y_train_std))
        test_indices = np.arange(len(y_train_std), len(all_std))
        
        ax3.scatter(train_indices, y_train_std, c='green', s=50, alpha=0.6, label='Train', edgecolors='black')
        ax3.scatter(test_indices, y_test_std, c='red', s=50, alpha=0.6, label='Test', edgecolors='black')
        ax3.axhline(y=np.mean(all_std), color='blue', linestyle='--', linewidth=2, label='Média')
        
        ax3.set_xlabel('Índice da Amostra', fontsize=11)
        ax3.set_ylabel('Desvio Padrão (Incerteza)', fontsize=11)
        ax3.set_title('Incerteza das Predições', fontsize=12, weight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Predições com barras de erro
        ax4 = fig.add_subplot(224)
        
        # Mostrar apenas alguns pontos para não poluir
        n_show = min(50, len(y_test))
        indices_show = np.linspace(0, len(y_test)-1, n_show, dtype=int)
        
        x_pos = np.arange(n_show)
        ax4.errorbar(
            x_pos, y_test_pred[indices_show], 
            yerr=1.96*y_test_std[indices_show],
            fmt='o', color='blue', ecolor='lightblue', 
            elinewidth=2, capsize=3, alpha=0.7, label='Predição ± IC 95%'
        )
        ax4.scatter(x_pos, y_test[indices_show], c='red', s=30, marker='x', label='Real', zorder=5)
        
        ax4.set_xlabel('Amostra (Teste)', fontsize=11)
        ax4.set_ylabel(self.results['y_column'], fontsize=11)
        ax4.set_title('Predições com Intervalos de Confiança', fontsize=12, weight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def _compare_kernels(self):
        """Compara diferentes kernels"""
        x_cols = self._get_selected_x()
        y_col = self.y_combobox.get()
        
        if not x_cols or not y_col:
            messagebox.showwarning("Dados", "Selecione features e target primeiro.")
            return
        
        try:
            from src.analytics.gaussian_process.gaussian_process_utils import train_gaussian_process, get_available_kernels
            
            kernel_names = list(get_available_kernels().keys())
            results_comparison = []
            
            # Treinar com cada kernel
            for kernel_name in kernel_names:
                try:
                    result = train_gaussian_process(
                        self.data,
                        x_cols,
                        y_col,
                        kernel_name=kernel_name,
                        test_size=self.test_size_var.get() / 100.0,
                        scale=self.scale_var.get(),
                        alpha=float(self.alpha_var.get())
                    )
                    
                    results_comparison.append({
                        'kernel': kernel_name,
                        'r2_test': result['test_metrics']['r2'],
                        'rmse_test': result['test_metrics']['rmse'],
                        'log_likelihood': result['log_marginal_likelihood']
                    })
                except:
                    pass
            
            # Mostrar comparação
            self._display_kernel_comparison(results_comparison)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao comparar kernels:\n{str(e)}")
    
    def _display_kernel_comparison(self, results):
        """Mostra comparação de kernels"""
        # Limpar anterior
        for widget in self.results_frame.winfo_children():
            if widget.winfo_class() != 'CTkLabel' or widget.cget('text') != 'Resultados da Análise':
                widget.destroy()
        
        comp_frame = ctk.CTkFrame(self.results_frame)
        comp_frame.pack(fill="both", expand=True, padx=15, pady=15)
        
        title = ctk.CTkLabel(
            comp_frame,
            text="Comparação de Kernels",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        title.pack(pady=10)
        
        # Criar tabela
        table_frame = ctk.CTkFrame(comp_frame)
        table_frame.pack(padx=15, pady=10)
        
        headers = ['Kernel', 'R² (Teste)', 'RMSE (Teste)', 'Log-Likelihood']
        for col_idx, header in enumerate(headers):
            label = ctk.CTkLabel(
                table_frame,
                text=header,
                font=ctk.CTkFont(size=11, weight="bold"),
                fg_color="#1f538d",
                corner_radius=0,
                width=150,
                height=30
            )
            label.grid(row=0, column=col_idx, sticky="ew", padx=1, pady=1)
        
        # Ordenar por R²
        results_sorted = sorted(results, key=lambda x: x['r2_test'], reverse=True)
        
        for row_idx, result in enumerate(results_sorted):
            # Kernel
            label = ctk.CTkLabel(
                table_frame,
                text=result['kernel'],
                font=ctk.CTkFont(size=10),
                fg_color="#2b2b2b",
                corner_radius=0,
                width=150,
                height=25
            )
            label.grid(row=row_idx + 1, column=0, sticky="ew", padx=1, pady=1)
            
            # R²
            label = ctk.CTkLabel(
                table_frame,
                text=f"{result['r2_test']:.4f}",
                font=ctk.CTkFont(size=10),
                fg_color="#2b2b2b",
                corner_radius=0,
                width=150,
                height=25
            )
            label.grid(row=row_idx + 1, column=1, sticky="ew", padx=1, pady=1)
            
            # RMSE
            label = ctk.CTkLabel(
                table_frame,
                text=f"{result['rmse_test']:.4f}",
                font=ctk.CTkFont(size=10),
                fg_color="#2b2b2b",
                corner_radius=0,
                width=150,
                height=25
            )
            label.grid(row=row_idx + 1, column=2, sticky="ew", padx=1, pady=1)
            
            # Log-Likelihood
            label = ctk.CTkLabel(
                table_frame,
                text=f"{result['log_likelihood']:.2f}",
                font=ctk.CTkFont(size=10),
                fg_color="#2b2b2b",
                corner_radius=0,
                width=150,
                height=25
            )
            label.grid(row=row_idx + 1, column=3, sticky="ew", padx=1, pady=1)
        
        # Recomendação
        best = results_sorted[0]
        rec_label = ctk.CTkLabel(
            comp_frame,
            text=f"💡 Melhor kernel: {best['kernel']} (R² = {best['r2_test']:.4f})",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#28a745"
        )
        rec_label.pack(pady=10)
    
    def _clear_results(self):
        """Limpa resultados"""
        for widget in self.results_frame.winfo_children():
            if widget.winfo_class() != 'CTkLabel' or widget.cget('text') != 'Resultados da Análise':
                widget.destroy()
        
        self.results = None
