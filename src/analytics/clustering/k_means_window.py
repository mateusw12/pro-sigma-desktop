"""
K-Means Clustering Window
Interface para análise de agrupamento K-Means
"""
import customtkinter as ctk
from tkinter import messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class KMeansWindow(ctk.CTkToplevel):
    """Janela para análise K-Means"""
    
    def __init__(self, parent, data=None):
        super().__init__(parent)
        
        self.title("K-Means Clustering")
        self.geometry("1400x900")
        
        # Maximizar janela
        self.state('zoomed')
        self.lift()
        self.focus_force()
        
        self.data = data
        self.results = None
        self.elbow_data = None
        
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
            text="K-Means Clustering",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=(0, 10))
        
        # Descrição
        desc = ctk.CTkLabel(
            main_frame,
            text="Agrupamento não supervisionado - Identifica padrões e agrupa observações similares",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        desc.pack(pady=(0, 20))
        
        # Frame de configuração
        config_frame = ctk.CTkFrame(main_frame)
        config_frame.pack(fill="x", pady=(0, 20))
        
        config_title = ctk.CTkLabel(
            config_frame,
            text="Configurações",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        config_title.grid(row=0, column=0, columnspan=3, sticky="w", padx=15, pady=(15, 10))
        
        # Seleção de features
        features_label = ctk.CTkLabel(config_frame, text="Features para Clustering:", anchor="w")
        features_label.grid(row=1, column=0, sticky="w", padx=15, pady=5)
        
        self.features_listbox = None
        self._create_features_listbox(config_frame)
        
        # Número de clusters
        clusters_label = ctk.CTkLabel(config_frame, text="Número de Clusters (K):", anchor="w")
        clusters_label.grid(row=1, column=1, sticky="w", padx=15, pady=5)
        
        self.n_clusters_var = ctk.IntVar(value=3)
        self.n_clusters_spinbox = ctk.CTkEntry(
            config_frame,
            width=100,
            textvariable=self.n_clusters_var
        )
        self.n_clusters_spinbox.grid(row=2, column=1, sticky="w", padx=15, pady=5)
        
        # Normalizar dados
        self.scale_var = ctk.BooleanVar(value=True)
        scale_check = ctk.CTkCheckBox(
            config_frame,
            text="Normalizar dados (recomendado)",
            variable=self.scale_var
        )
        scale_check.grid(row=3, column=1, sticky="w", padx=15, pady=10)
        
        # Botões de ação
        action_frame = ctk.CTkFrame(main_frame)
        action_frame.pack(fill="x", pady=(0, 20))
        
        elbow_btn = ctk.CTkButton(
            action_frame,
            text="📊 Método do Cotovelo",
            command=self._show_elbow_method,
            width=200,
            height=40
        )
        elbow_btn.pack(side="left", padx=15, pady=15)
        
        analyze_btn = ctk.CTkButton(
            action_frame,
            text="▶ Executar Clustering",
            command=self._analyze_kmeans,
            width=200,
            height=40,
            fg_color="#2E86DE",
            hover_color="#1c5fa8"
        )
        analyze_btn.pack(side="left", padx=5, pady=15)
        
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
    
    def _create_features_listbox(self, parent):
        """Cria listbox para seleção de features"""
        listbox_frame = ctk.CTkFrame(parent)
        listbox_frame.grid(row=2, column=0, sticky="nsew", padx=15, pady=5, rowspan=2)
        
        # Frame interno com scroll
        self.features_scroll_frame = ctk.CTkScrollableFrame(listbox_frame, height=150)
        self.features_scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.feature_vars = {}
    
    def _load_from_dataframe(self, df):
        """Carrega dados do DataFrame"""
        # Limpar features anteriores
        for widget in self.features_scroll_frame.winfo_children():
            widget.destroy()
        
        self.feature_vars.clear()
        
        # Adicionar apenas colunas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            var = ctk.BooleanVar()
            check = ctk.CTkCheckBox(
                self.features_scroll_frame,
                text=col,
                variable=var
            )
            check.pack(anchor="w", pady=2)
            self.feature_vars[col] = var
    
    def _get_selected_features(self):
        """Retorna lista de features selecionadas"""
        return [col for col, var in self.feature_vars.items() if var.get()]
    
    def _show_elbow_method(self):
        """Mostra método do cotovelo para sugerir K ótimo"""
        features = self._get_selected_features()
        
        if len(features) < 2:
            messagebox.showwarning(
                "Features Insuficientes",
                "Selecione pelo menos 2 features para o clustering."
            )
            return
        
        try:
            from src.analytics.clustering.k_means_utils import calculate_elbow_method, suggest_optimal_k
            
            # Preparar dados
            X = self.data[features].values
            
            # Normalizar
            if self.scale_var.get():
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
            
            # Calcular método do cotovelo
            self.elbow_data = calculate_elbow_method(X, max_k=10)
            
            # Sugerir K ótimo
            suggested_k = suggest_optimal_k(self.elbow_data)
            
            # Mostrar gráficos
            self._plot_elbow_method(suggested_k)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao calcular método do cotovelo:\n{str(e)}")
    
    def _plot_elbow_method(self, suggested_k):
        """Plota gráficos do método do cotovelo"""
        # Limpar resultados anteriores
        for widget in self.results_frame.winfo_children():
            if widget.winfo_class() != 'CTkLabel' or widget.cget('text') != 'Resultados da Análise':
                widget.destroy()
        
        # Frame para gráficos
        plots_frame = ctk.CTkFrame(self.results_frame)
        plots_frame.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Criar figura com 2 subplots
        fig = Figure(figsize=(14, 5))
        
        # Plot 1: Inércia (Elbow)
        ax1 = fig.add_subplot(121)
        ax1.plot(self.elbow_data['k_values'], self.elbow_data['inertias'], 'bo-', linewidth=2, markersize=8)
        ax1.axvline(x=suggested_k, color='red', linestyle='--', label=f'K sugerido = {suggested_k}')
        ax1.set_xlabel('Número de Clusters (K)', fontsize=12)
        ax1.set_ylabel('Inércia (Within-Cluster Sum of Squares)', fontsize=12)
        ax1.set_title('Método do Cotovelo', fontsize=14, weight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Silhouette Score
        ax2 = fig.add_subplot(122)
        ax2.plot(self.elbow_data['k_values'], self.elbow_data['silhouette_scores'], 'go-', linewidth=2, markersize=8)
        ax2.axvline(x=suggested_k, color='red', linestyle='--', label=f'K sugerido = {suggested_k}')
        ax2.set_xlabel('Número de Clusters (K)', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Qualidade dos Clusters', fontsize=14, weight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        fig.tight_layout()
        
        # Adicionar canvas
        canvas = FigureCanvasTkAgg(fig, plots_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Mensagem com sugestão
        suggestion_label = ctk.CTkLabel(
            self.results_frame,
            text=f"💡 K sugerido: {suggested_k} clusters (baseado no Silhouette Score)",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#2E86DE"
        )
        suggestion_label.pack(pady=10)
        
        # Atualizar spinbox
        self.n_clusters_var.set(suggested_k)
    
    def _analyze_kmeans(self):
        """Executa análise K-Means"""
        features = self._get_selected_features()
        
        if len(features) < 2:
            messagebox.showwarning(
                "Features Insuficientes",
                "Selecione pelo menos 2 features para o clustering."
            )
            return
        
        n_clusters = self.n_clusters_var.get()
        
        if n_clusters < 2:
            messagebox.showwarning("Valor Inválido", "O número de clusters deve ser pelo menos 2.")
            return
        
        if n_clusters >= len(self.data):
            messagebox.showwarning(
                "Valor Inválido",
                f"O número de clusters ({n_clusters}) não pode ser maior ou igual ao número de observações ({len(self.data)})."
            )
            return
        
        try:
            from src.analytics.clustering.k_means_utils import perform_kmeans
            
            # Executar clustering
            self.results = perform_kmeans(
                self.data,
                features,
                n_clusters=n_clusters,
                scale=self.scale_var.get()
            )
            
            # Mostrar resultados
            self._display_results(features)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao executar K-Means:\n{str(e)}")
    
    def _display_results(self, features):
        """Mostra resultados do clustering"""
        # Limpar resultados anteriores
        for widget in self.results_frame.winfo_children():
            if widget.winfo_class() != 'CTkLabel' or widget.cget('text') != 'Resultados da Análise':
                widget.destroy()
        
        # Métricas
        self._display_metrics()
        
        # Centroides
        self._display_centroids()
        
        # Visualizações
        self._display_visualizations(features)
    
    def _display_metrics(self):
        """Mostra métricas de qualidade"""
        metrics_frame = ctk.CTkFrame(self.results_frame)
        metrics_frame.pack(fill="x", padx=15, pady=10)
        
        title = ctk.CTkLabel(
            metrics_frame,
            text="Métricas de Qualidade",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        title.pack(pady=10)
        
        metrics = self.results['metrics']
        interpretations = self.results['interpretations']
        
        # Criar grid de métricas
        grid_frame = ctk.CTkFrame(metrics_frame)
        grid_frame.pack(padx=15, pady=10)
        
        row = 0
        
        # Silhouette Score
        if 'silhouette' in interpretations:
            interp = interpretations['silhouette']
            color_map = {'green': '#28a745', 'blue': '#2E86DE', 'yellow': '#ffc107', 'red': '#dc3545'}
            
            label = ctk.CTkLabel(
                grid_frame,
                text=f"• Silhouette Score: {interp['message']}",
                text_color=color_map.get(interp['color'], 'white'),
                font=ctk.CTkFont(size=12)
            )
            label.grid(row=row, column=0, sticky="w", padx=10, pady=5)
            row += 1
        
        # Davies-Bouldin
        if 'davies_bouldin' in interpretations:
            interp = interpretations['davies_bouldin']
            color_map = {'green': '#28a745', 'blue': '#2E86DE', 'yellow': '#ffc107', 'red': '#dc3545'}
            
            label = ctk.CTkLabel(
                grid_frame,
                text=f"• Davies-Bouldin Index: {interp['message']}",
                text_color=color_map.get(interp['color'], 'white'),
                font=ctk.CTkFont(size=12)
            )
            label.grid(row=row, column=0, sticky="w", padx=10, pady=5)
            row += 1
        
        # Balanceamento
        if 'balance' in interpretations:
            interp = interpretations['balance']
            color_map = {'green': '#28a745', 'blue': '#2E86DE', 'yellow': '#ffc107', 'red': '#dc3545'}
            
            label = ctk.CTkLabel(
                grid_frame,
                text=f"• Distribuição: {interp['message']}",
                text_color=color_map.get(interp['color'], 'white'),
                font=ctk.CTkFont(size=12)
            )
            label.grid(row=row, column=0, sticky="w", padx=10, pady=5)
            row += 1
        
        # Inércia
        if metrics['inertia'] is not None:
            label = ctk.CTkLabel(
                grid_frame,
                text=f"• Inércia: {metrics['inertia']:.2f}",
                font=ctk.CTkFont(size=12)
            )
            label.grid(row=row, column=0, sticky="w", padx=10, pady=5)
    
    def _display_centroids(self):
        """Mostra tabela de centroides"""
        centroids_frame = ctk.CTkFrame(self.results_frame)
        centroids_frame.pack(fill="x", padx=15, pady=10)
        
        title = ctk.CTkLabel(
            centroids_frame,
            text="Centroides dos Clusters",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        title.pack(pady=10)
        
        # Criar tabela
        centroids_df = self.results['centroids']
        
        table_frame = ctk.CTkFrame(centroids_frame)
        table_frame.pack(padx=15, pady=10)
        
        # Headers
        headers = list(centroids_df.columns)
        for col_idx, header in enumerate(headers):
            label = ctk.CTkLabel(
                table_frame,
                text=header,
                font=ctk.CTkFont(size=11, weight="bold"),
                fg_color="#1f538d",
                corner_radius=0,
                width=100,
                height=30
            )
            label.grid(row=0, column=col_idx, sticky="ew", padx=1, pady=1)
        
        # Dados
        for row_idx, row in centroids_df.iterrows():
            for col_idx, value in enumerate(row):
                if isinstance(value, (int, float)):
                    text = f"{value:.3f}" if isinstance(value, float) else str(value)
                else:
                    text = str(value)
                
                label = ctk.CTkLabel(
                    table_frame,
                    text=text,
                    font=ctk.CTkFont(size=10),
                    fg_color="#2b2b2b",
                    corner_radius=0,
                    width=100,
                    height=25
                )
                label.grid(row=row_idx + 1, column=col_idx, sticky="ew", padx=1, pady=1)
    
    def _display_visualizations(self, features):
        """Mostra visualizações dos clusters"""
        plots_frame = ctk.CTkFrame(self.results_frame)
        plots_frame.pack(fill="both", expand=True, padx=15, pady=10)
        
        title = ctk.CTkLabel(
            plots_frame,
            text="Visualização dos Clusters",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        title.pack(pady=10)
        
        # Criar figura
        n_features = len(features)
        
        if n_features == 2:
            # 2D scatter plot
            fig = Figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            
            labels = self.results['labels']
            X = self.data[features].values
            
            scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
            
            # Plotar centroides
            centroids = self.results['centroids'][features].values
            ax.scatter(centroids[:, 0], centroids[:, 1], 
                      c='red', marker='X', s=200, edgecolors='black', linewidths=2,
                      label='Centroides')
            
            ax.set_xlabel(features[0], fontsize=12)
            ax.set_ylabel(features[1], fontsize=12)
            ax.set_title('Clusters no Espaço 2D', fontsize=14, weight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Colorbar
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label('Cluster', fontsize=12)
            
        else:
            # Para mais de 2 features: matriz de scatter plots
            n_plots = min(n_features, 3)  # Máximo 3x3
            fig = Figure(figsize=(14, 12))
            
            labels = self.results['labels']
            
            plot_idx = 1
            for i in range(n_plots):
                for j in range(n_plots):
                    if i != j:
                        ax = fig.add_subplot(n_plots, n_plots, plot_idx)
                        
                        X_i = self.data[features[i]].values
                        X_j = self.data[features[j]].values
                        
                        scatter = ax.scatter(X_i, X_j, c=labels, cmap='viridis', s=30, alpha=0.6)
                        
                        # Centroides
                        centroids = self.results['centroids']
                        ax.scatter(centroids[features[i]], centroids[features[j]],
                                 c='red', marker='X', s=100, edgecolors='black', linewidths=1)
                        
                        if j == 0:
                            ax.set_ylabel(features[i], fontsize=10)
                        if i == n_plots - 1:
                            ax.set_xlabel(features[j], fontsize=10)
                        
                        ax.grid(True, alpha=0.3)
                    else:
                        ax = fig.add_subplot(n_plots, n_plots, plot_idx)
                        ax.hist(self.data[features[i]], bins=20, color='skyblue', edgecolor='black')
                        ax.set_ylabel('Frequência', fontsize=10)
                        if i == n_plots - 1:
                            ax.set_xlabel(features[i], fontsize=10)
                    
                    plot_idx += 1
        
        fig.tight_layout()
        
        # Adicionar canvas
        canvas = FigureCanvasTkAgg(fig, plots_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def _clear_results(self):
        """Limpa resultados"""
        for widget in self.results_frame.winfo_children():
            if widget.winfo_class() != 'CTkLabel' or widget.cget('text') != 'Resultados da Análise':
                widget.destroy()
        
        self.results = None
        self.elbow_data = None
