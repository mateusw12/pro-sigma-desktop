"""
Janela de análise Run Chart
"""

import customtkinter as ctk
from tkinter import messagebox
import pandas as pd
import numpy as np


class RunChartWindow(ctk.CTkToplevel):
    """Janela para análise Run Chart"""
    
    def __init__(self, parent, data):
        super().__init__(parent)
        
        # Configuração da janela
        self.title("📈 Run Chart - Gráfico de Execução")
        self.geometry("1600x950")
        self.state('zoomed')
        self.lift()
        self.focus_force()
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        
        # Dados
        self.data = data
        self.results = None
        
        # Imports
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.plt = plt
        self.FigureCanvasTkAgg = FigureCanvasTkAgg
        plt.style.use('seaborn-v0_8-darkgrid')
        
        self._create_widgets()
        self._populate_columns()
    
    def _create_widgets(self):
        """Cria interface"""
        main_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Título
        ctk.CTkLabel(
            main_frame,
            text="Run Chart - Gráfico de Execução",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(0, 5))
        
        ctk.CTkLabel(
            main_frame,
            text="Análise de tendências, shifts e padrões ao longo do tempo",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        ).pack(pady=(0, 20))
        
        # Configuração
        config_frame = ctk.CTkFrame(main_frame)
        config_frame.pack(fill="x", pady=(0, 15))
        
        ctk.CTkLabel(
            config_frame,
            text="⚙️ Configuração",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(15, 10), padx=20, anchor="w")
        
        config_grid = ctk.CTkFrame(config_frame, fg_color="transparent")
        config_grid.pack(fill="x", padx=20, pady=(0, 15))
        
        # Coluna de dados
        ctk.CTkLabel(config_grid, text="Coluna de Dados:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.data_col_combo = ctk.CTkComboBox(config_grid, values=["Selecione..."], width=250)
        self.data_col_combo.grid(row=0, column=1, padx=5, pady=5)
        
        # Coluna de ordem (opcional)
        ctk.CTkLabel(config_grid, text="Coluna de Ordem (opcional):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.order_col_combo = ctk.CTkComboBox(config_grid, values=["Sequencial", "Selecione..."], width=250)
        self.order_col_combo.set("Sequencial")
        self.order_col_combo.grid(row=1, column=1, padx=5, pady=5)
        
        # Botão analisar
        ctk.CTkButton(
            main_frame,
            text="📊 Analisar Run Chart",
            command=self._analyze,
            width=200,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#1f538d"
        ).pack(pady=(0, 15))
        
        # Resultados
        results_label = ctk.CTkLabel(
            main_frame,
            text="📈 Resultados da Análise",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        results_label.pack(pady=(10, 10), anchor="w")
        
        self.results_scroll = ctk.CTkFrame(main_frame, fg_color="transparent")
        self.results_scroll.pack(fill="both", expand=True)
    
    def _populate_columns(self):
        """Popula combos com colunas numéricas"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        all_cols = ["Sequencial"] + list(self.data.columns)
        
        self.data_col_combo.configure(values=numeric_cols)
        if numeric_cols:
            self.data_col_combo.set(numeric_cols[0])
        
        self.order_col_combo.configure(values=all_cols)
    
    def _analyze(self):
        """Executa análise"""
        data_col = self.data_col_combo.get()
        
        if not data_col or data_col == "Selecione...":
            messagebox.showwarning("Atenção", "Selecione a coluna de dados!")
            return
        
        try:
            from src.analytics.run_chart.run_chart_utils import analyze_run_chart, detect_astronomical_points
            
            # Prepara dados
            order_col = self.order_col_combo.get()
            if order_col == "Sequencial":
                data_series = self.data[data_col]
            else:
                sorted_data = self.data.sort_values(by=order_col)
                data_series = sorted_data[data_col].reset_index(drop=True)
            
            # Analisa
            self.results = analyze_run_chart(data_series)
            self.results['data_series'] = data_series
            self.results['data_col'] = data_col
            
            # Detecta pontos astronômicos
            astronomical = detect_astronomical_points(data_series, self.results['stats']['median'])
            self.results['astronomical_points'] = astronomical
            
            # Exibe resultados
            self._display_results()
            
            self.lift()
            self.focus_force()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Erro", f"Erro na análise:\n{str(e)}")
    
    def _display_results(self):
        """Exibe resultados"""
        for widget in self.results_scroll.winfo_children():
            widget.destroy()
        
        self._display_stats()
        self._display_patterns()
        self._display_chart()
    
    def _display_stats(self):
        """Exibe estatísticas"""
        frame = ctk.CTkFrame(self.results_scroll)
        frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(
            frame,
            text="📊 Estatísticas Descritivas",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        stats = self.results['stats']
        runs = self.results['runs_info']
        
        data = [
            ['Observações', int(stats['n'])],
            ['Média', f"{stats['mean']:.4f}"],
            ['Mediana', f"{stats['median']:.4f}"],
            ['Desvio Padrão', f"{stats['std']:.4f}"],
            ['Mínimo', f"{stats['min']:.4f}"],
            ['Máximo', f"{stats['max']:.4f}"],
            ['Range', f"{stats['range']:.4f}"],
            ['', ''],
            ['Runs Observados', int(runs['n_runs'])],
            ['Runs Esperados', f"{runs['expected_runs']:.2f}"],
            ['Z-Score', f"{runs['z_score']:.3f}"],
            ['Longest Run', int(runs['longest_run'])],
            ['Pontos Acima Mediana', int(runs['n_above'])],
            ['Pontos Abaixo Mediana', int(runs['n_below'])],
        ]
        
        table = self._create_table(frame, data)
        table.pack(fill="x", padx=10, pady=5)
    
    def _display_patterns(self):
        """Exibe padrões detectados"""
        frame = ctk.CTkFrame(self.results_scroll)
        frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(
            frame,
            text="🔍 Padrões Detectados",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        interpretations = self.results['interpretations']
        
        if not interpretations:
            ctk.CTkLabel(
                frame,
                text="Nenhum padrão significativo detectado",
                text_color="gray"
            ).pack(padx=20, pady=10)
        else:
            for interp in interpretations:
                color = {
                    'high': '#f87171',
                    'medium': '#fbbf24',
                    'low': '#60a5fa',
                    'none': '#4ade80'
                }.get(interp['severity'], 'white')
                
                pattern_frame = ctk.CTkFrame(frame, fg_color=color)
                pattern_frame.pack(fill="x", padx=10, pady=3)
                
                ctk.CTkLabel(
                    pattern_frame,
                    text=f"{interp['type']}: {interp['description']}",
                    text_color="black",
                    font=ctk.CTkFont(weight="bold")
                ).pack(padx=10, pady=8, anchor="w")
    
    def _display_chart(self):
        """Exibe gráfico"""
        frame = ctk.CTkFrame(self.results_scroll)
        frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(
            frame,
            text="📈 Run Chart",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        fig = self.plt.figure(figsize=(14, 6))
        ax = fig.add_subplot(1, 1, 1)
        
        data_series = self.results['data_series']
        median = self.results['stats']['median']
        
        # Plota dados
        ax.plot(range(len(data_series)), data_series, marker='o', color='#2563eb', linewidth=2, markersize=6, label='Dados')
        
        # Linha da mediana
        ax.axhline(y=median, color='#16a34a', linestyle='--', linewidth=2, label=f'Mediana ({median:.4f})')
        
        # Destaca pontos astronômicos
        astronomical = self.results['astronomical_points']
        if astronomical:
            ax.scatter(astronomical, data_series.iloc[astronomical], color='red', s=100, zorder=5, label='Pontos Astronômicos')
        
        ax.set_xlabel('Ordem de Observação', fontsize=12)
        ax.set_ylabel(self.results['data_col'], fontsize=12)
        ax.set_title('Run Chart - Análise Temporal', fontsize=14, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        canvas = self.FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=5)
    
    def _create_table(self, parent, data):
        """Cria tabela simples"""
        table_frame = ctk.CTkFrame(parent, fg_color="#2b2b2b")
        
        for i, row in enumerate(data):
            if len(row) == 2 and row[0] == '':
                continue
            
            row_frame = ctk.CTkFrame(table_frame, fg_color="#2b2b2b", height=25)
            row_frame.pack(fill="x")
            row_frame.pack_propagate(False)
            
            ctk.CTkLabel(row_frame, text=str(row[0]), anchor="w").place(relx=0, rely=0, relwidth=0.6, relheight=1)
            ctk.CTkLabel(row_frame, text=str(row[1]), anchor="e").place(relx=0.6, rely=0, relwidth=0.4, relheight=1)
        
        return table_frame
