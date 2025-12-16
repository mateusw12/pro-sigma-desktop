"""
Janela de análise Pareto Chart
"""

import customtkinter as ctk
from tkinter import messagebox
import numpy as np


class ParetoWindow(ctk.CTkToplevel):
    """Janela para análise de Pareto"""
    
    def __init__(self, parent, data):
        super().__init__(parent)
        
        # Configuração da janela
        self.title("📊 Pareto Chart - Análise 80/20")
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
            text="Pareto Chart - Análise 80/20",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(0, 5))
        
        ctk.CTkLabel(
            main_frame,
            text="Identificação dos fatores vitais (vital few) vs fatores triviais (trivial many)",
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
        
        # Coluna de categorias
        ctk.CTkLabel(config_grid, text="Coluna de Categorias:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.category_col_combo = ctk.CTkComboBox(config_grid, values=["Selecione..."], width=250)
        self.category_col_combo.grid(row=0, column=1, padx=5, pady=5)
        
        # Coluna de valores (opcional)
        ctk.CTkLabel(config_grid, text="Coluna de Valores (opcional):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.value_col_combo = ctk.CTkComboBox(config_grid, values=["Frequência (contar)", "Selecione..."], width=250)
        self.value_col_combo.set("Frequência (contar)")
        self.value_col_combo.grid(row=1, column=1, padx=5, pady=5)
        
        # Botão analisar
        ctk.CTkButton(
            main_frame,
            text="📊 Analisar Pareto",
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
        """Popula combos"""
        all_cols = list(self.data.columns)
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        value_options = ["Frequência (contar)"] + numeric_cols
        
        self.category_col_combo.configure(values=all_cols)
        if all_cols:
            self.category_col_combo.set(all_cols[0])
        
        self.value_col_combo.configure(values=value_options)
    
    def _analyze(self):
        """Executa análise"""
        category_col = self.category_col_combo.get()
        
        if not category_col or category_col == "Selecione...":
            messagebox.showwarning("Atenção", "Selecione a coluna de categorias!")
            return
        
        try:
            from src.analytics.pareto.pareto_utils import calculate_pareto, analyze_pareto_principle
            
            # Prepara dados
            value_col_selection = self.value_col_combo.get()
            value_col = None if value_col_selection == "Frequência (contar)" else value_col_selection
            
            # Analisa
            self.results = calculate_pareto(self.data, category_col, value_col)
            self.results['pareto_analysis'] = analyze_pareto_principle(self.results)
            
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
        
        self._display_summary()
        self._display_vital_few()
        self._display_abc_table()
        self._display_chart()
    
    def _display_summary(self):
        """Exibe resumo"""
        frame = ctk.CTkFrame(self.results_scroll)
        frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(
            frame,
            text="📊 Resumo da Análise",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        analysis = self.results['pareto_analysis']
        
        # Card de aderência
        adherence_frame = ctk.CTkFrame(frame, fg_color=analysis['adherence_color'])
        adherence_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(
            adherence_frame,
            text=f"Aderência ao Princípio de Pareto: {analysis['adherence']}",
            text_color="black",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(padx=10, pady=5)
        
        ctk.CTkLabel(
            adherence_frame,
            text=analysis['description'],
            text_color="black"
        ).pack(padx=10, pady=(0, 5))
        
        # Estatísticas
        data = [
            ['Total de Categorias', self.results['n_categories']],
            ['Total Geral', f"{self.results['total']:.2f}"],
            ['', ''],
            ['Categorias Vitais (A)', self.results['n_vital_few']],
            ['% de Categorias Vitais', f"{self.results['percent_vital']:.1f}%"],
            ['Contribuição das Vitais', f"{self.results['vital_contribution_pct']:.1f}%"],
        ]
        
        table = self._create_table(frame, data)
        table.pack(fill="x", padx=10, pady=5)
    
    def _display_vital_few(self):
        """Exibe categorias vitais"""
        frame = ctk.CTkFrame(self.results_scroll)
        frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(
            frame,
            text="⭐ Categorias Vitais (Vital Few)",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        pareto_data = self.results['pareto_data']
        vital_data = pareto_data[pareto_data['Category'].isin(self.results['vital_few_categories'])]
        
        # Cria tabela de vitais
        table_frame = ctk.CTkFrame(frame, fg_color="#2b2b2b")
        table_frame.pack(fill="x", padx=10, pady=5)
        
        # Header
        header = ctk.CTkFrame(table_frame, fg_color="#1f538d", height=30)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        headers = ['Categoria', 'Valor', '% Individual', '% Acumulado', 'Classe']
        for i, h in enumerate(headers):
            ctk.CTkLabel(header, text=h, font=ctk.CTkFont(weight="bold"), text_color="white").place(
                relx=i/len(headers), rely=0, relwidth=1/len(headers), relheight=1
            )
        
        # Rows
        for _, row in vital_data.iterrows():
            row_frame = ctk.CTkFrame(table_frame, fg_color="#2b2b2b", height=25)
            row_frame.pack(fill="x")
            row_frame.pack_propagate(False)
            
            values = [
                str(row['Category']),
                f"{row['Count']:.2f}",
                f"{row['Percent']:.2f}%",
                f"{row['Cumulative_Percent']:.2f}%",
                row['ABC_Class']
            ]
            
            for i, val in enumerate(values):
                ctk.CTkLabel(row_frame, text=val, text_color="white").place(
                    relx=i/len(values), rely=0, relwidth=1/len(values), relheight=1
                )
    
    def _display_abc_table(self):
        """Exibe classificação ABC"""
        frame = ctk.CTkFrame(self.results_scroll)
        frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(
            frame,
            text="🏷️ Classificação ABC",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        abc_counts = self.results['abc_counts']
        
        data = [
            ['Classe A (até 80%)', abc_counts.get('A', 0)],
            ['Classe B (80% - 95%)', abc_counts.get('B', 0)],
            ['Classe C (acima 95%)', abc_counts.get('C', 0)],
        ]
        
        table = self._create_table(frame, data)
        table.pack(fill="x", padx=10, pady=5)
    
    def _display_chart(self):
        """Exibe gráfico de Pareto"""
        frame = ctk.CTkFrame(self.results_scroll)
        frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(
            frame,
            text="📈 Gráfico de Pareto",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        fig = self.plt.figure(figsize=(14, 7))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        
        pareto_data = self.results['pareto_data']
        categories = pareto_data['Category'].astype(str)
        counts = pareto_data['Count']
        cumulative = pareto_data['Cumulative_Percent']
        
        # Limita a 20 categorias para visualização
        if len(categories) > 20:
            categories = categories[:20]
            counts = counts[:20]
            cumulative = cumulative[:20]
        
        x = range(len(categories))
        
        # Barras
        colors = ['#4ade80' if cat in self.results['vital_few_categories'] else '#60a5fa' 
                  for cat in pareto_data['Category'][:len(categories)]]
        ax1.bar(x, counts, color=colors, alpha=0.8, label='Frequência')
        
        # Linha acumulada
        ax2.plot(x, cumulative, color='#ef4444', marker='D', linewidth=2, markersize=8, label='% Acumulado')
        ax2.axhline(y=80, color='#fbbf24', linestyle='--', linewidth=2, label='80%')
        
        # Configurações
        ax1.set_xlabel('Categorias', fontsize=12)
        ax1.set_ylabel('Frequência/Valor', fontsize=12, color='#4ade80')
        ax1.tick_params(axis='y', labelcolor='#4ade80')
        
        ax2.set_ylabel('% Acumulado', fontsize=12, color='#ef4444')
        ax2.set_ylim(0, 105)
        ax2.tick_params(axis='y', labelcolor='#ef4444')
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, rotation=45, ha='right')
        
        ax1.set_title('Gráfico de Pareto - Vital Few vs Trivial Many', fontsize=14, weight='bold')
        
        # Legendas
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax1.grid(True, alpha=0.3, axis='y')
        
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
