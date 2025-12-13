"""
Control Charts Window
Cartas de Controle Estatístico de Processo (CEP)
Supports: X-bar & R, X-bar & S, I-MR, P, NP, C, U charts
"""
import customtkinter as ctk
from tkinter import messagebox
from src.utils.lazy_imports import get_numpy, get_pandas, get_matplotlib_figure, get_matplotlib_backend
from src.analytics.control_charts.control_chart_utils import (
    calculate_individual_mr,
    calculate_xbar_r,
    calculate_xbar_s,
    calculate_p_chart,
    calculate_np_chart,
    calculate_c_chart,

    calculate_u_chart,
    get_control_constants
)
from typing import Dict


class ControlChartWindow(ctk.CTkToplevel):
    """Janela de Cartas de Controle"""
    
    def __init__(self, parent, df):
        super().__init__(parent)
        
        # Carrega bibliotecas lazy
        self.pd = get_pandas()
        self.np = get_numpy()
        self.Figure = get_matplotlib_figure()
        self.FigureCanvasTkAgg = get_matplotlib_backend()
        
        self.title("Cartas de Controle - CEP")
        self.geometry("1200x800")
        self.minsize(1000, 700)
        
        try:
            self.state('zoomed')
        except:
            pass
        
        self.transient(parent)
        self.grab_set()
        
        self.df = df.copy()
        self.numeric_cols = [c for c in self.df.columns if self.pd.api.types.is_numeric_dtype(self.df[c])]
        
        self._build_ui()
    
    def _build_ui(self):
        """Constrói a interface"""
        main = ctk.CTkFrame(self)
        main.pack(fill="both", expand=True, padx=16, pady=16)
        
        # Left panel
        left_container = ctk.CTkFrame(main, width=300)
        left_container.pack(side="left", fill="y")
        left_container.pack_propagate(False)
        
        left = ctk.CTkScrollableFrame(left_container)
        left.pack(fill="both", expand=True)
        
        # Right panel
        right_container = ctk.CTkFrame(main)
        right_container.pack(side="right", fill="both", expand=True, padx=(12, 0))
        
        right = ctk.CTkScrollableFrame(right_container)
        right.pack(fill="both", expand=True)
        
        # === LEFT PANEL ===
        ctk.CTkLabel(
            left,
            text="Cartas de Controle",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(12, 4))
        
        ctk.CTkLabel(
            left,
            text="Controle Estatístico de Processo (CEP)",
            text_color="gray",
            wraplength=260
        ).pack(pady=(0, 12), padx=8)
        
        # Chart type selection
        type_label = ctk.CTkLabel(left, text="Tipo de Carta", font=ctk.CTkFont(size=12, weight="bold"))
        type_label.pack(anchor="w", padx=12, pady=(8, 4))
        
        self.chart_type = ctk.StringVar(value="xbar_r")
        
        chart_types = [
            ("X̄ & R (Média e Amplitude)", "xbar_r"),
            ("X̄ & S (Média e Desvio)", "xbar_s"),
            ("I-MR (Individual e Amplitude Móvel)", "i_mr"),
            ("P (Proporção de Defeituosos)", "p"),
            ("NP (Número de Defeituosos)", "np"),
            ("C (Número de Defeitos)", "c"),
            ("U (Defeitos por Unidade)", "u"),
        ]
        
        for label, value in chart_types:
            ctk.CTkRadioButton(
                left,
                text=label,
                variable=self.chart_type,
                value=value,
                command=self._on_chart_type_change
            ).pack(anchor="w", padx=20, pady=2)
        
        # Column selection
        col_label = ctk.CTkLabel(left, text="Coluna de Dados", font=ctk.CTkFont(size=12, weight="bold"))
        col_label.pack(anchor="w", padx=12, pady=(12, 4))
        
        if not self.numeric_cols:
            ctk.CTkLabel(left, text="Sem colunas numéricas", text_color="red").pack(pady=6, padx=12)
            self.data_column = None
        else:
            self.data_column = ctk.CTkOptionMenu(
                left,
                values=self.numeric_cols,
                width=260
            )
            self.data_column.pack(padx=12, pady=4)
            self.data_column.set(self.numeric_cols[0])
        
        # Subgroup size (for variable charts)
        self.subgroup_frame = ctk.CTkFrame(left, fg_color="transparent")
        self.subgroup_frame.pack(fill="x", padx=12, pady=(12, 0))
        
        ctk.CTkLabel(
            self.subgroup_frame,
            text="Tamanho do Subgrupo",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w")
        
        self.subgroup_size = ctk.CTkEntry(self.subgroup_frame, placeholder_text="Ex: 5")
        self.subgroup_size.pack(fill="x", pady=4)
        self.subgroup_size.insert(0, "5")
        
        # Sample size (for attribute charts)
        self.sample_frame = ctk.CTkFrame(left, fg_color="transparent")
        
        ctk.CTkLabel(
            self.sample_frame,
            text="Tamanho da Amostra (n)",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w")
        
        self.sample_size = ctk.CTkEntry(self.sample_frame, placeholder_text="Ex: 100")
        self.sample_size.pack(fill="x", pady=4)
        self.sample_size.insert(0, "100")
        
        # Sigma limits
        limits_label = ctk.CTkLabel(left, text="Limites de Controle", font=ctk.CTkFont(size=12, weight="bold"))
        limits_label.pack(anchor="w", padx=12, pady=(12, 4))
        
        self.sigma_limits = ctk.CTkOptionMenu(
            left,
            values=["3 sigma (99.73%)", "2 sigma (95.45%)", "1 sigma (68.27%)"],
            width=260
        )
        self.sigma_limits.pack(padx=12, pady=4)
        self.sigma_limits.set("3 sigma (99.73%)")
        
        # Spacer
        ctk.CTkFrame(left, fg_color="transparent").pack(fill="both", expand=True)
        
        # Generate button
        generate_btn = ctk.CTkButton(
            left,
            text="Gerar Carta",
            command=self._generate_chart,
            height=44,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#2E86DE",
            hover_color="#1E5BA8"
        )
        generate_btn.pack(fill="x", padx=12, pady=(10, 14))
        
        ctk.CTkFrame(left, fg_color="transparent", height=20).pack()
        
        # === RIGHT PANEL ===
        # Stats table
        self.stats_frame = ctk.CTkFrame(right)
        self.stats_frame.pack(fill="x", padx=8, pady=(8, 6))
        
        ctk.CTkLabel(
            self.stats_frame,
            text="Estatísticas da Carta",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=8)
        
        # Chart area
        self.plot_frame = ctk.CTkFrame(right)
        self.plot_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        
        # Initial state
        self._on_chart_type_change()
    
    def _on_chart_type_change(self):
        """Atualiza UI baseado no tipo de carta selecionado"""
        chart_type = self.chart_type.get()
        
        # Variable charts need subgroup size
        if chart_type in ['xbar_r', 'xbar_s']:
            self.subgroup_frame.pack(fill="x", padx=12, pady=(12, 0))
            self.sample_frame.pack_forget()
        # I-MR doesn't need subgroup
        elif chart_type == 'i_mr':
            self.subgroup_frame.pack_forget()
            self.sample_frame.pack_forget()
        # Attribute charts need sample size
        else:
            self.subgroup_frame.pack_forget()
            self.sample_frame.pack(fill="x", padx=12, pady=(12, 0))
    
    def _generate_chart(self):
        """Gera a carta de controle"""
        if not self.data_column:
            messagebox.showerror("Erro", "Nenhuma coluna numérica disponível")
            return
        
        chart_type = self.chart_type.get()
        column = self.data_column.get()
        
        try:
            data = self.df[column].dropna().values
            
            if len(data) < 2:
                messagebox.showerror("Erro", "Dados insuficientes para gerar carta")
                return
            
            # Generate appropriate chart
            if chart_type == 'xbar_r':
                self._generate_xbar_r_chart(data)
            elif chart_type == 'xbar_s':
                self._generate_xbar_s_chart(data)
            elif chart_type == 'i_mr':
                self._generate_i_mr_chart(data)
            elif chart_type == 'p':
                self._generate_p_chart(data)
            elif chart_type == 'np':
                self._generate_np_chart(data)
            elif chart_type == 'c':
                self._generate_c_chart(data)
            elif chart_type == 'u':
                self._generate_u_chart(data)
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao gerar carta:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def _generate_xbar_r_chart(self, data):
        """Gera carta X-bar & R"""
        try:
            n = int(self.subgroup_size.get())
        except ValueError:
            messagebox.showerror("Erro", "Tamanho de subgrupo inválido")
            return
        
        if n < 2 or n > 25:
            messagebox.showerror("Erro", "Tamanho de subgrupo deve estar entre 2 e 25")
            return
        
        constants = get_control_constants()
        if n not in constants:
            messagebox.showerror("Erro", f"Constantes não disponíveis para n={n}")
            return
        
        num_complete = len(data) // n
        if num_complete < 2:
            messagebox.showerror("Erro", f"Dados insuficientes para {num_complete} subgrupos")
            return
        
        # Calcula usando utils
        result = calculate_xbar_r(data, n)
        
        xbar_data = result['xbar']
        range_data = result['range']
        
        # Plot
        self._plot_two_charts(
            xbar_data['data'], range_data['data'],
            xbar_data['mean'], range_data['mean'],
            xbar_data['ucl'], xbar_data['lcl'],
            range_data['ucl'], range_data['lcl'],
            "Carta X̄", "Carta R",
            f"X̄ (n={n})", "R (Amplitude)"
        )
        
        # Update stats com tabela horizontal
        self._update_stats({
            "chart1_name": "Xbar",
            "lcl1": f"{xbar_data['lcl']:.4f}",
            "mean1": f"{xbar_data['mean']:.4f}",
            "std1": f"{xbar_data['std']:.4f}",
            "ucl1": f"{xbar_data['ucl']:.4f}",
            "chart2_name": "Range",
            "lcl2": f"{range_data['lcl']:.4f}",
            "mean2": f"{range_data['mean']:.4f}",
            "std2": f"{range_data['std']:.4f}",
            "ucl2": f"{range_data['ucl']:.4f}",
            "subgroup_size": str(n),
            "num_levels": f"> {result['num_subgroups']}"
        }, horizontal=True)
    
    def _generate_xbar_s_chart(self, data):
        """Gera carta X-bar & S"""
        try:
            n = int(self.subgroup_size.get())
        except ValueError:
            messagebox.showerror("Erro", "Tamanho de subgrupo inválido")
            return
        
        if n < 2 or n > 25:
            messagebox.showerror("Erro", "Tamanho de subgrupo deve estar entre 2 e 25")
            return
        
        constants = get_control_constants()
        if n not in constants:
            messagebox.showerror("Erro", f"Constantes não disponíveis para n={n}")
            return
        
        num_complete = len(data) // n
        if num_complete < 2:
            messagebox.showerror("Erro", f"Dados insuficientes para {num_complete} subgrupos")
            return
        
        # Calcula usando utils
        result = calculate_xbar_s(data, n)
        
        xbar_data = result['xbar']
        s_data = result['stdev']
        
        self._plot_two_charts(
            xbar_data['data'], s_data['data'],
            xbar_data['mean'], s_data['mean'],
            xbar_data['ucl'], xbar_data['lcl'],
            s_data['ucl'], s_data['lcl'],
            "Carta X̄", "Carta S",
            f"X̄ (n={n})", "S (Desvio Padrão)"
        )
        
        self._update_stats({
            "chart1_name": "Xbar",
            "lcl1": f"{xbar_data['lcl']:.4f}",
            "mean1": f"{xbar_data['mean']:.4f}",
            "std1": f"{xbar_data['std']:.4f}",
            "ucl1": f"{xbar_data['ucl']:.4f}",
            "chart2_name": "StdDev",
            "lcl2": f"{s_data['lcl']:.4f}",
            "mean2": f"{s_data['mean']:.4f}",
            "std2": f"{s_data['std']:.4f}",
            "ucl2": f"{s_data['ucl']:.4f}",
            "subgroup_size": str(n),
            "num_levels": f"> {result['num_subgroups']}"
        }, horizontal=True)
    
    def _generate_i_mr_chart(self, data):
        """Gera carta I-MR (Individual e Moving Range)"""
        # Calcula usando utils (seguindo o código original)
        result = calculate_individual_mr(data)
        
        i_data = result['individuals']
        mr_data = result['moving_range']
        
        # Pad MR com NaN no início para alinhar com dados individuais
        mr_plot = self.np.concatenate([[self.np.nan], mr_data['data']])
        
        self._plot_two_charts(
            i_data['data'], mr_plot,
            i_data['mean'], mr_data['mean'],
            i_data['ucl'], i_data['lcl'],
            mr_data['ucl'], mr_data['lcl'],
            "Carta I (Individual)", "Carta MR (Moving Range)",
            "Valores Individuais", "Amplitude Móvel"
        )
        
        self._update_stats({
            "chart1_name": "Individual",
            "lcl1": f"{i_data['lcl']:.4f}",
            "mean1": f"{i_data['mean']:.4f}",
            "std1": f"{i_data['std']:.4f}",
            "ucl1": f"{i_data['ucl']:.4f}",
            "chart2_name": "Moving Range",
            "lcl2": f"{mr_data['lcl']:.4f}",
            "mean2": f"{mr_data['mean']:.4f}",
            "std2": "0.0000",
            "ucl2": f"{mr_data['ucl']:.4f}",
            "subgroup_size": "1",
            "num_levels": f"> {len(data)}"
        }, horizontal=True)
    
    def _generate_p_chart(self, data):
        """Gera carta P (proporção de defeituosos)"""
        try:
            n = int(self.sample_size.get())
        except ValueError:
            messagebox.showerror("Erro", "Tamanho de amostra inválido")
            return
        
        if n < 1:
            messagebox.showerror("Erro", "Tamanho de amostra deve ser > 0")
            return
        
        # Calcula usando utils
        result = calculate_p_chart(data, n)
        
        self._plot_single_chart(
            result['proportions'], result['mean'], result['ucl'], result['lcl'],
            "Carta P (Proporção de Defeituosos)",
            "Proporção (p)"
        )
        
        # Tabela horizontal profissional
        std_p = self.np.std(result['proportions'])
        self._update_stats({
            "chart1_name": "P Chart",
            "lcl1": f"{result['lcl']:.4f}",
            "mean1": f"{result['mean']:.4f}",
            "std1": f"{std_p:.4f}",
            "ucl1": f"{result['ucl']:.4f}",
            "chart2_name": "",
            "lcl2": "",
            "mean2": "",
            "std2": "",
            "ucl2": "",
            "subgroup_size": str(n),
            "num_levels": f"> {len(data)}"
        }, horizontal=True)
    
    def _generate_np_chart(self, data):
        """Gera carta NP (número de defeituosos)"""
        try:
            n = int(self.sample_size.get())
        except ValueError:
            messagebox.showerror("Erro", "Tamanho de amostra inválido")
            return
        
        if n < 1:
            messagebox.showerror("Erro", "Tamanho de amostra deve ser > 0")
            return
        
        # Calcula usando utils
        result = calculate_np_chart(data, n)
        
        self._plot_single_chart(
            result['data'], result['mean'], result['ucl'], result['lcl'],
            "Carta NP (Número de Defeituosos)",
            "Número de Defeituosos (np)"
        )
        
        # Tabela horizontal profissional
        std_np = self.np.std(result['data'])
        self._update_stats({
            "chart1_name": "NP Chart",
            "lcl1": f"{result['lcl']:.4f}",
            "mean1": f"{result['mean']:.4f}",
            "std1": f"{std_np:.4f}",
            "ucl1": f"{result['ucl']:.4f}",
            "chart2_name": "",
            "lcl2": "",
            "mean2": "",
            "std2": "",
            "ucl2": "",
            "subgroup_size": str(n),
            "num_levels": f"> {len(data)}"
        }, horizontal=True)
    
    def _generate_c_chart(self, data):
        """Gera carta C (número de defeitos)"""
        # Calcula usando utils
        result = calculate_c_chart(data)
        
        self._plot_single_chart(
            result['data'], result['mean'], result['ucl'], result['lcl'],
            "Carta C (Número de Defeitos)",
            "Número de Defeitos (c)"
        )
        
        # Tabela horizontal profissional
        std_c = self.np.std(result['data'])
        self._update_stats({
            "chart1_name": "C Chart",
            "lcl1": f"{result['lcl']:.4f}",
            "mean1": f"{result['mean']:.4f}",
            "std1": f"{std_c:.4f}",
            "ucl1": f"{result['ucl']:.4f}",
            "chart2_name": "",
            "lcl2": "",
            "mean2": "",
            "std2": "",
            "ucl2": "",
            "subgroup_size": "1",
            "num_levels": f"> {len(data)}"
        }, horizontal=True)
    
    def _generate_u_chart(self, data):
        """Gera carta U (defeitos por unidade)"""
        try:
            n = int(self.sample_size.get())
        except ValueError:
            messagebox.showerror("Erro", "Tamanho de amostra inválido")
            return
        
        if n < 1:
            messagebox.showerror("Erro", "Tamanho de amostra deve ser > 0")
            return
        
        # Calcula usando utils
        result = calculate_u_chart(data, n)
        
        self._plot_single_chart(
            result['u_values'], result['mean'], result['ucl'], result['lcl'],
            "Carta U (Defeitos por Unidade)",
            "Defeitos por Unidade (u)"
        )
        
        # Tabela horizontal profissional
        std_u = self.np.std(result['u_values'])
        self._update_stats({
            "chart1_name": "U Chart",
            "lcl1": f"{result['lcl']:.4f}",
            "mean1": f"{result['mean']:.4f}",
            "std1": f"{std_u:.4f}",
            "ucl1": f"{result['ucl']:.4f}",
            "chart2_name": "",
            "lcl2": "",
            "mean2": "",
            "std2": "",
            "ucl2": "",
            "subgroup_size": str(n),
            "num_levels": f"> {len(data)}"
        }, horizontal=True)
    
    def _plot_single_chart(self, data, center, ucl, lcl, title, ylabel):
        """Plota uma única carta de controle"""
        # Clear previous
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        fig = self.Figure(figsize=(10, 5), dpi=100)
        ax = fig.add_subplot(111)
        
        x = self.np.arange(1, len(data) + 1)
        
        # Plot data
        ax.plot(x, data, 'bo-', linewidth=1.5, markersize=6, label='Dados')
        
        # Control limits
        ax.axhline(center, color='green', linestyle='-', linewidth=2, label='Linha Central')
        ax.axhline(ucl, color='red', linestyle='--', linewidth=2, label='LCS')
        ax.axhline(lcl, color='red', linestyle='--', linewidth=2, label='LCI')
        
        # Highlight out of control points
        out_of_control = (data > ucl) | (data < lcl)
        if self.np.any(out_of_control):
            ax.plot(x[out_of_control], data[out_of_control], 'ro', markersize=10, 
                   markerfacecolor='none', markeredgewidth=2, label='Fora de Controle')
        
        ax.set_xlabel('Amostra', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        canvas = self.FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def _plot_two_charts(self, data1, data2, center1, center2, 
                         ucl1, lcl1, ucl2, lcl2, 
                         title1, title2, ylabel1, ylabel2):
        """Plota duas cartas de controle"""
        # Clear previous
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        fig = self.Figure(figsize=(10, 8), dpi=100)
        
        # Upper chart
        ax1 = fig.add_subplot(211)
        x = self.np.arange(1, len(data1) + 1)
        
        ax1.plot(x, data1, 'bo-', linewidth=1.5, markersize=6, label='Dados')
        ax1.axhline(center1, color='green', linestyle='-', linewidth=2, label='Linha Central')
        ax1.axhline(ucl1, color='red', linestyle='--', linewidth=2, label='LCS')
        ax1.axhline(lcl1, color='red', linestyle='--', linewidth=2, label='LCI')
        
        out1 = (data1 > ucl1) | (data1 < lcl1)
        if self.np.any(out1):
            ax1.plot(x[out1], data1[out1], 'ro', markersize=10, 
                    markerfacecolor='none', markeredgewidth=2, label='Fora de Controle')
        
        ax1.set_ylabel(ylabel1, fontweight='bold')
        ax1.set_title(title1, fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Lower chart
        ax2 = fig.add_subplot(212)
        
        # Handle NaN in data2
        valid_mask = ~self.np.isnan(data2)
        x2 = x[valid_mask]
        data2_clean = data2[valid_mask]
        
        ax2.plot(x2, data2_clean, 'bo-', linewidth=1.5, markersize=6, label='Dados')
        ax2.axhline(center2, color='green', linestyle='-', linewidth=2, label='Linha Central')
        ax2.axhline(ucl2, color='red', linestyle='--', linewidth=2, label='LCS')
        ax2.axhline(lcl2, color='red', linestyle='--', linewidth=2, label='LCI')
        
        out2 = (data2_clean > ucl2) | (data2_clean < lcl2)
        if self.np.any(out2):
            ax2.plot(x2[out2], data2_clean[out2], 'ro', markersize=10,
                    markerfacecolor='none', markeredgewidth=2, label='Fora de Controle')
        
        ax2.set_xlabel('Subgrupo', fontweight='bold')
        ax2.set_ylabel(ylabel2, fontweight='bold')
        ax2.set_title(title2, fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        canvas = self.FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def _update_stats(self, stats: Dict[str, str], horizontal: bool = False):
        """Atualiza tabela de estatísticas"""
        # Clear previous
        for widget in self.stats_frame.winfo_children():
            widget.destroy()
        
        ctk.CTkLabel(
            self.stats_frame,
            text="Model Adjustement Information",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=8)
        
        if horizontal:
            # Tabela horizontal para cartas duplas (X-bar & R, X-bar & S)
            self._create_horizontal_table(stats)
        else:
            # Tabela vertical para cartas simples
            grid_frame = ctk.CTkFrame(self.stats_frame, fg_color="transparent")
            grid_frame.pack(padx=10, pady=(0, 10))
            
            row = 0
            for key, value in stats.items():
                ctk.CTkLabel(
                    grid_frame,
                    text=f"{key}:",
                    font=ctk.CTkFont(weight="bold"),
                    anchor="w",
                    width=150
                ).grid(row=row, column=0, sticky="w", padx=5, pady=3)
                
                ctk.CTkLabel(
                    grid_frame,
                    text=value,
                    anchor="w",
                    width=150
                ).grid(row=row, column=1, sticky="w", padx=5, pady=3)
                
                row += 1
    
    def _create_horizontal_table(self, stats: Dict[str, str]):
        """Cria tabela horizontal no estilo Minitab"""
        import tkinter as tk
        from tkinter import ttk
        
        # Container for table
        table_container = ctk.CTkFrame(self.stats_frame)
        table_container.pack(fill="x", padx=10, pady=(0, 10))
        
        # Create Treeview for table
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview",
                       background="#2b2b2b",
                       foreground="white",
                       fieldbackground="#2b2b2b",
                       borderwidth=0,
                       rowheight=30)
        style.configure("Treeview.Heading",
                       background="#1f538d",
                       foreground="white",
                       borderwidth=1,
                       relief="solid",
                       font=('TkDefaultFont', 9, 'bold'))
        style.map("Treeview.Heading",
                 background=[('active', '#2E86DE')])
        style.map("Treeview",
                 background=[('selected', '#2E86DE')])
        
        # Define columns
        columns = ("points", "lcl", "mean", "std", "ucl", "subgroup", "levels")
        
        # Determina se é carta dupla ou simples
        has_second_chart = stats.get("chart2_name", "").strip() != ""
        height = 2 if has_second_chart else 1
        
        tree = ttk.Treeview(table_container, columns=columns, show='headings', height=height)
        
        # Configure column headings
        tree.heading("points", text="Points Plotted")
        tree.heading("lcl", text="LCL")
        tree.heading("mean", text="Mean")
        tree.heading("std", text="Standard Deviation")
        tree.heading("ucl", text="UCL")
        tree.heading("subgroup", text="Subgroup Size")
        tree.heading("levels", text="Number of Levels")
        
        # Configure column widths
        tree.column("points", width=120, anchor="center")
        tree.column("lcl", width=100, anchor="center")
        tree.column("mean", width=100, anchor="center")
        tree.column("std", width=140, anchor="center")
        tree.column("ucl", width=100, anchor="center")
        tree.column("subgroup", width=110, anchor="center")
        tree.column("levels", width=130, anchor="center")
        
        # Insert data for first chart (Xbar or main chart)
        tree.insert("", "end", values=(
            stats.get("chart1_name", "Chart"),
            stats.get("lcl1", ""),
            stats.get("mean1", ""),
            stats.get("std1", ""),
            stats.get("ucl1", ""),
            stats.get("subgroup_size", ""),
            stats.get("num_levels", "")
        ))
        
        # Insert data for second chart only if it exists
        if has_second_chart:
            tree.insert("", "end", values=(
                stats.get("chart2_name", ""),
                stats.get("lcl2", ""),
                stats.get("mean2", ""),
                stats.get("std2", ""),
                stats.get("ucl2", ""),
                "",  # Subgroup size only in first row
                ""   # Levels only in first row
            ))
        
        tree.pack(fill="x", padx=5, pady=5)
