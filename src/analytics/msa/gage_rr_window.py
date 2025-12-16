"""
Janela de análise Gage R&R (Measurement System Analysis)
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import json
from pathlib import Path


class GageRRWindow(ctk.CTkToplevel):
    """Janela para análise Gage R&R"""
    
    def __init__(self, parent, data=None):
        super().__init__(parent)
        
        # Configuração da janela
        self.title("📊 Gage R&R - Measurement System Analysis")
        self.geometry("1600x950")
        
        # Maximiza a janela
        self.state('zoomed')
        
        # Mantém foco na janela
        self.lift()
        self.focus_force()
        
        # Previne fechar a aplicação toda
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        
        # Estado
        self.data = None
        self.results = None
        self.part_col = None
        self.operator_col = None
        self.measurement_cols = []
        self.tolerance = None
        
        # Lazy imports
        self.pd = None
        self.np = None
        self.plt = None
        self.FigureCanvasTkAgg = None
        
        self._init_lazy_imports()
        self._create_widgets()
        
        # Se dados foram fornecidos, carrega automaticamente
        if data is not None:
            self._load_from_dataframe(data)
        
    def _init_lazy_imports(self):
        """Inicializa imports lazy"""
        try:
            import pandas as pd
            import numpy as np
            self.pd = pd
            self.np = np
            
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            self.plt = plt
            self.FigureCanvasTkAgg = FigureCanvasTkAgg
            
            # Configuração do matplotlib
            plt.style.use('seaborn-v0_8-darkgrid')
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao importar bibliotecas: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _create_widgets(self):
        """Cria todos os widgets da interface"""
        
        # Frame principal com scroll
        main_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Título e descrição
        title_label = ctk.CTkLabel(
            main_frame,
            text="Gage R&R - Measurement System Analysis",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack(pady=(0, 5))
        
        desc_label = ctk.CTkLabel(
            main_frame,
            text="Análise de Repetibilidade e Reprodutibilidade do Sistema de Medição",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        desc_label.pack(pady=(0, 20))
        
        # ========== SEÇÃO DE CONFIGURAÇÃO ==========
        config_frame = ctk.CTkFrame(main_frame)
        config_frame.pack(fill="x", pady=(0, 15))
        
        ctk.CTkLabel(
            config_frame,
            text="⚙️ Configuração da Análise",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(15, 10), padx=20, anchor="w")
        
        # Grid para configurações
        config_grid = ctk.CTkFrame(config_frame, fg_color="transparent")
        config_grid.pack(fill="x", padx=20, pady=(0, 15))
        
        # Coluna de Peças (Parts)
        ctk.CTkLabel(config_grid, text="Coluna de Peças:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.part_col_combo = ctk.CTkComboBox(config_grid, values=["Selecione..."], width=200, state="disabled")
        self.part_col_combo.grid(row=0, column=1, padx=5, pady=5)
        
        # Coluna de Operadores
        ctk.CTkLabel(config_grid, text="Coluna de Operadores:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.operator_col_combo = ctk.CTkComboBox(config_grid, values=["Selecione..."], width=200, state="disabled")
        self.operator_col_combo.grid(row=1, column=1, padx=5, pady=5)
        
        # Colunas de Medições
        ctk.CTkLabel(config_grid, text="Colunas de Medições:").grid(row=2, column=0, padx=5, pady=5, sticky="nw")
        
        measurement_frame = ctk.CTkFrame(config_grid, fg_color="transparent")
        measurement_frame.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        self.measurement_listbox = ctk.CTkTextbox(measurement_frame, width=200, height=80, state="disabled")
        self.measurement_listbox.pack(side="left", fill="both", expand=True)
        
        measurement_btn_frame = ctk.CTkFrame(measurement_frame, fg_color="transparent")
        measurement_btn_frame.pack(side="left", padx=5)
        
        self.add_measurement_btn = ctk.CTkButton(
            measurement_btn_frame,
            text="Adicionar",
            command=self._add_measurement_column,
            width=80,
            state="disabled"
        )
        self.add_measurement_btn.pack(pady=2)
        
        self.clear_measurement_btn = ctk.CTkButton(
            measurement_btn_frame,
            text="Limpar",
            command=self._clear_measurement_columns,
            width=80,
            state="disabled"
        )
        self.clear_measurement_btn.pack(pady=2)
        
        # Tolerância (opcional)
        ctk.CTkLabel(config_grid, text="Tolerância (USL-LSL):").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.tolerance_entry = ctk.CTkEntry(config_grid, width=200, placeholder_text="Opcional")
        self.tolerance_entry.grid(row=3, column=1, padx=5, pady=5)
        
        # Combo para selecionar coluna de medição
        ctk.CTkLabel(config_grid, text="Selecionar Coluna:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.select_measurement_combo = ctk.CTkComboBox(config_grid, values=["Selecione..."], width=200, state="disabled")
        self.select_measurement_combo.grid(row=4, column=1, padx=5, pady=5)
        
        # ========== BOTÕES DE AÇÃO ==========
        action_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        action_frame.pack(fill="x", pady=(0, 15))
        
        ctk.CTkButton(
            action_frame,
            text="🔬 Analisar Gage R&R",
            command=self._analyze_gage_rr,
            width=200,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#1f538d"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            action_frame,
            text="💾 Exportar Relatório",
            command=self._export_report,
            width=180,
            height=40
        ).pack(side="left", padx=5)
        
        # ========== SEÇÃO DE RESULTADOS ==========
        results_label = ctk.CTkLabel(
            main_frame,
            text="📈 Resultados da Análise",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        results_label.pack(pady=(10, 10), padx=0, anchor="w")
        
        # Frame para resultados (usa o scroll do main_frame)
        self.results_scroll = ctk.CTkFrame(main_frame, fg_color="transparent")
        self.results_scroll.pack(fill="both", expand=True, pady=(0, 0))
    
    def _load_from_dataframe(self, df):
        """Carrega dados a partir de um DataFrame já fornecido"""
        try:
            self.data = df.copy()
            
            print(f"Dados carregados: {self.data.shape}")
            print(f"Colunas: {list(self.data.columns)}")
            
            # Atualiza combos
            columns = list(self.data.columns)
            self.part_col_combo.configure(values=columns, state="normal")
            self.part_col_combo.set(columns[0] if columns else "Selecione...")
            
            self.operator_col_combo.configure(values=columns, state="normal")
            self.operator_col_combo.set(columns[1] if len(columns) > 1 else "Selecione...")
            
            self.select_measurement_combo.configure(values=columns, state="normal")
            self.select_measurement_combo.set("Selecione...")
            
            self.add_measurement_btn.configure(state="normal")
            self.clear_measurement_btn.configure(state="normal")
            self.measurement_listbox.configure(state="normal")
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"Erro ao carregar dados:\n{error_msg}")
            messagebox.showerror("Erro", f"Erro ao carregar dados:\n{str(e)}")
    
    def _add_measurement_column(self):
        """Adiciona coluna de medição à lista"""
        col = self.select_measurement_combo.get()
        if col and col != "Selecione..." and col not in self.measurement_cols:
            self.measurement_cols.append(col)
            self._update_measurement_listbox()
    
    def _clear_measurement_columns(self):
        """Limpa lista de colunas de medição"""
        self.measurement_cols = []
        self._update_measurement_listbox()
    
    def _update_measurement_listbox(self):
        """Atualiza listbox de medições"""
        self.measurement_listbox.delete("1.0", "end")
        if self.measurement_cols:
            text = "\n".join(self.measurement_cols)
            self.measurement_listbox.insert("1.0", text)
    
    def _analyze_gage_rr(self):
        """Executa análise Gage R&R"""
        # Validações
        if self.data is None:
            messagebox.showwarning("Atenção", "Carregue um arquivo primeiro!")
            return
        
        part_col = self.part_col_combo.get()
        operator_col = self.operator_col_combo.get()
        
        if not part_col or part_col == "Selecione...":
            messagebox.showwarning("Atenção", "Selecione a coluna de Peças!")
            return
        
        if not operator_col or operator_col == "Selecione...":
            messagebox.showwarning("Atenção", "Selecione a coluna de Operadores!")
            return
        
        if not self.measurement_cols:
            messagebox.showwarning("Atenção", "Adicione ao menos uma coluna de medição!")
            return
        
        try:
            # Prepara dados
            from src.analytics.msa.gage_rr_utils import prepare_gage_rr_data, calculate_gage_rr
            
            prepared_data = prepare_gage_rr_data(
                self.data,
                part_col,
                operator_col,
                self.measurement_cols
            )
            
            # Pega tolerância se fornecida
            tolerance = None
            tolerance_text = self.tolerance_entry.get().strip()
            if tolerance_text:
                try:
                    tolerance = float(tolerance_text)
                except:
                    messagebox.showwarning("Atenção", "Tolerância deve ser um número!")
                    return
            
            # Calcula Gage R&R
            self.results = calculate_gage_rr(
                prepared_data,
                'Part',
                'Operator',
                'Measurement',
                tolerance=tolerance
            )
            
            # Salva informações
            self.part_col = part_col
            self.operator_col = operator_col
            self.tolerance = tolerance
            
            # Exibe resultados
            self._display_results()
            
            # Mantém foco na janela
            self.lift()
            self.focus_force()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na análise:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def _display_results(self):
        """Exibe todos os resultados da análise"""
        # Limpa resultados anteriores
        for widget in self.results_scroll.winfo_children():
            widget.destroy()
        
        if not self.results:
            return
        
        # ========== RESUMO ==========
        self._display_summary()
        
        # ========== COMPONENTES DE VARIÂNCIA ==========
        self._display_variance_components()
        
        # ========== TABELA ANOVA ==========
        self._display_anova_table()
        
        # ========== GRÁFICOS ==========
        self._display_charts()
    
    def _display_summary(self):
        """Exibe resumo da análise"""
        summary_frame = ctk.CTkFrame(self.results_scroll)
        summary_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(
            summary_frame,
            text="📊 Resumo da Análise",
            font=("Arial", 14, "bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        summary = self.results['summary']
        
        # Tabela de resumo
        table_frame = ctk.CTkFrame(summary_frame, fg_color="#2b2b2b")
        table_frame.pack(fill="x", padx=10, pady=5)
        
        info = [
            ("Número de Peças", summary['n_parts']),
            ("Número de Operadores", summary['n_operators']),
            ("Número de Tentativas", summary['n_trials']),
            ("Total de Medições", summary['n_measurements']),
            ("Média Geral", f"{summary['grand_average']:.4f}"),
            ("", ""),
            ("Gage R&R (%SV)", f"{summary['grr_percent']:.2f}%"),
            ("Status", summary['grr_interpretation']),
            ("", ""),
            ("Categorias Distintas (NDC)", summary['ndc']),
            ("Status NDC", summary['ndc_interpretation']),
        ]
        
        for i, (label, value) in enumerate(info):
            if label == "":
                # Linha separadora
                separator = ctk.CTkFrame(table_frame, height=1, fg_color="#444444")
                separator.grid(row=i, column=0, columnspan=2, sticky="ew", padx=5, pady=2)
            else:
                # Determina cor com base no status
                text_color = "white"
                if "Status" in label:
                    if summary['grr_status'] == 'accept':
                        text_color = "#4ade80"  # Verde
                    elif summary['grr_status'] == 'marginal':
                        text_color = "#fbbf24"  # Amarelo
                    else:
                        text_color = "#f87171"  # Vermelho
                
                ctk.CTkLabel(
                    table_frame,
                    text=label,
                    anchor="w",
                    width=200
                ).grid(row=i, column=0, sticky="w", padx=10, pady=3)
                
                ctk.CTkLabel(
                    table_frame,
                    text=str(value),
                    anchor="e",
                    width=150,
                    text_color=text_color
                ).grid(row=i, column=1, sticky="e", padx=10, pady=3)
    
    def _display_variance_components(self):
        """Exibe tabela de componentes de variância"""
        vc_frame = ctk.CTkFrame(self.results_scroll)
        vc_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(
            vc_frame,
            text="📐 Componentes de Variância",
            font=("Arial", 14, "bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        vc = self.results['variance_components']
        
        # Cria tabela
        table = self._create_compact_table(
            vc_frame,
            headers=list(vc.keys()),
            data=[vc[key] for key in vc.keys()]
        )
        table.pack(fill="x", padx=10, pady=5)
    
    def _display_anova_table(self):
        """Exibe tabela ANOVA"""
        anova_frame = ctk.CTkFrame(self.results_scroll)
        anova_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(
            anova_frame,
            text="📊 Tabela ANOVA",
            font=("Arial", 14, "bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        anova = self.results['anova_table']
        
        # Cria tabela
        table = self._create_compact_table(
            anova_frame,
            headers=list(anova.keys()),
            data=[anova[key] for key in anova.keys()]
        )
        table.pack(fill="x", padx=10, pady=5)
    
    def _create_compact_table(self, parent, headers, data):
        """Cria tabela compacta no estilo ProSigma"""
        table_frame = ctk.CTkFrame(parent, fg_color="#2b2b2b")
        
        # Header
        header_frame = ctk.CTkFrame(table_frame, fg_color="#1f538d", height=30)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)
        
        n_cols = len(headers)
        for i, header in enumerate(headers):
            ctk.CTkLabel(
                header_frame,
                text=header,
                font=("Arial", 11, "bold"),
                text_color="white"
            ).place(relx=i/n_cols, rely=0, relwidth=1/n_cols, relheight=1)
        
        # Rows
        n_rows = len(data[0]) if data else 0
        for row_idx in range(n_rows):
            row_frame = ctk.CTkFrame(table_frame, fg_color="#2b2b2b", height=25)
            row_frame.pack(fill="x")
            row_frame.pack_propagate(False)
            
            for col_idx, col_data in enumerate(data):
                value = col_data[row_idx]
                
                # Formata valor
                if value is None:
                    text = "-"
                elif isinstance(value, float):
                    text = f"{value:.4f}"
                else:
                    text = str(value)
                
                # Indentação para subcategorias
                if col_idx == 0 and text.startswith("  "):
                    text = "    " + text.strip()
                
                ctk.CTkLabel(
                    row_frame,
                    text=text,
                    font=("Arial", 10),
                    text_color="white"
                ).place(relx=col_idx/n_cols, rely=0, relwidth=1/n_cols, relheight=1)
        
        return table_frame
    
    def _display_charts(self):
        """Exibe gráfico de componentes de variação"""
        charts_frame = ctk.CTkFrame(self.results_scroll)
        charts_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(
            charts_frame,
            text="📈 Componentes de Variação",
            font=("Arial", 14, "bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        # Cria figura com apenas 1 gráfico
        fig = self.plt.figure(figsize=(12, 6))
        
        # Components of Variation
        self._plot_components_variation(fig.add_subplot(1, 1, 1))
        
        fig.tight_layout()
        
        # Adiciona à interface
        canvas = self.FigureCanvasTkAgg(fig, charts_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=5)
    
    def _plot_range_chart(self, ax):
        """Plota gráfico de Range por Operador"""
        pivot_data = self.results['pivot_data']
        summary = self.results['summary']
        
        operators = sorted(set(d['Operator'] for d in pivot_data))
        
        for operator in operators:
            op_data = [d for d in pivot_data if d['Operator'] == operator]
            ranges = [d['Range'] for d in op_data]
            parts = [d['Part'] for d in op_data]
            
            ax.plot(parts, ranges, marker='o', label=operator)
        
        ax.axhline(y=summary['avg_range'], color='green', linestyle='--', label='R̄')
        ax.axhline(y=summary['UCL_range'], color='red', linestyle='--', label='UCL')
        if summary['LCL_range'] > 0:
            ax.axhline(y=summary['LCL_range'], color='red', linestyle='--', label='LCL')
        
        ax.set_title('Range Chart por Operador')
        ax.set_xlabel('Peça')
        ax.set_ylabel('Range')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_average_chart(self, ax):
        """Plota gráfico de Média por Operador"""
        pivot_data = self.results['pivot_data']
        
        operators = sorted(set(d['Operator'] for d in pivot_data))
        
        for operator in operators:
            op_data = [d for d in pivot_data if d['Operator'] == operator]
            averages = [d['Average'] for d in op_data]
            parts = [d['Part'] for d in op_data]
            
            ax.plot(parts, averages, marker='s', label=operator)
        
        ax.set_title('Average Chart por Operador')
        ax.set_xlabel('Peça')
        ax.set_ylabel('Média')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_xbar_chart(self, ax):
        """Plota X-bar Chart"""
        pivot_data = self.results['pivot_data']
        summary = self.results['summary']
        
        # Média por peça
        parts = sorted(set(d['Part'] for d in pivot_data))
        part_averages = []
        for part in parts:
            part_data = [d['Average'] for d in pivot_data if d['Part'] == part]
            part_averages.append(self.np.mean(part_data))
        
        ax.plot(parts, part_averages, marker='o', color='blue')
        ax.axhline(y=summary['grand_average'], color='green', linestyle='--', label='X̄̄')
        ax.axhline(y=summary['UCL_xbar'], color='red', linestyle='--', label='UCL')
        ax.axhline(y=summary['LCL_xbar'], color='red', linestyle='--', label='LCL')
        
        ax.set_title('X-bar Chart')
        ax.set_xlabel('Peça')
        ax.set_ylabel('Média')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_r_chart(self, ax):
        """Plota R Chart"""
        pivot_data = self.results['pivot_data']
        summary = self.results['summary']
        
        # Range por peça
        parts = sorted(set(d['Part'] for d in pivot_data))
        part_ranges = []
        for part in parts:
            part_data = [d['Range'] for d in pivot_data if d['Part'] == part]
            part_ranges.append(self.np.mean(part_data))
        
        ax.plot(parts, part_ranges, marker='o', color='orange')
        ax.axhline(y=summary['avg_range'], color='green', linestyle='--', label='R̄')
        ax.axhline(y=summary['UCL_range'], color='red', linestyle='--', label='UCL')
        if summary['LCL_range'] > 0:
            ax.axhline(y=summary['LCL_range'], color='red', linestyle='--', label='LCL')
        
        ax.set_title('R Chart')
        ax.set_xlabel('Peça')
        ax.set_ylabel('Range')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_components_variation(self, ax):
        """Plota componentes de variação"""
        vc = self.results['variance_components']
        
        # Pega apenas os 3 principais componentes
        components = ['  Repeatability', '  Reproducibility', 'Part-to-Part']
        values = []
        
        for comp in components:
            idx = vc['Source'].index(comp)
            values.append(vc['%StudyVar'][idx])
        
        # Remove indentação para exibição
        labels = [c.strip() for c in components]
        
        colors = ['#f87171', '#fbbf24', '#4ade80']
        bars = ax.barh(labels, values, color=colors)
        
        # Adiciona valores nas barras
        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                   f'{value:.2f}%', va='center', fontsize=9)
        
        ax.set_title('Componentes de Variação')
        ax.set_xlabel('% Study Variation')
        ax.grid(True, alpha=0.3, axis='x')
    
    def _plot_by_part(self, ax):
        """Plota medições por Peça"""
        raw_data = self.pd.DataFrame(self.results['raw_data'])
        
        parts = sorted(raw_data['Part'].unique())
        part_positions = range(len(parts))
        
        for i, part in enumerate(parts):
            part_data = raw_data[raw_data['Part'] == part]['Measurement']
            y_positions = [i] * len(part_data)
            ax.scatter(part_data, y_positions, alpha=0.6, s=50)
        
        ax.set_yticks(part_positions)
        ax.set_yticklabels(parts)
        ax.set_title('Medições por Peça')
        ax.set_xlabel('Medição')
        ax.set_ylabel('Peça')
        ax.grid(True, alpha=0.3, axis='x')
    
    def _export_report(self):
        """Exporta relatório em JSON"""
        if not self.results:
            messagebox.showwarning("Atenção", "Execute a análise primeiro!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Salvar Relatório",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Prepara dados para exportação
            export_data = {
                'configuration': {
                    'part_column': self.part_col,
                    'operator_column': self.operator_col,
                    'measurement_columns': self.measurement_cols,
                    'tolerance': self.tolerance
                },
                'summary': self.results['summary'],
                'variance_components': self.results['variance_components'],
                'anova_table': self.results['anova_table'],
                'operator_stats': self.results['operator_stats'],
                'part_stats': self.results['part_stats']
            }
            
            # Salva
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            messagebox.showinfo("Sucesso", f"Relatório exportado para:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao exportar:\n{str(e)}")
