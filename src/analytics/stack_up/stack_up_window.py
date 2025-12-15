"""
Stack-Up Analysis Window
Interface principal para análise de empilhamento de tolerâncias usando customtkinter
"""

import customtkinter as ctk
from tkinter import ttk, messagebox, filedialog
from src.utils.lazy_imports import get_pandas

from src.analytics.stack_up.stack_up_utils import (
    calculate_stack_up,
    validate_factors
)


class StackUpWindow(ctk.CTkToplevel):
    """Janela principal para análise de Stack-Up"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Análise de Empilhamento de Tolerâncias (Stack-Up)")
        self.geometry("1400x800")
        self.minsize(1200, 700)
        
        # Maximizar janela
        try:
            self.state("zoomed")
        except Exception:
            pass
        
        # Configurar como modal
        self.transient(parent)
        self.grab_set()
        
        self.characteristics = []  # Lista de dicionários com dados das características
        self.results = None
        
        self._build_ui()
    
    def _build_ui(self):
        """Constrói a interface"""
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=16, pady=16)
        
        # ===== PAINEL ESQUERDO =====
        left_container = ctk.CTkFrame(main_frame, width=600)
        left_container.pack(side="left", fill="both", expand=True)
        left_container.pack_propagate(False)
        
        left_panel = ctk.CTkScrollableFrame(left_container)
        left_panel.pack(fill="both", expand=True)
        
        # Título
        ctk.CTkLabel(
            left_panel,
            text="Stack-Up Analysis",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(12, 4))
        
        ctk.CTkLabel(
            left_panel,
            text="Empilhamento de Tolerâncias",
            text_color="gray"
        ).pack(pady=(0, 20))
        
        # Configurações gerais
        config_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
        config_frame.pack(fill="x", padx=12, pady=(0, 12))
        
        ctk.CTkLabel(
            config_frame,
            text="Configurações",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", pady=(0, 8))
        
        # Número de características
        chars_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        chars_frame.pack(fill="x", pady=(0, 8))
        
        ctk.CTkLabel(
            chars_frame,
            text="Número de Características:",
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=(0, 8))
        
        self.num_chars_entry = ctk.CTkEntry(chars_frame, width=80)
        self.num_chars_entry.pack(side="left")
        self.num_chars_entry.insert(0, "0")
        
        # Número de rodadas
        rounds_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        rounds_frame.pack(fill="x", pady=(0, 8))
        
        ctk.CTkLabel(
            rounds_frame,
            text="Número de Rodadas:",
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=(0, 8))
        
        self.num_rounds_entry = ctk.CTkEntry(rounds_frame, width=100)
        self.num_rounds_entry.pack(side="left")
        self.num_rounds_entry.insert(0, "5000")
        
        # Botão gerar características
        ctk.CTkButton(
            config_frame,
            text="🔄 Gerar Características",
            command=self._generate_characteristics,
            height=36,
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(fill="x", pady=(8, 0))
        
        # Botões de arquivo
        file_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
        file_frame.pack(fill="x", padx=12, pady=(12, 12))
        
        ctk.CTkButton(
            file_frame,
            text="📥 Baixar Modelo Padrão",
            command=self._download_template,
            height=32
        ).pack(fill="x", pady=(0, 4))
        
        ctk.CTkButton(
            file_frame,
            text="📂 Importar Arquivo",
            command=self._import_file,
            height=32
        ).pack(fill="x")
        
        # Área de características
        ctk.CTkLabel(
            left_panel,
            text="Características",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=12, pady=(12, 8))
        
        self.chars_container = ctk.CTkScrollableFrame(left_panel, height=300)
        self.chars_container.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        
        # Botões de ação
        action_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
        action_frame.pack(fill="x", padx=12, pady=(0, 12))
        
        ctk.CTkButton(
            action_frame,
            text="🔢 Calcular",
            command=self._calculate,
            height=40,
            fg_color="#2ECC71",
            hover_color="#27AE60",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(fill="x", pady=(0, 4))
        
        ctk.CTkButton(
            action_frame,
            text="💾 Exportar Tabela de Dados",
            command=self._export_data,
            height=36
        ).pack(fill="x", pady=(0, 4))
        
        ctk.CTkButton(
            action_frame,
            text="🔄 Limpar Tudo",
            command=self._clear_all,
            height=36,
            fg_color="#E74C3C",
            hover_color="#C0392B"
        ).pack(fill="x")
        
        # ===== PAINEL DIREITO - RESULTADOS =====
        right_container = ctk.CTkFrame(main_frame)
        right_container.pack(side="right", fill="both", expand=True, padx=(12, 0))
        
        right_panel = ctk.CTkScrollableFrame(right_container)
        right_panel.pack(fill="both", expand=True)
        
        ctk.CTkLabel(
            right_panel,
            text="Resultados",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(12, 20))
        
        # Resumo da distribuição
        ctk.CTkLabel(
            right_panel,
            text="Resumo da Distribuição",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=12, pady=(0, 8))
        
        # Frame para tabela
        table_frame = ctk.CTkFrame(right_panel, fg_color="white")
        table_frame.pack(fill="both", padx=12, pady=(0, 20))
        
        self.summary_tree = ttk.Treeview(
            table_frame,
            columns=("Termo", "Média", "Desvio Padrão"),
            show="headings",
            height=10
        )
        self.summary_tree.heading("Termo", text="Termo")
        self.summary_tree.heading("Média", text="Média")
        self.summary_tree.heading("Desvio Padrão", text="Desvio Padrão")
        
        self.summary_tree.column("Termo", width=150)
        self.summary_tree.column("Média", width=120)
        self.summary_tree.column("Desvio Padrão", width=120)
        
        self.summary_tree.pack(fill="both", expand=True, padx=4, pady=4)
        
        # Equação
        ctk.CTkLabel(
            right_panel,
            text="Equação",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="black"
        ).pack(anchor="w", padx=12, pady=(12, 8))
        
        equation_frame = ctk.CTkFrame(right_panel, fg_color="#F5F5F5")
        equation_frame.pack(fill="x", padx=12, pady=(0, 12))
        
        self.equation_label = ctk.CTkLabel(
            equation_frame,
            text="Y = ",
            font=ctk.CTkFont(family="Courier New", size=12),
            wraplength=500,
            justify="left",
            text_color="black"
        )
        self.equation_label.pack(fill="x", padx=10, pady=10)
    
    def _generate_characteristics(self):
        """Gera os campos para entrada de características"""
        try:
            num_chars = int(self.num_chars_entry.get())
            
            if num_chars < 1 or num_chars > 50:
                messagebox.showerror("Erro", "Número de características deve estar entre 1 e 50")
                return
            
            # Limpa características existentes
            for widget in self.chars_container.winfo_children():
                widget.destroy()
            
            self.characteristics = []
            
            # Cria widgets para cada característica
            for i in range(num_chars):
                char_frame = self._create_characteristic_widget(i)
                char_frame.pack(fill="x", pady=4, padx=4)
                
                # Adiciona aos dados
                self.characteristics.append({
                    'name': f'Característica_{chr(65 + i)}',
                    'min': 0.0,
                    'max': 0.0,
                    'sensitivity': 1.0,
                    'quota': '1',
                    'widgets': char_frame
                })
            
            # Atualiza a pré-visualização da equação
            self._update_equation_preview()
            
            messagebox.showinfo("Sucesso", f"{num_chars} características geradas!")
            
        except ValueError:
            messagebox.showerror("Erro", "Digite um número válido de características")
    
    def _create_characteristic_widget(self, index):
        """Cria widget para uma característica"""
        main_frame = ctk.CTkFrame(self.chars_container)
        
        # Título
        title_label = ctk.CTkLabel(
            main_frame,
            text=f"Característica {index + 1}",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        title_label.pack(anchor="w", padx=8, pady=(8, 4))
        
        # Nome
        name_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        name_frame.pack(fill="x", padx=8, pady=2)
        
        ctk.CTkLabel(name_frame, text="Nome:", width=100).pack(side="left")
        name_entry = ctk.CTkEntry(name_frame)
        name_entry.pack(side="left", fill="x", expand=True)
        name_entry.insert(0, f"Característica_{chr(65 + index)}")
        name_entry.bind("<KeyRelease>", lambda e: self._update_equation_preview())
        
        # Valores
        values_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        values_frame.pack(fill="x", padx=8, pady=2)
        
        # Min
        ctk.CTkLabel(values_frame, text="Mín:", width=50).pack(side="left")
        min_entry = ctk.CTkEntry(values_frame, width=80)
        min_entry.pack(side="left", padx=(0, 8))
        min_entry.insert(0, "0.0")
        
        # Max
        ctk.CTkLabel(values_frame, text="Máx:", width=50).pack(side="left")
        max_entry = ctk.CTkEntry(values_frame, width=80)
        max_entry.pack(side="left", padx=(0, 8))
        max_entry.insert(0, "0.0")
        
        # Sensibilidade
        ctk.CTkLabel(values_frame, text="Sens:", width=50).pack(side="left")
        sens_entry = ctk.CTkEntry(values_frame, width=80)
        sens_entry.pack(side="left", padx=(0, 8))
        sens_entry.insert(0, "1.0")
        sens_entry.bind("<KeyRelease>", lambda e: self._update_equation_preview())
        
        # Quota
        ctk.CTkLabel(values_frame, text="Quota:", width=50).pack(side="left")
        quota_combo = ctk.CTkComboBox(
            values_frame,
            values=["Standard (1)", "CTS (1.33)", "CTQ (2)"],
            width=120,
            state="readonly",
            command=lambda _: self._update_equation_preview()
        )
        quota_combo.pack(side="left")
        quota_combo.set("Standard (1)")
        
        # Armazena referências
        main_frame.name_entry = name_entry
        main_frame.min_entry = min_entry
        main_frame.max_entry = max_entry
        main_frame.sens_entry = sens_entry
        main_frame.quota_combo = quota_combo
        main_frame.index = index
        
        return main_frame
    
    def _update_equation_preview(self):
        """Atualiza a pré-visualização da equação baseada nas cotas e sensibilidades"""
        if not self.chars_container.winfo_children():
            self.equation_label.configure(text="Y = ")
            return
        
        quota_map = {
            "Standard (1)": "1",
            "CTS (1.33)": "1.33",
            "CTQ (2)": "2"
        }
        
        equation_parts = []
        
        for widget in self.chars_container.winfo_children():
            try:
                name = widget.name_entry.get().strip() or f'Char_{widget.index + 1}'
                sens_val = float(widget.sens_entry.get() or 1.0)
                quota_str = widget.quota_combo.get()
                quota_val = float(quota_map[quota_str])
                
                # Calcula o coeficiente (sensibilidade * quota)
                coef = sens_val * quota_val
                
                if coef >= 0:
                    sign = '+' if equation_parts else ''
                    equation_parts.append(f"{sign} {coef:.2f}*{name}")
                else:
                    equation_parts.append(f"- {abs(coef):.2f}*{name}")
            except (ValueError, AttributeError):
                continue
        
        if equation_parts:
            equation = ' '.join(equation_parts)
            self.equation_label.configure(text=f"Y = {equation}")
        else:
            self.equation_label.configure(text="Y = ")
    
    def _get_characteristics_data(self):
        """Obtém dados das características dos widgets"""
        factors = {}
        quota_map = {
            "Standard (1)": "1",
            "CTS (1.33)": "1.33",
            "CTQ (2)": "2"
        }
        
        for widget in self.chars_container.winfo_children():
            try:
                name = widget.name_entry.get().strip()
                min_val = float(widget.min_entry.get())
                max_val = float(widget.max_entry.get())
                sens_val = float(widget.sens_entry.get())
                quota_str = widget.quota_combo.get()
                quota_val = quota_map[quota_str]
                
                factors[f'factor_{widget.index}'] = {
                    'name': name,
                    'min': min_val,
                    'max': max_val,
                    'sensitivity': sens_val,
                    'quota': quota_val
                }
            except (ValueError, AttributeError) as e:
                messagebox.showerror("Erro", f"Erro ao ler dados da característica {widget.index + 1}: {str(e)}")
                return None
        
        return factors
    
    def _calculate(self):
        """Calcula o stack-up"""
        if not self.chars_container.winfo_children():
            messagebox.showwarning("Aviso", "Gere as características primeiro")
            return
        
        # Obtém dados
        factors = self._get_characteristics_data()
        if not factors:
            return
        
        # Valida
        is_valid, error_msg = validate_factors(factors)
        if not is_valid:
            messagebox.showerror("Erro de Validação", error_msg)
            return
        
        try:
            rounds = int(self.num_rounds_entry.get())
            
            if rounds < 100 or rounds > 250000:
                messagebox.showerror("Erro", "Número de rodadas deve estar entre 100 e 250.000")
                return
            
            # Calcula
            self.results = calculate_stack_up(rounds, factors)
            
            # Atualiza interface
            self._update_results()
            
            messagebox.showinfo("Sucesso", "Cálculo realizado com sucesso!")
            
        except ValueError:
            messagebox.showerror("Erro", "Digite um número válido de rodadas")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao calcular: {str(e)}")
    
    def _update_results(self):
        """Atualiza a exibição dos resultados"""
        if not self.results:
            return
        
        # Limpa tabela
        for item in self.summary_tree.get_children():
            self.summary_tree.delete(item)
        
        # Atualiza tabela
        means = self.results['means']
        stds = self.results['stds']
        
        for key in means.keys():
            self.summary_tree.insert(
                "",
                "end",
                values=(key, f"{means[key]:.4f}", f"{stds[key]:.6f}")
            )
        
        # Atualiza equação
        equation = self.results['equation']
        self.equation_label.configure(text=f"Y = {equation}")
    
    def _export_data(self):
        """Exporta a tabela de dados"""
        if not self.results:
            messagebox.showwarning("Aviso", "Calcule os resultados primeiro")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx"), ("CSV Files", "*.csv")],
            initialfile="stack_up_data.xlsx"
        )
        
        if file_path:
            try:
                df = self.results['dataframe']
                
                if file_path.endswith('.csv'):
                    df.to_csv(file_path, index=False)
                else:
                    df.to_excel(file_path, index=False)
                
                messagebox.showinfo("Sucesso", f"Dados exportados para:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao exportar: {str(e)}")
    
    def _download_template(self):
        """Baixa arquivo modelo"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx")],
            initialfile="stack_up_template.xlsx"
        )
        
        if file_path:
            try:
                pd = get_pandas()
                template_data = {
                    'Característica': [''],
                    'Valor Mínimo': [0.0],
                    'Valor Máximo': [0.0],
                    'Sensibilidade': [1.0],
                    'Quota': ['Standard']
                }
                df = pd.DataFrame(template_data)
                df.to_excel(file_path, index=False)
                
                messagebox.showinfo("Sucesso", f"Modelo salvo em:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao criar modelo: {str(e)}")
    
    def _import_file(self):
        """Importa dados de arquivo"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Excel/CSV Files", "*.xlsx *.csv"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            pd = get_pandas()
            # Lê arquivo
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            quota_mapping = {
                'Standard': '1',
                'CTS': '1.33',
                'CTQ': '2'
            }
            
            # Atualiza número de características
            num_chars = len(df)
            self.num_chars_entry.delete(0, 'end')
            self.num_chars_entry.insert(0, str(num_chars))
            self._generate_characteristics()
            
            # Preenche dados
            for i, (_, row) in enumerate(df.iterrows()):
                if i < len(self.chars_container.winfo_children()):
                    widget = list(self.chars_container.winfo_children())[i]
                    
                    widget.name_entry.delete(0, 'end')
                    widget.name_entry.insert(0, str(row.get('Característica', row.get('Characteristic', f'Char_{i}'))))
                    
                    widget.min_entry.delete(0, 'end')
                    widget.min_entry.insert(0, str(float(row.get('Valor Mínimo', row.get('Min Value', 0.0)))))
                    
                    widget.max_entry.delete(0, 'end')
                    widget.max_entry.insert(0, str(float(row.get('Valor Máximo', row.get('Max Value', 0.0)))))
                    
                    widget.sens_entry.delete(0, 'end')
                    widget.sens_entry.insert(0, str(float(row.get('Sensibilidade', row.get('Sensitivity', 1.0)))))
                    
                    quota_val = str(row.get('Quota', 'Standard'))
                    if quota_val in ['1', 'Standard']:
                        widget.quota_combo.set("Standard (1)")
                    elif quota_val in ['1.33', 'CTS']:
                        widget.quota_combo.set("CTS (1.33)")
                    elif quota_val in ['2', 'CTQ']:
                        widget.quota_combo.set("CTQ (2)")
            
            messagebox.showinfo("Sucesso", f"Arquivo importado!\n{num_chars} características carregadas.")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao importar arquivo:\n{str(e)}")
    
    def _clear_all(self):
        """Limpa todos os dados"""
        if messagebox.askyesno("Confirmar", "Tem certeza que deseja limpar todos os dados?"):
            # Limpa características
            for widget in self.chars_container.winfo_children():
                widget.destroy()
            
            self.characteristics = []
            
            # Reset campos
            self.num_chars_entry.delete(0, 'end')
            self.num_chars_entry.insert(0, "0")
            
            self.num_rounds_entry.delete(0, 'end')
            self.num_rounds_entry.insert(0, "5000")
            
            # Limpa resultados
            for item in self.summary_tree.get_children():
                self.summary_tree.delete(item)
            
            self.equation_label.configure(text="Y = ")
            self.results = None
            
            messagebox.showinfo("Sucesso", "Todos os dados foram limpos")
