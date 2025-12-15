"""
Generate Space Filling Experiment Window
Modal para geração de experimentos Space Filling
"""

import customtkinter as ctk
from tkinter import messagebox, filedialog
import random

from src.analytics.space_filling.space_filling_utils import (
    generate_lhs,
    generate_lhs_min,
    generate_lhs_max,
    generate_sphere_packing,
    scale_to_range
)


class GenerateExperimentWindow(ctk.CTkToplevel):
    """Janela para gerar experimentos Space Filling"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Gerar Experimento Space Filling")
        self.geometry("700x750")
        self.resizable(False, False)
        
        # Configurar como modal
        self.transient(parent)
        self.grab_set()
        
        self.factors_list = []
        
        self._build_ui()
    
    def _build_ui(self):
        """Constrói a interface"""
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Título
        ctk.CTkLabel(
            main_frame,
            text="Gerar Experimento Space Filling",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(0, 15))
        
        # === Tipo de design ===
        type_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        type_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(
            type_frame,
            text="Tipo de Design:",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(side="left", padx=(0, 10))
        
        self.type_var = ctk.StringVar(value="lhs")
        type_options = [
            ("Latin Hypercube", "lhs"),
            ("LHS Minimin", "lhsMin"),
            ("LHS Maximin", "lhsMax"),
            ("Sphere Packing", "sp")
        ]
        
        for text, value in type_options:
            ctk.CTkRadioButton(
                type_frame,
                text=text,
                variable=self.type_var,
                value=value
            ).pack(side="left", padx=5)
        
        # === Parâmetros básicos ===
        params_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        params_frame.pack(fill="x", pady=10)
        
        # Número de fatores
        factor_frame = ctk.CTkFrame(params_frame, fg_color="transparent")
        factor_frame.pack(side="left", padx=5)
        
        ctk.CTkLabel(
            factor_frame,
            text="Fatores:",
            font=ctk.CTkFont(size=11)
        ).pack()
        
        self.factors_entry = ctk.CTkEntry(factor_frame, width=80)
        self.factors_entry.pack()
        self.factors_entry.insert(0, "3")
        self.factors_entry.bind("<KeyRelease>", self._on_factors_change)
        
        # Número de rodadas
        rounds_frame = ctk.CTkFrame(params_frame, fg_color="transparent")
        rounds_frame.pack(side="left", padx=5)
        
        ctk.CTkLabel(
            rounds_frame,
            text="Rodadas:",
            font=ctk.CTkFont(size=11)
        ).pack()
        
        self.rounds_entry = ctk.CTkEntry(rounds_frame, width=80)
        self.rounds_entry.pack()
        self.rounds_entry.insert(0, "30")
        
        # Colunas de resposta
        response_frame = ctk.CTkFrame(params_frame, fg_color="transparent")
        response_frame.pack(side="left", padx=5)
        
        ctk.CTkLabel(
            response_frame,
            text="Colunas Y:",
            font=ctk.CTkFont(size=11)
        ).pack()
        
        self.response_cols_entry = ctk.CTkEntry(response_frame, width=80)
        self.response_cols_entry.pack()
        self.response_cols_entry.insert(0, "1")
        
        # === Opção de valores aleatórios para Y ===
        random_y_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        random_y_frame.pack(fill="x", pady=10)
        
        self.random_y_var = ctk.BooleanVar(value=False)
        self.random_y_check = ctk.CTkCheckBox(
            random_y_frame,
            text="Gerar valores aleatórios para Y",
            variable=self.random_y_var,
            command=self._toggle_random_y
        )
        self.random_y_check.pack(side="left")
        
        # Parâmetros para valores aleatórios
        self.random_params_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        self.random_params_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(
            self.random_params_frame,
            text="Mín:",
            font=ctk.CTkFont(size=11)
        ).pack(side="left", padx=(0, 5))
        
        self.y_min_entry = ctk.CTkEntry(self.random_params_frame, width=70)
        self.y_min_entry.pack(side="left", padx=5)
        self.y_min_entry.insert(0, "0")
        
        ctk.CTkLabel(
            self.random_params_frame,
            text="Máx:",
            font=ctk.CTkFont(size=11)
        ).pack(side="left", padx=(10, 5))
        
        self.y_max_entry = ctk.CTkEntry(self.random_params_frame, width=70)
        self.y_max_entry.pack(side="left", padx=5)
        self.y_max_entry.insert(0, "100")
        
        ctk.CTkLabel(
            self.random_params_frame,
            text="Intervalo:",
            font=ctk.CTkFont(size=11)
        ).pack(side="left", padx=(10, 5))
        
        self.y_interval_entry = ctk.CTkEntry(self.random_params_frame, width=70)
        self.y_interval_entry.pack(side="left", padx=5)
        self.y_interval_entry.insert(0, "1")
        
        self._toggle_random_y()
        
        # === Configuração de fatores ===
        ctk.CTkLabel(
            main_frame,
            text="Configuração de Fatores:",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(pady=(15, 5), anchor="w")
        
        # Frame para adicionar fator
        add_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        add_frame.pack(fill="x", pady=5)
        
        # Seletor de fator
        factor_select_frame = ctk.CTkFrame(add_frame, fg_color="transparent")
        factor_select_frame.pack(side="left", padx=5)
        
        ctk.CTkLabel(
            factor_select_frame,
            text="Fator:",
            font=ctk.CTkFont(size=11)
        ).pack()
        
        self.factor_combo = ctk.CTkComboBox(
            factor_select_frame,
            width=80,
            values=[]
        )
        self.factor_combo.pack()
        
        # Nome da coluna
        name_frame = ctk.CTkFrame(add_frame, fg_color="transparent")
        name_frame.pack(side="left", padx=5)
        
        ctk.CTkLabel(
            name_frame,
            text="Nome Coluna:",
            font=ctk.CTkFont(size=11)
        ).pack()
        
        self.column_name_entry = ctk.CTkEntry(name_frame, width=120)
        self.column_name_entry.pack()
        
        # Nível mínimo
        min_frame = ctk.CTkFrame(add_frame, fg_color="transparent")
        min_frame.pack(side="left", padx=5)
        
        ctk.CTkLabel(
            min_frame,
            text="Nível Mín:",
            font=ctk.CTkFont(size=11)
        ).pack()
        
        self.min_level_entry = ctk.CTkEntry(min_frame, width=80)
        self.min_level_entry.pack()
        self.min_level_entry.insert(0, "-1")
        
        # Nível máximo
        max_frame = ctk.CTkFrame(add_frame, fg_color="transparent")
        max_frame.pack(side="left", padx=5)
        
        ctk.CTkLabel(
            max_frame,
            text="Nível Máx:",
            font=ctk.CTkFont(size=11)
        ).pack()
        
        self.max_level_entry = ctk.CTkEntry(max_frame, width=80)
        self.max_level_entry.pack()
        self.max_level_entry.insert(0, "1")
        
        # Botão adicionar
        ctk.CTkButton(
            add_frame,
            text="➕ Adicionar",
            command=self._add_factor,
            width=100,
            fg_color="#27AE60",
            hover_color="#1E8449"
        ).pack(side="left", padx=(10, 0))
        
        # Lista de fatores configurados
        ctk.CTkLabel(
            main_frame,
            text="Fatores Configurados:",
            font=ctk.CTkFont(size=11)
        ).pack(pady=(10, 5), anchor="w")
        
        self.factors_listbox = ctk.CTkTextbox(main_frame, height=150)
        self.factors_listbox.pack(fill="both", expand=True, pady=(0, 10))
        
        # === Botões ===
        buttons_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        buttons_frame.pack(fill="x", pady=(10, 0))
        
        ctk.CTkButton(
            buttons_frame,
            text="✖ Cancelar",
            command=self.destroy,
            width=120,
            fg_color="#7F8C8D",
            hover_color="#5D6D7E"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            buttons_frame,
            text="🗑 Limpar Lista",
            command=self._clear_factors,
            width=120,
            fg_color="#E67E22",
            hover_color="#CA6F1E"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            buttons_frame,
            text="📊 Gerar Experimento",
            command=self._generate_experiment,
            width=150,
            fg_color="#27AE60",
            hover_color="#1E8449"
        ).pack(side="right", padx=5)
        
        # Inicializa combo de fatores
        self._on_factors_change(None)
    
    def _toggle_random_y(self):
        """Ativa/desativa parâmetros de valores aleatórios"""
        if self.random_y_var.get():
            for widget in self.random_params_frame.winfo_children():
                if isinstance(widget, ctk.CTkEntry):
                    widget.configure(state="normal")
        else:
            for widget in self.random_params_frame.winfo_children():
                if isinstance(widget, ctk.CTkEntry):
                    widget.configure(state="disabled")
    
    def _on_factors_change(self, event):
        """Atualiza combo de fatores quando número muda"""
        try:
            n_factors = int(self.factors_entry.get())
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            factors = [alphabet[i] for i in range(min(n_factors, 26))]
            
            # Remove fatores já adicionados
            available = [f for f in factors if not any(item['factor'] == f for item in self.factors_list)]
            
            self.factor_combo.configure(values=available)
            if available:
                self.factor_combo.set(available[0])
            
            # Atualiza número de rodadas sugerido
            rounds = n_factors * 10
            self.rounds_entry.delete(0, "end")
            self.rounds_entry.insert(0, str(rounds))
        except:
            pass
    
    def _add_factor(self):
        """Adiciona fator à lista"""
        try:
            factor = self.factor_combo.get()
            column_name = self.column_name_entry.get().strip()
            min_level = float(self.min_level_entry.get())
            max_level = float(self.max_level_entry.get())
            
            if not factor:
                messagebox.showwarning("Aviso", "Selecione um fator")
                return
            
            if min_level >= max_level:
                messagebox.showerror("Erro", "Nível mínimo deve ser menor que máximo")
                return
            
            # Nome padrão = letra do fator
            if not column_name:
                column_name = factor
            
            self.factors_list.append({
                'factor': factor,
                'columnName': column_name,
                'minLevel': min_level,
                'maxLevel': max_level
            })
            
            self._update_factors_display()
            self._on_factors_change(None)
            
            # Limpa campos
            self.column_name_entry.delete(0, "end")
            self.min_level_entry.delete(0, "end")
            self.min_level_entry.insert(0, "-1")
            self.max_level_entry.delete(0, "end")
            self.max_level_entry.insert(0, "1")
            
        except ValueError:
            messagebox.showerror("Erro", "Valores inválidos para níveis")
    
    def _update_factors_display(self):
        """Atualiza exibição da lista de fatores"""
        self.factors_listbox.delete("1.0", "end")
        for item in self.factors_list:
            text = f"{item['factor']} → {item['columnName']} | Mín: {item['minLevel']} | Máx: {item['maxLevel']}\n"
            self.factors_listbox.insert("end", text)
    
    def _clear_factors(self):
        """Limpa lista de fatores"""
        self.factors_list = []
        self._update_factors_display()
        self._on_factors_change(None)
    
    def _generate_experiment(self):
        """Gera o experimento"""
        try:
            # Valida
            n_factors = int(self.factors_entry.get())
            n_rounds = int(self.rounds_entry.get())
            n_response = int(self.response_cols_entry.get())
            design_type = self.type_var.get()
            
            if n_factors < 1:
                messagebox.showerror("Erro", "Número de fatores deve ser >= 1")
                return
            
            if n_rounds < n_factors:
                messagebox.showerror("Erro", f"Número de rodadas deve ser >= {n_factors}")
                return
            
            if n_response < 1:
                messagebox.showerror("Erro", "Número de colunas de resposta deve ser >= 1")
                return
            
            # Gera valores aleatórios para Y se necessário
            response_values = []
            if self.random_y_var.get():
                y_min = float(self.y_min_entry.get())
                y_max = float(self.y_max_entry.get())
                y_interval = float(self.y_interval_entry.get())
                
                if y_interval <= 0:
                    y_interval = 1
                
                # Gera valores no intervalo
                possible_values = []
                current = y_min
                while current <= y_max:
                    possible_values.append(current)
                    current += y_interval
                
                response_values = [random.choice(possible_values) for _ in range(n_rounds)]
            
            # Gera design
            if design_type == "lhs":
                data = generate_lhs(n_rounds, n_factors, response_values, n_response)
            elif design_type == "lhsMin":
                data = generate_lhs_min(n_rounds, n_factors, 5, response_values, n_response)
            elif design_type == "lhsMax":
                data = generate_lhs_max(n_rounds, n_factors, 5, response_values, n_response)
            elif design_type == "sp":
                data = generate_sphere_packing(n_factors, n_rounds, n_response, response_values)
            
            # Aplica escala aos fatores configurados
            pd = __import__('pandas')
            df = pd.DataFrame(data)
            
            for item in self.factors_list:
                factor = item['factor']
                col_name = item['columnName']
                min_level = item['minLevel']
                max_level = item['maxLevel']
                
                # Identifica coluna do fator (X1, X2, etc)
                factor_idx = ord(factor) - ord('A')
                original_col = f"X{factor_idx + 1}"
                
                if original_col in df.columns:
                    # Escala de [0,1] para [min, max]
                    df[original_col] = scale_to_range(df[original_col].values, min_level, max_level)
                    
                    # Renomeia coluna
                    if col_name != original_col:
                        df.rename(columns={original_col: col_name}, inplace=True)
            
            # Salva em Excel
            file_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                initialfile="space_filling_experiment.xlsx"
            )
            
            if file_path:
                df.to_excel(file_path, index=False)
                messagebox.showinfo("Sucesso", f"Experimento gerado com sucesso!\n{n_rounds} rodadas, {n_factors} fatores")
                self.destroy()
        
        except ValueError as e:
            messagebox.showerror("Erro", f"Valores inválidos: {str(e)}")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao gerar experimento: {str(e)}")
