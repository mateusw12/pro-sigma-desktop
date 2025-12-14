"""
Monte Carlo Simulation Window
- Configure number of simulations (rows)
- Define variables with distributions and parameters
- Optional output formula
- Export results to CSV/XLSX
"""
import customtkinter as ctk
from tkinter import ttk, messagebox, filedialog
from src.utils.lazy_imports import get_pandas, get_numpy, get_scipy_stats, get_matplotlib_figure, get_matplotlib_backend
from src.utils.ui_components import create_action_button


DISTRIBUTIONS = {
    "Normal": {"params": ["mean", "std"], "scipy": stats.norm},
    "Uniform": {"params": ["min", "max"], "scipy": stats.uniform},
    "Triangular": {"params": ["left", "mode", "right"], "scipy": stats.triang},
    "Lognormal": {"params": ["mean", "std"], "scipy": stats.lognorm},
    "Exponential": {"params": ["scale"], "scipy": stats.expon},
    "Gamma": {"params": ["shape", "scale"], "scipy": stats.gamma},
    "Beta": {"params": ["alpha", "beta"], "scipy": stats.beta},
}


class MonteCarloWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Monte Carlo Simulation")
        self.geometry("1200x750")
        self.minsize(1000, 600)
        try:
            self.state("zoomed")
        except Exception:
            pass
        self.transient(parent)
        self.grab_set()

        self.variables = []  # Lista de {name, dist, params}
        self.simulated_data = None

        self._build_ui()

    def _build_ui(self):
        main = ctk.CTkFrame(self)
        main.pack(fill="both", expand=True, padx=16, pady=16)

        left_container = ctk.CTkFrame(main, width=320)
        left_container.pack(side="left", fill="y")
        left_container.pack_propagate(False)
        
        left = ctk.CTkScrollableFrame(left_container)
        left.pack(fill="both", expand=True)

        # Right panel with scroll
        right_container = ctk.CTkFrame(main)
        right_container.pack(side="right", fill="both", expand=True, padx=(12, 0))
        
        right = ctk.CTkScrollableFrame(right_container)
        right.pack(fill="both", expand=True)

        # === LEFT PANEL: Configuration ===
        ctk.CTkLabel(left, text="Monte Carlo", font=ctk.CTkFont(size=18, weight="bold"))\
            .pack(pady=(12, 4))
        ctk.CTkLabel(left, text="Configure variáveis e rode simulações", text_color="gray")\
            .pack(pady=(0, 12), padx=8)

        # Number of simulations
        sim_frame = ctk.CTkFrame(left, fg_color="transparent")
        sim_frame.pack(fill="x", padx=12, pady=(0, 8))
        ctk.CTkLabel(sim_frame, text="Número de simulações:", font=ctk.CTkFont(size=12, weight="bold"))\
            .pack(anchor="w")
        self.num_sims_entry = ctk.CTkEntry(sim_frame, placeholder_text="10000")
        self.num_sims_entry.pack(fill="x", pady=(4, 0))
        self.num_sims_entry.insert(0, "10000")

        # Variables list
        ctk.CTkLabel(left, text="Variáveis:", font=ctk.CTkFont(size=12, weight="bold"))\
            .pack(anchor="w", padx=12, pady=(8, 4))

        self.vars_scroll = ctk.CTkScrollableFrame(left, width=280, height=200)
        self.vars_scroll.pack(padx=12, pady=(0, 8), fill="both", expand=True)
        self._update_vars_list()

        btn_frame = ctk.CTkFrame(left, fg_color="transparent")
        btn_frame.pack(fill="x", padx=12, pady=(0, 8))
        
        ctk.CTkButton(btn_frame, text="+ Adicionar Variável", command=self._add_variable, height=36)\
            .pack(fill="x", pady=(0, 4))
        ctk.CTkButton(btn_frame, text="✎ Editar Selecionada", command=self._edit_variable, height=36)\
            .pack(fill="x", pady=(0, 4))
        ctk.CTkButton(btn_frame, text="✕ Remover Selecionada", command=self._remove_variable, height=36, fg_color="#E74C3C")\
            .pack(fill="x")

        # Output formula (optional)
        ctk.CTkLabel(left, text="Fórmula de saída (opcional):", font=ctk.CTkFont(size=11, weight="bold"))\
            .pack(anchor="w", padx=12, pady=(8, 2))
        self.formula_entry = ctk.CTkEntry(left, placeholder_text="Ex: X + Y * 2")
        self.formula_entry.pack(fill="x", padx=12, pady=(0, 12))

        # Botão padronizado
        self.run_btn = create_action_button(
            left,
            text="Rodar Simulação",
            command=self._run_simulation,
            icon="🎲"
        )
        self.run_btn.pack(fill="x", padx=12, pady=(4, 8))

        # Export buttons
        export_frame = ctk.CTkFrame(left, fg_color="transparent")
        export_frame.pack(fill="x", padx=12, pady=(0, 14))
        
        self.export_csv_btn = ctk.CTkButton(
            export_frame,
            text="💾 CSV",
            command=lambda: self._export_data("csv"),
            height=40,
            font=ctk.CTkFont(size=12, weight="bold"),
            state="disabled"
        )
        self.export_csv_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))
        
        self.export_xlsx_btn = ctk.CTkButton(
            export_frame,
            text="💾 Excel",
            command=lambda: self._export_data("xlsx"),
            height=40,
            font=ctk.CTkFont(size=12, weight="bold"),
            state="disabled"
        )
        self.export_xlsx_btn.pack(side="right", fill="x", expand=True)
        
        # Bottom spacer for footer margin
        ctk.CTkFrame(left, fg_color="transparent", height=20).pack()

        # === RIGHT PANEL: Results ===
        # Stats table
        stats_label = ctk.CTkLabel(right, text="Estatísticas das Simulações", font=ctk.CTkFont(size=14, weight="bold"))
        stats_label.pack(pady=(8, 4), padx=8, anchor="w")

        self.stats_frame = ctk.CTkFrame(right)
        self.stats_frame.pack(fill="x", padx=8, pady=(0, 8))
        
        cols = ["Variável", "Média", "Desvio", "Min", "Q25", "Mediana", "Q75", "Max"]
        self.stats_tree = ttk.Treeview(self.stats_frame, columns=cols, show="headings", height=6)
        for col in cols:
            self.stats_tree.heading(col, text=col)
            width = 140 if col == "Variável" else 90
            self.stats_tree.column(col, anchor="w", stretch=True, width=width)
        self.stats_tree.pack(fill="x", padx=6, pady=6)

        # Plots
        plot_label = ctk.CTkLabel(right, text="Histogramas", font=ctk.CTkFont(size=14, weight="bold"))
        plot_label.pack(pady=(8, 4), padx=8, anchor="w")

        self.plot_frame = ctk.CTkFrame(right)
        self.plot_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        
        self.figure = Figure(figsize=(10, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.figure.tight_layout()

        self.selected_var_idx = None

    def _update_vars_list(self):
        for w in self.vars_scroll.winfo_children():
            w.destroy()

        if not self.variables:
            ctk.CTkLabel(self.vars_scroll, text="Nenhuma variável definida", text_color="gray")\
                .pack(pady=20)
            return

        for idx, var in enumerate(self.variables):
            var_frame = ctk.CTkFrame(self.vars_scroll, fg_color="#2b2b2b", corner_radius=6)
            var_frame.pack(fill="x", pady=4, padx=4)
            
            info_text = f"{var['name']} ~ {var['dist']}"
            params_text = ", ".join([f"{k}={v}" for k, v in var['params'].items()])
            
            ctk.CTkLabel(var_frame, text=info_text, font=ctk.CTkFont(size=12, weight="bold"))\
                .pack(anchor="w", padx=10, pady=(8, 2))
            ctk.CTkLabel(var_frame, text=params_text, font=ctk.CTkFont(size=10), text_color="gray")\
                .pack(anchor="w", padx=10, pady=(0, 8))

            # Click to select
            def make_selector(i):
                return lambda e: self._select_var(i)
            var_frame.bind("<Button-1>", make_selector(idx))
            for child in var_frame.winfo_children():
                child.bind("<Button-1>", make_selector(idx))

    def _select_var(self, idx):
        self.selected_var_idx = idx
        messagebox.showinfo("Selecionado", f"Variável '{self.variables[idx]['name']}' selecionada.")

    def _add_variable(self):
        dialog = VarConfigDialog(self, None)
        self.wait_window(dialog)
        if dialog.result:
            self.variables.append(dialog.result)
            self._update_vars_list()

    def _edit_variable(self):
        if self.selected_var_idx is None or self.selected_var_idx >= len(self.variables):
            messagebox.showwarning("Seleção", "Selecione uma variável para editar.")
            return
        
        var = self.variables[self.selected_var_idx]
        dialog = VarConfigDialog(self, var)
        self.wait_window(dialog)
        if dialog.result:
            self.variables[self.selected_var_idx] = dialog.result
            self._update_vars_list()

    def _remove_variable(self):
        if self.selected_var_idx is None or self.selected_var_idx >= len(self.variables):
            messagebox.showwarning("Seleção", "Selecione uma variável para remover.")
            return
        
        del self.variables[self.selected_var_idx]
        self.selected_var_idx = None
        self._update_vars_list()

    def _run_simulation(self):
        if not self.variables:
            messagebox.showerror("Erro", "Defina ao menos uma variável antes de simular.")
            return

        try:
            n_sims = int(self.num_sims_entry.get())
            if n_sims <= 0:
                raise ValueError("Número de simulações deve ser positivo.")
        except Exception as e:
            messagebox.showerror("Erro", f"Número de simulações inválido: {e}")
            return

        # Generate data
        try:
            data = {}
            for var in self.variables:
                dist_info = DISTRIBUTIONS[var['dist']]
                params = var['params']
                
                if var['dist'] == "Normal":
                    data[var['name']] = np.random.normal(params['mean'], params['std'], n_sims)
                elif var['dist'] == "Uniform":
                    data[var['name']] = np.random.uniform(params['min'], params['max'], n_sims)
                elif var['dist'] == "Triangular":
                    # scipy triang needs normalized mode
                    left, mode, right = params['left'], params['mode'], params['right']
                    c = (mode - left) / (right - left)
                    data[var['name']] = dist_info['scipy'].rvs(c, loc=left, scale=right-left, size=n_sims)
                elif var['dist'] == "Lognormal":
                    # lognorm(s, loc, scale) where s is shape (sigma), scale is exp(mu)
                    data[var['name']] = np.random.lognormal(params['mean'], params['std'], n_sims)
                elif var['dist'] == "Exponential":
                    data[var['name']] = np.random.exponential(params['scale'], n_sims)
                elif var['dist'] == "Gamma":
                    data[var['name']] = np.random.gamma(params['shape'], params['scale'], n_sims)
                elif var['dist'] == "Beta":
                    data[var['name']] = np.random.beta(params['alpha'], params['beta'], n_sims)

            self.simulated_data = pd.DataFrame(data)

            # Optional formula
            formula = self.formula_entry.get().strip()
            if formula:
                try:
                    env = {var['name']: self.simulated_data[var['name']].values for var in self.variables}
                    result = pd.eval(formula, engine="python", local_dict=env)
                    self.simulated_data['Result'] = result
                except Exception as e:
                    messagebox.showwarning("Fórmula", f"Erro ao avaliar fórmula: {e}\nContinuando sem coluna de resultado.")

            self._display_results()
            self.export_csv_btn.configure(state="normal")
            self.export_xlsx_btn.configure(state="normal")

        except Exception as e:
            messagebox.showerror("Erro na Simulação", f"Erro ao gerar dados: {e}")

    def _display_results(self):
        # Clear stats
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)

        # Compute stats
        for col in self.simulated_data.columns:
            series = self.simulated_data[col]
            mean = series.mean()
            std = series.std()
            q = series.quantile([0, 0.25, 0.5, 0.75, 1.0])
            self.stats_tree.insert("", "end", values=[
                col,
                f"{mean:.4g}",
                f"{std:.4g}",
                f"{q.iloc[0]:.4g}",
                f"{q.iloc[1]:.4g}",
                f"{q.iloc[2]:.4g}",
                f"{q.iloc[3]:.4g}",
                f"{q.iloc[4]:.4g}"
            ])

        # Plot histograms
        self.figure.clear()
        n_vars = len(self.simulated_data.columns)
        n_cols = min(3, n_vars)
        n_rows = int(np.ceil(n_vars / n_cols))

        for i, col in enumerate(self.simulated_data.columns):
            ax = self.figure.add_subplot(n_rows, n_cols, i + 1)
            ax.hist(self.simulated_data[col], bins=50, alpha=0.7, color="#5DADE2", edgecolor="#1B4F72")
            ax.set_title(col, fontsize=10)
            ax.grid(alpha=0.2)

        self.figure.tight_layout()
        self.canvas.draw_idle()

        messagebox.showinfo("Sucesso", f"{len(self.simulated_data)} simulações geradas com sucesso!")

    def _export_data(self, format_type: str):
        if self.simulated_data is None:
            messagebox.showwarning("Nenhum Dado", "Execute a simulação primeiro.")
            return

        if format_type == "xlsx":
            path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
            if path:
                self.simulated_data.to_excel(path, index=False)
                messagebox.showinfo("Exportado", f"Dados salvos em:\n{path}")
        else:
            path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
            if path:
                self.simulated_data.to_csv(path, index=False)
                messagebox.showinfo("Exportado", f"Dados salvos em:\n{path}")


class VarConfigDialog(ctk.CTkToplevel):
    def __init__(self, parent, var_data=None):
        super().__init__(parent)
        self.title("Configurar Variável")
        self.geometry("420x480")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.result = None
        self.var_data = var_data

        self._build_ui()

    def _build_ui(self):
        main = ctk.CTkFrame(self)
        main.pack(fill="both", expand=True, padx=16, pady=16)

        # Name
        ctk.CTkLabel(main, text="Nome da Variável:", font=ctk.CTkFont(size=12, weight="bold"))\
            .pack(anchor="w", pady=(0, 4))
        self.name_entry = ctk.CTkEntry(main)
        self.name_entry.pack(fill="x", pady=(0, 12))
        if self.var_data:
            self.name_entry.insert(0, self.var_data['name'])

        # Distribution
        ctk.CTkLabel(main, text="Distribuição:", font=ctk.CTkFont(size=12, weight="bold"))\
            .pack(anchor="w", pady=(0, 4))
        self.dist_menu = ctk.CTkOptionMenu(main, values=list(DISTRIBUTIONS.keys()), command=self._on_dist_change)
        self.dist_menu.pack(fill="x", pady=(0, 12))
        if self.var_data:
            self.dist_menu.set(self.var_data['dist'])

        # Parameters frame
        self.params_frame = ctk.CTkFrame(main, fg_color="transparent")
        self.params_frame.pack(fill="both", expand=True, pady=(0, 12))
        
        self.param_entries = {}
        self._on_dist_change(self.dist_menu.get())

        # Buttons
        btn_frame = ctk.CTkFrame(main, fg_color="transparent")
        btn_frame.pack(fill="x")
        ctk.CTkButton(btn_frame, text="Salvar", command=self._save, height=40, font=ctk.CTkFont(size=12, weight="bold"))\
            .pack(side="left", fill="x", expand=True, padx=(0, 6))
        ctk.CTkButton(btn_frame, text="Cancelar", command=self.destroy, height=40, fg_color="#7F8C8D")\
            .pack(side="right", fill="x", expand=True)

    def _on_dist_change(self, dist_name):
        for w in self.params_frame.winfo_children():
            w.destroy()
        self.param_entries.clear()

        params = DISTRIBUTIONS[dist_name]["params"]
        for param in params:
            ctk.CTkLabel(self.params_frame, text=f"{param}:", font=ctk.CTkFont(size=11))\
                .pack(anchor="w", pady=(4, 2))
            entry = ctk.CTkEntry(self.params_frame)
            entry.pack(fill="x", pady=(0, 8))
            
            # Default values
            if self.var_data and self.var_data['dist'] == dist_name and param in self.var_data['params']:
                entry.insert(0, str(self.var_data['params'][param]))
            else:
                # Provide sensible defaults
                defaults = {
                    "mean": "0", "std": "1", "min": "0", "max": "1",
                    "left": "0", "mode": "0.5", "right": "1",
                    "scale": "1", "shape": "2", "alpha": "2", "beta": "2"
                }
                entry.insert(0, defaults.get(param, "1"))
            
            self.param_entries[param] = entry

    def _save(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Erro", "Nome da variável é obrigatório.")
            return

        dist = self.dist_menu.get()
        params = {}
        try:
            for param, entry in self.param_entries.items():
                params[param] = float(entry.get())
        except ValueError:
            messagebox.showerror("Erro", "Todos os parâmetros devem ser numéricos.")
            return

        self.result = {"name": name, "dist": dist, "params": params}
        self.destroy()
