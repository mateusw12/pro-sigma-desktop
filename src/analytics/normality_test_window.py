"""
Normality Test Window
- Single set of Y columns (no X)
- Tests: Shapiro-Wilk, Jarque-Bera, Kolmogorov-Smirnov (vs. normal)
- Plots: Histogram + Normal PDF, QQ Plot, ECDF vs Normal CDF
"""
import customtkinter as ctk
from tkinter import messagebox
import pandas as pd
import numpy as np
from scipy import stats
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class NormalityTestWindow(ctk.CTkToplevel):
    def __init__(self, parent, df: pd.DataFrame):
        super().__init__(parent)
        self.title("Normality Test")
        self.geometry("1000x700")
        self.minsize(820, 560)
        try:
            self.state("zoomed")
        except Exception:
            pass
        self.transient(parent)
        self.grab_set()

        self.df = df.copy()
        self.numeric_cols = [c for c in self.df.columns if pd.api.types.is_numeric_dtype(self.df[c])]
        if not self.numeric_cols:
            messagebox.showerror("Erro", "Nenhuma coluna numérica disponível para teste de normalidade.")
            self.destroy()
            return

        self._build_ui()

    def _build_ui(self):
        main = ctk.CTkFrame(self)
        main.pack(fill="both", expand=True, padx=16, pady=16)

        left_container = ctk.CTkFrame(main, width=280)
        left_container.pack(side="left", fill="y")
        left_container.pack_propagate(False)
        
        left = ctk.CTkScrollableFrame(left_container)
        left.pack(fill="both", expand=True)

        # Right panel with scroll
        right_container = ctk.CTkFrame(main)
        right_container.pack(side="right", fill="both", expand=True, padx=(12, 0))
        
        right = ctk.CTkScrollableFrame(right_container)
        right.pack(fill="both", expand=True)

        ctk.CTkLabel(left, text="Testes de Normalidade", font=ctk.CTkFont(size=18, weight="bold"))\
            .pack(pady=(12, 4))
        ctk.CTkLabel(left, text="Selecione colunas Y numéricas e execute Shapiro, Jarque-Bera e KS.", text_color="gray")\
            .pack(pady=(0, 12), padx=8)

        # Y selection
        y_label = ctk.CTkLabel(left, text="Colunas Y", font=ctk.CTkFont(size=12, weight="bold"))
        y_label.pack(anchor="w", padx=12)
        self.cols_var = {}
        cols_frame = ctk.CTkScrollableFrame(left, width=240, height=260)
        cols_frame.pack(padx=12, pady=(4, 10), fill="both", expand=True)
        for col in self.numeric_cols:
            var = ctk.BooleanVar(value=False)
            self.cols_var[col] = var
            chk = ctk.CTkCheckBox(cols_frame, text=col, variable=var)
            chk.pack(anchor="w", padx=4, pady=2)

        # Spacer to keep the action button visible at the bottom
        ctk.CTkFrame(left, fg_color="transparent").pack(fill="both", expand=True)

        run_btn = ctk.CTkButton(
            left,
            text="Rodar Testes",
            command=self._run_tests,
            height=44,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        run_btn.pack(fill="x", padx=12, pady=(10, 14))
        
        # Bottom spacer for footer margin
        ctk.CTkFrame(left, fg_color="transparent", height=20).pack()

        # Right side: table + plot
        table_frame = ctk.CTkFrame(right)
        table_frame.pack(fill="x", padx=8, pady=(6, 8))
        headers = ["Coluna", "Teste", "Estatística", "p-value", "Conclusão"]
        import tkinter.ttk as ttk
        self.tree = ttk.Treeview(table_frame, columns=headers, show="headings", height=9)
        for h in headers:
            self.tree.heading(h, text=h)
            self.tree.column(h, anchor="w", stretch=True, width=140 if h == "Conclusão" else 110)
        self.tree.pack(fill="x", padx=6, pady=6)

        self.plot_frame = ctk.CTkFrame(right)
        self.plot_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.figure = Figure(figsize=(8, 4.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.figure.tight_layout()

        # Plot selector (only after running)
        selector_frame = ctk.CTkFrame(right, fg_color="transparent")
        selector_frame.pack(fill="x", padx=8, pady=(0, 4))
        ctk.CTkLabel(selector_frame, text="Visualizar gráficos para:", font=ctk.CTkFont(size=12, weight="bold"))\
            .pack(side="left", padx=(4, 8))
        self.plot_selector = ctk.CTkOptionMenu(selector_frame, values=[], command=self._on_select_plot_col)
        self.plot_selector.pack(side="left")
        self.selected_cols = []

    def _run_tests(self):
        selected = [c for c, v in self.cols_var.items() if v.get()]
        if not selected:
            messagebox.showerror("Seleção", "Selecione ao menos uma coluna Y para testar.")
            return

        # Clear previous
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.figure.clear()

        alpha = 0.05
        self.selected_cols = selected
        plotted_col = None
        for col in selected:
            series = self.df[col].dropna().astype(float)
            if len(series) < 3:
                self._add_row(col, "N/A", "N/A", "N/A", "Dados insuficientes")
                continue
            if plotted_col is None:
                plotted_col = col

            # Shapiro-Wilk
            try:
                stat, p = stats.shapiro(series)
                concl = "Não rejeita H0" if p > alpha else "Rejeita H0"
                self._add_row(col, "Shapiro-Wilk", stat, p, concl)
            except Exception as e:
                self._add_row(col, "Shapiro-Wilk", "Erro", "Erro", str(e))

            # Jarque-Bera
            try:
                jb_stat, jb_p = stats.jarque_bera(series)
                concl = "Não rejeita H0" if jb_p > alpha else "Rejeita H0"
                self._add_row(col, "Jarque-Bera", jb_stat, jb_p, concl)
            except Exception as e:
                self._add_row(col, "Jarque-Bera", "Erro", "Erro", str(e))

            # Kolmogorov-Smirnov vs N(mean, std)
            try:
                mu, sigma = series.mean(), series.std(ddof=1)
                if sigma == 0 or np.isnan(sigma):
                    raise ValueError("Desvio padrão zero ou inválido")
                ks_stat, ks_p = stats.kstest(series, 'norm', args=(mu, sigma))
                concl = "Não rejeita H0" if ks_p > alpha else "Rejeita H0"
                self._add_row(col, "Kolmogorov-Smirnov", ks_stat, ks_p, concl)
            except Exception as e:
                self._add_row(col, "Kolmogorov-Smirnov", "Erro", "Erro", str(e))

        # Update selector and render plots for the first available column
        if self.selected_cols:
            self.plot_selector.configure(values=self.selected_cols)
            target_col = plotted_col or self.selected_cols[0]
            self.plot_selector.set(target_col)
            self._render_plots(target_col)
        else:
            self.plot_selector.configure(values=[])

    def _add_row(self, col, test, stat, pval, conclusion):
        def fmt(x):
            if isinstance(x, (float, np.floating)):
                return f"{x:.4g}"
            return str(x)
        self.tree.insert("", "end", values=[col, test, fmt(stat), fmt(pval), conclusion])

    def _on_select_plot_col(self, col_name: str):
        if col_name and col_name in self.selected_cols:
            self._render_plots(col_name)

    def _render_plots(self, col_name: str):
        series = self.df[col_name].dropna().astype(float)
        if len(series) < 3:
            return

        self.figure.clear()
        gs = self.figure.add_gridspec(1, 3, wspace=0.3)

        mu, sigma = series.mean(), series.std(ddof=1)

        # Histogram + fitted normal PDF
        ax_hist = self.figure.add_subplot(gs[0, 0])
        ax_hist.hist(series, bins="auto", density=True, alpha=0.65, color="#5DADE2", edgecolor="#1B4F72")
        if sigma and not np.isnan(sigma):
            xs = np.linspace(series.min(), series.max(), 200)
            pdf = stats.norm.pdf(xs, mu, sigma)
            ax_hist.plot(xs, pdf, color="#E74C3C", linewidth=2, label="Normal PDF")
        ax_hist.set_title(f"Histograma · {col_name}")
        ax_hist.grid(alpha=0.2)
        ax_hist.legend()

        # QQ plot
        ax_qq = self.figure.add_subplot(gs[0, 1])
        stats.probplot(series, dist="norm", plot=ax_qq)
        ax_qq.set_title("QQ Plot (Normal)")
        ax_qq.grid(alpha=0.2)

        # ECDF vs Normal CDF
        ax_cdf = self.figure.add_subplot(gs[0, 2])
        sorted_vals = np.sort(series)
        n = len(sorted_vals)
        ecdf = np.arange(1, n + 1) / n
        ax_cdf.step(sorted_vals, ecdf, where="post", color="#5DADE2", label="ECDF")
        if sigma and not np.isnan(sigma):
            cdf = stats.norm.cdf(sorted_vals, mu, sigma)
            ax_cdf.plot(sorted_vals, cdf, color="#E74C3C", linewidth=2, label="Normal CDF")
        ax_cdf.set_ylim(0, 1)
        ax_cdf.set_title("ECDF vs Normal CDF")
        ax_cdf.grid(alpha=0.2)
        ax_cdf.legend()

        self.figure.tight_layout()
        self.canvas.draw_idle()