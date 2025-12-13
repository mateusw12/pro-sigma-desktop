"""
Distribution Fit Test Window
- Select one or multiple Y columns
- Choose candidate distributions (Normal, Lognormal, Weibull, Exponential, Gamma)
- Compute parameters, AIC, BIC; compare models
- Show histogram overlaid with fitted PDFs
"""
import customtkinter as ctk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats


CANDIDATES = {
    "Normal": stats.norm,
    "Lognormal": stats.lognorm,  # shape, loc, scale (shape=sigma)
    "Weibull": stats.weibull_min,  # shape=k, loc, scale
    "Exponential": stats.expon,  # loc, scale
    "Gamma": stats.gamma,  # a, loc, scale
}


def compute_aic(n, loglik, k):
    return 2 * k - 2 * loglik


def compute_bic(n, loglik, k):
    return k * np.log(n) - 2 * loglik


class DistributionTestWindow(ctk.CTkToplevel):
    def __init__(self, parent, df: pd.DataFrame):
        super().__init__(parent)
        self.title("Distribution Fit Test")
        self.geometry("1000x700")
        self.minsize(800, 550)
        self.transient(parent)
        self.grab_set()

        self.df = df.copy()
        self.numeric_cols = list(self.df.select_dtypes(include="number").columns)

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

        title = ctk.CTkLabel(left, text="Distribution Test", font=ctk.CTkFont(size=18, weight="bold"))
        title.pack(pady=(12, 4))
        desc = ctk.CTkLabel(left, text="Pick Y columns and candidate distributions to compare.", text_color="gray")
        desc.pack(pady=(0, 12), padx=8)

        # Y selection
        y_label = ctk.CTkLabel(left, text="Y columns", font=ctk.CTkFont(size=12, weight="bold"))
        y_label.pack(anchor="w", padx=12)
        self.y_checks = {}
        y_scroll = ctk.CTkScrollableFrame(left, width=240, height=240)
        y_scroll.pack(padx=12, pady=(4, 8), fill="both", expand=True)
        if not self.numeric_cols:
            ctk.CTkLabel(y_scroll, text="No numeric columns found.", text_color="red").pack(pady=6)
        else:
            for col in self.numeric_cols:
                var = ctk.BooleanVar(value=False)
                cb = ctk.CTkCheckBox(y_scroll, text=col, variable=var)
                cb.pack(anchor="w", pady=2, padx=4)
                self.y_checks[col] = var

        # Distributions selection
        d_label = ctk.CTkLabel(left, text="Distributions", font=ctk.CTkFont(size=12, weight="bold"))
        d_label.pack(anchor="w", padx=12, pady=(8, 0))
        self.d_checks = {}
        d_scroll = ctk.CTkScrollableFrame(left, width=240, height=160)
        d_scroll.pack(padx=12, pady=(4, 8), fill="x")
        for name in CANDIDATES.keys():
            var = ctk.BooleanVar(value=name in ("Normal", "Lognormal", "Weibull"))
            cb = ctk.CTkCheckBox(d_scroll, text=name, variable=var)
            cb.pack(anchor="w", pady=2, padx=4)
            self.d_checks[name] = var

        btn_row = ctk.CTkFrame(left, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=(6, 12))
        run_btn = ctk.CTkButton(btn_row, text="Run", command=self._run, height=40, font=ctk.CTkFont(size=14, weight="bold"))
        run_btn.pack(fill="x")
        
        # Bottom spacer for footer margin
        ctk.CTkFrame(left, fg_color="transparent", height=20).pack()

        # Right: table + plots
        table_frame = ctk.CTkFrame(right)
        table_frame.pack(fill="x", padx=8, pady=(8, 6))
        cols = ["Column", "Distribution", "Params", "LogLik", "AIC", "BIC"]
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=8)
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="w", stretch=True, width=150 if col == "Params" else 100)
        self.tree.pack(fill="x", padx=6, pady=6)

        self.plot_frame = ctk.CTkFrame(right)
        self.plot_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.figure, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.figure.tight_layout()

    def _run(self):
        y_cols = [c for c, v in self.y_checks.items() if v.get()]
        dists = [n for n, v in self.d_checks.items() if v.get()]
        if not y_cols:
            messagebox.showwarning("Select Y", "Select at least one numeric Y column.")
            return
        if not dists:
            messagebox.showwarning("Select distributions", "Pick at least one candidate distribution.")
            return

        # Clear table and plot
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

        # For plotting we overlay for the first selected column
        first_col = y_cols[0]
        vals = self.df[first_col].dropna().to_numpy()
        if vals.size == 0:
            messagebox.showerror("Empty data", f"Column {first_col} has no numeric values.")
            return

        # Histogram
        self.ax.hist(vals, bins=20, density=True, alpha=0.35, color="#2E86DE", label=f"Histogram: {first_col}")

        xgrid = np.linspace(np.nanmin(vals), np.nanmax(vals), 300)

        # Fit each selected distribution for each Y, fill table; plot for first Y
        for col in y_cols:
            series = self.df[col].dropna()
            data = series.to_numpy()
            n = data.size
            if n == 0:
                continue
            for dist_name in dists:
                dist = CANDIDATES[dist_name]
                try:
                    params = dist.fit(data)
                    logpdf = dist.logpdf(data, *params)
                    loglik = float(np.sum(logpdf))
                    k = len(params)
                    aic = compute_aic(n, loglik, k)
                    bic = compute_bic(n, loglik, k)

                    # Params pretty
                    ptxt = ", ".join([f"{p:.4g}" if isinstance(p, (int, float, np.floating)) else str(p) for p in params])
                    self.tree.insert("", "end", values=[col, dist_name, ptxt, f"{loglik:.4g}", f"{aic:.4g}", f"{bic:.4g}"])

                    # Plot only for first column to keep chart readable
                    if col == first_col:
                        ypdf = dist.pdf(xgrid, *params)
                        self.ax.plot(xgrid, ypdf, label=f"{dist_name}")
                except Exception as e:
                    self.tree.insert("", "end", values=[col, dist_name, f"fit error: {e}", "-", "-", "-"])

        self.ax.set_title("Histogram with fitted PDFs")
        self.ax.set_xlabel("Value")
        self.ax.set_ylabel("Density")
        self.ax.legend()
        self.figure.tight_layout()
        self.canvas.draw_idle()
