"""
Time Series Analysis Window — decomposição e diagnóstico ADF.
"""
import threading
import customtkinter as ctk
from tkinter import messagebox

from src.utils.lazy_imports import get_matplotlib_figure, get_matplotlib_backend

from .time_series_utils import decompose_series, adf_test

_Figure = None
_FigureCanvasTkAgg = None


def _ensure_libs():
    global _Figure, _FigureCanvasTkAgg
    if _Figure is None:
        _Figure = get_matplotlib_figure()
        _FigureCanvasTkAgg = get_matplotlib_backend()
    return _Figure, _FigureCanvasTkAgg


class TimeSeriesWindow(ctk.CTkToplevel):
    def __init__(self, parent, df):
        super().__init__(parent)
        Figure, FigureCanvasTkAgg = _ensure_libs()
        self.Figure = Figure
        self.FigureCanvasTkAgg = FigureCanvasTkAgg

        self.df = df
        self._canvas = None

        self.title("Análise de Séries Temporais")
        self.resizable(True, True)
        self.minsize(900, 620)
        self.geometry("1100x750")
        self.transient(parent)
        self.grab_set()

        self._build_ui()

    # ──────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        top = ctk.CTkFrame(self)
        top.pack(fill="x", padx=20, pady=(15, 0))
        ctk.CTkLabel(top, text="📅 Análise de Séries Temporais",
                     font=ctk.CTkFont(size=20, weight="bold")).pack(side="left")

        # Config bar
        cfg = ctk.CTkFrame(self)
        cfg.pack(fill="x", padx=20, pady=10)

        all_cols = list(self.df.columns)

        # Date column
        ctk.CTkLabel(cfg, text="Coluna de Data:", font=ctk.CTkFont(size=12)).pack(side="left", padx=(10, 4))
        self._date_col = ctk.StringVar(value=all_cols[0] if all_cols else "")
        ctk.CTkComboBox(cfg, variable=self._date_col, values=all_cols, width=160).pack(side="left", padx=4)

        # Value column
        ctk.CTkLabel(cfg, text="Coluna de Valor:", font=ctk.CTkFont(size=12)).pack(side="left", padx=(14, 4))
        numeric_cols = [c for c in all_cols if self.df[c].dtype.kind in 'fiu']
        value_default = numeric_cols[-1] if numeric_cols else all_cols[-1]
        self._val_col = ctk.StringVar(value=value_default)
        ctk.CTkComboBox(cfg, variable=self._val_col, values=all_cols, width=160).pack(side="left", padx=4)

        # Period
        ctk.CTkLabel(cfg, text="Período:", font=ctk.CTkFont(size=12)).pack(side="left", padx=(14, 4))
        self._period_entry = ctk.CTkEntry(cfg, width=60)
        self._period_entry.insert(0, "12")
        self._period_entry.pack(side="left", padx=4)

        # Model
        ctk.CTkLabel(cfg, text="Modelo:", font=ctk.CTkFont(size=12)).pack(side="left", padx=(14, 4))
        self._model_var = ctk.StringVar(value="additive")
        ctk.CTkComboBox(cfg, variable=self._model_var,
                        values=["additive", "multiplicative"], width=140).pack(side="left", padx=4)

        # Button
        self._btn = ctk.CTkButton(cfg, text="Decompor", command=self._start,
                                  fg_color="#2E86DE", hover_color="#1E5BA8", width=120)
        self._btn.pack(side="left", padx=16)

        self._status_lbl = ctk.CTkLabel(cfg, text="", font=ctk.CTkFont(size=11), text_color="gray")
        self._status_lbl.pack(side="left", padx=4)

        # Main content: chart + ADF panel
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        # Chart area
        self._chart_container = ctk.CTkFrame(content)
        self._chart_container.pack(side="left", fill="both", expand=True, padx=(0, 8))

        # ADF panel
        self._adf_panel = ctk.CTkFrame(content, width=240)
        self._adf_panel.pack(side="right", fill="y")
        self._adf_panel.pack_propagate(False)

        ctk.CTkLabel(self._adf_panel, text="Teste ADF",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(15, 10))

        self._adf_stat_lbl = self._adf_row("Estatística ADF", "—")
        self._adf_pval_lbl = self._adf_row("p-valor", "—")
        self._adf_lags_lbl = self._adf_row("Lags usados", "—")
        self._adf_cv1_lbl = self._adf_row("Valor crítico 1%", "—")
        self._adf_cv5_lbl = self._adf_row("Valor crítico 5%", "—")
        self._adf_cv10_lbl = self._adf_row("Valor crítico 10%", "—")

        self._adf_verdict = ctk.CTkLabel(
            self._adf_panel, text="", font=ctk.CTkFont(size=12, weight="bold"),
            wraplength=210)
        self._adf_verdict.pack(pady=(20, 5), padx=10)

        note = ctk.CTkLabel(self._adf_panel,
                            text="H₀: Série não estacionária\n(p < 0.05 → rejeita H₀)",
                            font=ctk.CTkFont(size=10), text_color="gray")
        note.pack(pady=(5, 10), padx=10)

    def _adf_row(self, label: str, value: str) -> ctk.CTkLabel:
        row = ctk.CTkFrame(self._adf_panel, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=3)
        ctk.CTkLabel(row, text=label + ":", font=ctk.CTkFont(size=10),
                     text_color="gray", anchor="w").pack(anchor="w")
        lbl = ctk.CTkLabel(row, text=value, font=ctk.CTkFont(size=12, weight="bold"), anchor="w")
        lbl.pack(anchor="w")
        return lbl

    # ──────────────────────────────────────────────────────────────────────────
    def _start(self):
        date_col = self._date_col.get()
        val_col = self._val_col.get()
        try:
            period = int(self._period_entry.get())
            if period < 2:
                raise ValueError()
        except ValueError:
            messagebox.showerror("Erro", "Período deve ser um inteiro ≥ 2.", parent=self)
            return

        model = self._model_var.get()
        self._btn.configure(state="disabled", text="Calculando...")
        self._status_lbl.configure(text="Decompondo...", text_color="orange")

        def _work():
            try:
                series = self.df[val_col].dropna().reset_index(drop=True)
                decomp = decompose_series(series, period=period, model=model)
                adf = adf_test(series)
                self.after(0, lambda: self._update(decomp, adf, series, date_col))
            except Exception as e:
                self.after(0, lambda: self._err(str(e)))

        threading.Thread(target=_work, daemon=True).start()

    def _err(self, msg: str):
        self._btn.configure(state="normal", text="Decompor")
        self._status_lbl.configure(text="Erro!", text_color="red")
        messagebox.showerror("Erro na decomposição", msg, parent=self)

    def _update(self, decomp: dict, adf: dict, series, date_col: str):
        self._btn.configure(state="normal", text="Decompor")
        self._status_lbl.configure(text="✓ Concluído", text_color="#4CAF50")

        # Update ADF panel
        self._adf_stat_lbl.configure(text=f"{adf['statistic']:.4f}")
        self._adf_pval_lbl.configure(text=f"{adf['p_value']:.4f}")
        self._adf_lags_lbl.configure(text=str(adf['n_lags']))
        cv = adf['critical_values']
        self._adf_cv1_lbl.configure(text=f"{cv.get('1%', 0):.4f}")
        self._adf_cv5_lbl.configure(text=f"{cv.get('5%', 0):.4f}")
        self._adf_cv10_lbl.configure(text=f"{cv.get('10%', 0):.4f}")

        if adf['is_stationary']:
            self._adf_verdict.configure(text="✅ Estacionária\n(rejeita H₀)", text_color="#4CAF50")
        else:
            self._adf_verdict.configure(text="⚠️ Não Estacionária\n(não rejeita H₀)", text_color="orange")

        # Draw chart
        self._draw_chart(decomp, series, date_col)

    def _draw_chart(self, decomp: dict, series, date_col: str):
        for w in self._chart_container.winfo_children():
            if hasattr(w, 'get_tk_widget') or str(type(w)) != "<class 'customtkinter.windows.widgets.ctk_label.CTkLabel'>":
                w.destroy()

        import numpy as np

        # Build x axis
        if date_col and date_col in self.df.columns:
            x = self.df[date_col].reset_index(drop=True).iloc[:len(decomp['observed'])].astype(str).tolist()
        else:
            x = list(range(len(decomp['observed'])))

        fig = self.Figure(figsize=(9, 7), dpi=80)

        panels = [
            ('Observado', decomp['observed'], '#2196F3'),
            ('Tendência', decomp['trend'], '#FF5722'),
            ('Sazonalidade', decomp['seasonal'], '#4CAF50'),
            ('Resíduo', decomp['residual'], '#9C27B0'),
        ]

        for i, (title, data, color) in enumerate(panels, 1):
            ax = fig.add_subplot(4, 1, i)
            valid = ~np.isnan(data)
            xi = [x[j] for j in range(len(x)) if valid[j]]
            yi = data[valid]
            ax.plot(xi, yi, color=color, linewidth=1.2)
            ax.set_ylabel(title, fontsize=8)
            ax.tick_params(axis='x', labelsize=7, rotation=30)
            ax.tick_params(axis='y', labelsize=7)
            ax.grid(True, alpha=0.25)

        fig.tight_layout(pad=1.2)

        canvas = self.FigureCanvasTkAgg(fig, master=self._chart_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        self._canvas = canvas
