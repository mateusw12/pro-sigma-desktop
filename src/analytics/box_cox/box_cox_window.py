"""
Box-Cox Transformation Window
"""
import threading
import customtkinter as ctk
from tkinter import messagebox

from src.utils.lazy_imports import get_numpy, get_matplotlib_figure, get_matplotlib_backend

from .box_cox_utils import BoxCoxTransformer

_np = None
_Figure = None
_FigureCanvasTkAgg = None


def _ensure_libs():
    global _np, _Figure, _FigureCanvasTkAgg
    if _np is None:
        _np = get_numpy()
        _Figure = get_matplotlib_figure()
        _FigureCanvasTkAgg = get_matplotlib_backend()
    return _np, _Figure, _FigureCanvasTkAgg


class BoxCoxWindow(ctk.CTkToplevel):
    def __init__(self, parent, df):
        super().__init__(parent)
        np, Figure, FigureCanvasTkAgg = _ensure_libs()
        self.np = np
        self.Figure = Figure
        self.FigureCanvasTkAgg = FigureCanvasTkAgg

        self.df = df
        self.transformer = BoxCoxTransformer()
        self._result = None
        self._canvas = None
        self._ax = None

        self.title("Transformação Box-Cox")
        self.resizable(True, True)
        self.minsize(800, 580)
        self.geometry("1000x650")
        self.transient(parent)
        self.grab_set()

        self._build_ui()

    def _build_ui(self):
        # ── Top bar ──────────────────────────────────────────────────────────
        top = ctk.CTkFrame(self)
        top.pack(fill="x", padx=20, pady=(15, 0))

        ctk.CTkLabel(top, text="🔄 Transformação Box-Cox",
                     font=ctk.CTkFont(size=20, weight="bold")).pack(side="left")

        # ── Config row ───────────────────────────────────────────────────────
        cfg = ctk.CTkFrame(self)
        cfg.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(cfg, text="Coluna Y:", font=ctk.CTkFont(size=12)).pack(side="left", padx=(10, 5))

        numeric_cols = [c for c in self.df.columns if self.df[c].dtype.kind in 'fiu']
        if not numeric_cols:
            numeric_cols = list(self.df.columns)

        self._col_var = ctk.StringVar(value=numeric_cols[0] if numeric_cols else "")
        col_cb = ctk.CTkComboBox(cfg, variable=self._col_var, values=numeric_cols, width=180)
        col_cb.pack(side="left", padx=5)

        ctk.CTkLabel(cfg, text="λ mín:", font=ctk.CTkFont(size=12)).pack(side="left", padx=(20, 5))
        self._lam_min = ctk.CTkEntry(cfg, width=60)
        self._lam_min.insert(0, "-2")
        self._lam_min.pack(side="left", padx=5)

        ctk.CTkLabel(cfg, text="λ máx:", font=ctk.CTkFont(size=12)).pack(side="left", padx=(10, 5))
        self._lam_max = ctk.CTkEntry(cfg, width=60)
        self._lam_max.insert(0, "2")
        self._lam_max.pack(side="left", padx=5)

        self._btn = ctk.CTkButton(cfg, text="Calcular Box-Cox", command=self._start_calc,
                                  fg_color="#2E86DE", hover_color="#1E5BA8", width=160)
        self._btn.pack(side="left", padx=20)

        self._status = ctk.CTkLabel(cfg, text="", font=ctk.CTkFont(size=11), text_color="gray")
        self._status.pack(side="left", padx=5)

        # ── Content area (chart + results side by side) ───────────────────────
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=(0, 15))

        # Chart frame
        self._chart_frame = ctk.CTkFrame(content)
        self._chart_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        ctk.CTkLabel(self._chart_frame, text="SSE vs Lambda",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(pady=(10, 5))

        self._plot_container = ctk.CTkFrame(self._chart_frame, fg_color="transparent")
        self._plot_container.pack(fill="both", expand=True, padx=5, pady=(0, 10))

        # Results frame
        self._results_frame = ctk.CTkFrame(content, width=240)
        self._results_frame.pack(side="right", fill="y")
        self._results_frame.pack_propagate(False)

        ctk.CTkLabel(self._results_frame, text="Resultados",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(15, 10))

        self._lbl_lambda = self._result_row("Lambda Ótimo (λ)", "—")
        self._lbl_geomean = self._result_row("Média Geométrica", "—")
        self._lbl_shift = self._result_row("Shift aplicado", "—")
        self._lbl_sse = self._result_row("SSE mínimo", "—")

        self._interp_label = ctk.CTkLabel(
            self._results_frame, text="", font=ctk.CTkFont(size=11),
            text_color="#4CAF50", wraplength=200)
        self._interp_label.pack(pady=(20, 5), padx=10)

    def _result_row(self, label: str, value: str) -> ctk.CTkLabel:
        row = ctk.CTkFrame(self._results_frame, fg_color="transparent")
        row.pack(fill="x", padx=15, pady=4)
        ctk.CTkLabel(row, text=label + ":", font=ctk.CTkFont(size=11),
                     text_color="gray", anchor="w").pack(anchor="w")
        lbl = ctk.CTkLabel(row, text=value, font=ctk.CTkFont(size=13, weight="bold"), anchor="w")
        lbl.pack(anchor="w")
        return lbl

    def _start_calc(self):
        col = self._col_var.get()
        if not col or col not in self.df.columns:
            messagebox.showerror("Erro", "Selecione uma coluna numérica válida.", parent=self)
            return
        try:
            lam_min = float(self._lam_min.get())
            lam_max = float(self._lam_max.get())
        except ValueError:
            messagebox.showerror("Erro", "Valores de lambda inválidos.", parent=self)
            return

        self._btn.configure(state="disabled", text="Calculando...")
        self._status.configure(text="Calculando...", text_color="orange")

        def _calc():
            try:
                y = self.df[col].dropna().to_numpy(dtype=float)
                result = self.transformer.calculate(y, lambda_start=lam_min, lambda_end=lam_max)
                self.after(0, lambda: self._update_ui(result))
            except Exception as e:
                self.after(0, lambda: self._on_error(str(e)))

        threading.Thread(target=_calc, daemon=True).start()

    def _on_error(self, msg: str):
        self._btn.configure(state="normal", text="Calcular Box-Cox")
        self._status.configure(text="Erro!", text_color="red")
        messagebox.showerror("Erro no cálculo", msg, parent=self)

    def _update_ui(self, result: dict):
        self._result = result
        self._btn.configure(state="normal", text="Calcular Box-Cox")
        self._status.configure(text="✓ Concluído", text_color="#4CAF50")

        bl = result['best_lambda']
        gm = result['geom_mean']
        sh = result['shift']
        best_sse = result['sse_dict'].get(bl)

        self._lbl_lambda.configure(text=f"{bl:.4f}")
        self._lbl_geomean.configure(text=f"{gm:.4f}")
        self._lbl_shift.configure(text=f"{sh:.4f}")
        self._lbl_sse.configure(text=f"{best_sse:.4f}" if best_sse is not None else "—")

        interp = self._interpret_lambda(bl)
        self._interp_label.configure(text=interp)

        self._draw_chart(result)

    def _interpret_lambda(self, lam: float) -> str:
        if lam is None:
            return ""
        if abs(lam) < 0.05:
            return "λ ≈ 0 → Transformação logarítmica recomendada"
        elif abs(lam - 0.5) < 0.1:
            return "λ ≈ 0.5 → Transformação raiz quadrada"
        elif abs(lam - 1.0) < 0.1:
            return "λ ≈ 1.0 → Dados já aproximadamente normais"
        elif abs(lam + 1.0) < 0.1:
            return "λ ≈ -1 → Transformação recíproca (1/y)"
        elif lam > 1.1:
            return "λ > 1 → Distribuição com cauda longa à esquerda"
        else:
            return f"λ = {lam:.2f} → Transformação Box-Cox aplicada"

    def _draw_chart(self, result: dict):
        # Destroy old canvas
        for w in self._plot_container.winfo_children():
            w.destroy()

        lambdas = result['lambdas']
        sses = result['sses']
        best_lam = result['best_lambda']

        fig = self.Figure(figsize=(6, 4), dpi=80)
        ax = fig.add_subplot(111)

        valid = [(l, s) for l, s in zip(lambdas, sses) if s is not None]
        if valid:
            ls, ss = zip(*valid)
            ax.plot(ls, ss, 'b-o', markersize=4, linewidth=1.5, label='SSE')
            if best_lam is not None:
                best_sse = result['sse_dict'].get(best_lam)
                ax.axvline(x=best_lam, color='red', linestyle='--', linewidth=1.5,
                           label=f'λ ótimo = {best_lam:.2f}')
                if best_sse is not None:
                    ax.plot(best_lam, best_sse, 'r*', markersize=12)

        ax.set_xlabel("Lambda (λ)")
        ax.set_ylabel("SSE")
        ax.set_title("SSE vs Lambda — Box-Cox")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout(pad=1.0)

        canvas = self.FigureCanvasTkAgg(fig, master=self._plot_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._canvas = canvas
