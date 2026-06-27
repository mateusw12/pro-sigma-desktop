"""
Sample Size Explorer Window — calcula N, margem de erro e nível de confiança.
Não requer dados (abre direto).
"""
import customtkinter as ctk
from tkinter import messagebox

from .sample_size_explorer_utils import (
    calc_sample_size,
    calc_margin_of_error,
    calc_confidence_level,
    sensitivity_table,
)


class SampleSizeExplorerWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Explorador de Tamanho de Amostra")
        self.resizable(True, True)
        self.minsize(720, 560)
        self.geometry("860x680")
        self.transient(parent)
        self.grab_set()
        self._build_ui()

    # ──────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        header = ctk.CTkLabel(self, text="🔢 Explorador de Tamanho de Amostra",
                              font=ctk.CTkFont(size=20, weight="bold"))
        header.pack(pady=(15, 5))

        sub = ctk.CTkLabel(self,
                           text="Calcule N, margem de erro ou nível de confiança para estudos Six Sigma",
                           font=ctk.CTkFont(size=11), text_color="gray")
        sub.pack(pady=(0, 10))

        tabs = ctk.CTkTabview(self)
        tabs.pack(fill="both", expand=True, padx=20, pady=(0, 15))

        self._build_calc_n_tab(tabs.add("Calcular N"))
        self._build_calc_margin_tab(tabs.add("Calcular Margem"))
        self._build_calc_confidence_tab(tabs.add("Calcular Confiança"))
        self._build_sensitivity_tab(tabs.add("Tabela de Sensibilidade"))

    # ── Tab 1: Calcular N ──────────────────────────────────────────────────
    def _build_calc_n_tab(self, tab):
        frame = ctk.CTkFrame(tab, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=30, pady=20)

        ctk.CTkLabel(frame, text="Calcular Tamanho de Amostra (N)",
                     font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", pady=(0, 15))

        self._n_conf = self._input_row(frame, "Nível de Confiança (ex: 0.95):", "0.95")
        self._n_margin = self._input_row(frame, "Margem de Erro % (ex: 5.0):", "5.0")

        ctk.CTkButton(frame, text="Calcular N", command=self._do_calc_n,
                      fg_color="#2E86DE", hover_color="#1E5BA8", width=160).pack(pady=15)

        self._n_result_frame = ctk.CTkFrame(frame)
        self._n_result_frame.pack(fill="x", pady=5)
        self._n_lbl_n = self._res_row(self._n_result_frame, "Tamanho de Amostra (N)", "—")
        self._n_lbl_z = self._res_row(self._n_result_frame, "Z-Score", "—")
        self._n_lbl_sig = self._res_row(self._n_result_frame, "Erro Padrão de σ (%)", "—")

    def _do_calc_n(self):
        try:
            cl = float(self._n_conf.get())
            m = float(self._n_margin.get())
            r = calc_sample_size(cl, m)
            self._n_lbl_n.configure(text=str(r['sample_size']))
            self._n_lbl_z.configure(text=f"{r['z_score']:.4f}")
            self._n_lbl_sig.configure(text=f"{r['sigma_error_pct']:.4f} %")
        except Exception as e:
            messagebox.showerror("Erro", str(e), parent=self)

    # ── Tab 2: Calcular Margem ─────────────────────────────────────────────
    def _build_calc_margin_tab(self, tab):
        frame = ctk.CTkFrame(tab, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=30, pady=20)

        ctk.CTkLabel(frame, text="Calcular Margem de Erro",
                     font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", pady=(0, 15))

        self._m_n = self._input_row(frame, "Tamanho de Amostra (N):", "100")
        self._m_conf = self._input_row(frame, "Nível de Confiança (ex: 0.95):", "0.95")

        ctk.CTkButton(frame, text="Calcular Margem", command=self._do_calc_margin,
                      fg_color="#2E86DE", hover_color="#1E5BA8", width=160).pack(pady=15)

        self._m_result_frame = ctk.CTkFrame(frame)
        self._m_result_frame.pack(fill="x", pady=5)
        self._m_lbl_m = self._res_row(self._m_result_frame, "Margem de Erro (%)", "—")
        self._m_lbl_z = self._res_row(self._m_result_frame, "Z-Score", "—")
        self._m_lbl_sig = self._res_row(self._m_result_frame, "Erro Padrão de σ (%)", "—")

    def _do_calc_margin(self):
        try:
            n = int(self._m_n.get())
            cl = float(self._m_conf.get())
            r = calc_margin_of_error(n, cl)
            self._m_lbl_m.configure(text=f"{r['margin_pct']:.4f} %")
            self._m_lbl_z.configure(text=f"{r['z_score']:.4f}")
            self._m_lbl_sig.configure(text=f"{r['sigma_error_pct']:.4f} %")
        except Exception as e:
            messagebox.showerror("Erro", str(e), parent=self)

    # ── Tab 3: Calcular Confiança ──────────────────────────────────────────
    def _build_calc_confidence_tab(self, tab):
        frame = ctk.CTkFrame(tab, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=30, pady=20)

        ctk.CTkLabel(frame, text="Calcular Nível de Confiança",
                     font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", pady=(0, 15))

        self._c_n = self._input_row(frame, "Tamanho de Amostra (N):", "100")
        self._c_margin = self._input_row(frame, "Margem de Erro % (ex: 5.0):", "5.0")

        ctk.CTkButton(frame, text="Calcular Confiança", command=self._do_calc_confidence,
                      fg_color="#2E86DE", hover_color="#1E5BA8", width=180).pack(pady=15)

        self._c_result_frame = ctk.CTkFrame(frame)
        self._c_result_frame.pack(fill="x", pady=5)
        self._c_lbl_cl = self._res_row(self._c_result_frame, "Nível de Confiança", "—")
        self._c_lbl_cl_pct = self._res_row(self._c_result_frame, "Nível de Confiança (%)", "—")
        self._c_lbl_z = self._res_row(self._c_result_frame, "Z-Score", "—")
        self._c_lbl_sig = self._res_row(self._c_result_frame, "Erro Padrão de σ (%)", "—")

    def _do_calc_confidence(self):
        try:
            n = int(self._c_n.get())
            m = float(self._c_margin.get())
            r = calc_confidence_level(n, m)
            cl = r['confidence_level']
            self._c_lbl_cl.configure(text=f"{cl:.6f}")
            self._c_lbl_cl_pct.configure(text=f"{cl*100:.2f} %")
            self._c_lbl_z.configure(text=f"{r['z_score']:.4f}")
            self._c_lbl_sig.configure(text=f"{r['sigma_error_pct']:.4f} %")
        except Exception as e:
            messagebox.showerror("Erro", str(e), parent=self)

    # ── Tab 4: Tabela de Sensibilidade ──────────────────────────────────────
    def _build_sensitivity_tab(self, tab):
        frame = ctk.CTkFrame(tab, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=20, pady=15)

        ctk.CTkLabel(frame,
                     text="Tabela de Sensibilidade — N × Margem de Erro",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", pady=(0, 10))

        ctk.CTkLabel(frame,
                     text="Confiança: 90%, 95%, 99% | Margem de erro: 1%, 2%, 5%, 10%, 15%",
                     font=ctk.CTkFont(size=10), text_color="gray").pack(anchor="w", pady=(0, 8))

        ctk.CTkButton(frame, text="Gerar Tabela", command=self._do_sensitivity,
                      fg_color="#2E86DE", hover_color="#1E5BA8", width=160).pack(anchor="w", pady=(0, 10))

        self._sens_scroll = ctk.CTkScrollableFrame(frame, fg_color="transparent")
        self._sens_scroll.pack(fill="both", expand=True)

    def _do_sensitivity(self):
        for w in self._sens_scroll.winfo_children():
            w.destroy()

        cls = [0.90, 0.95, 0.99]
        margins = [1.0, 2.0, 5.0, 10.0, 15.0]
        rows = sensitivity_table(cls, margins)

        # Header
        hdrs = ["Confiança", "Margem (%)"] + [f"N @ {m}%" for m in margins]
        cols = len(margins) + 2
        widths = [100, 90] + [80] * len(margins)

        hdr_frame = ctk.CTkFrame(self._sens_scroll)
        hdr_frame.pack(fill="x", pady=(0, 2))
        header_labels = ["Confiança", "Margem (%)"] + [f"{m}%" for m in margins]
        full_hdrs = ["Confiança (%)"] + [f"N @ Margem {m}%" for m in margins]
        col_widths = [110] + [90] * len(margins)

        # Build a pivot table: rows = confidence, cols = margin
        pivot = {cl: {} for cl in cls}
        for r in rows:
            pivot[r['confidence']][r['margin_pct']] = r['sample_size']

        # Header row
        h_row = ctk.CTkFrame(self._sens_scroll, fg_color="gray25")
        h_row.pack(fill="x")
        ctk.CTkLabel(h_row, text="Confiança", font=ctk.CTkFont(size=11, weight="bold"),
                     width=110, anchor="center").pack(side="left", padx=2, pady=4)
        for m in margins:
            ctk.CTkLabel(h_row, text=f"Margem {m}%", font=ctk.CTkFont(size=11, weight="bold"),
                         width=90, anchor="center").pack(side="left", padx=2, pady=4)

        # Data rows
        for i, cl in enumerate(cls):
            bg = "gray20" if i % 2 == 0 else "gray17"
            d_row = ctk.CTkFrame(self._sens_scroll, fg_color=bg)
            d_row.pack(fill="x")
            ctk.CTkLabel(d_row, text=f"{cl*100:.0f}%", font=ctk.CTkFont(size=11),
                         width=110, anchor="center").pack(side="left", padx=2, pady=3)
            for m in margins:
                n = pivot[cl].get(m)
                txt = f"{n:,}" if n is not None else "—"
                ctk.CTkLabel(d_row, text=txt, font=ctk.CTkFont(size=11),
                             width=90, anchor="center").pack(side="left", padx=2, pady=3)

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _input_row(self, parent, label: str, default: str) -> ctk.CTkEntry:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=4)
        ctk.CTkLabel(row, text=label, font=ctk.CTkFont(size=12), width=250, anchor="w").pack(side="left")
        entry = ctk.CTkEntry(row, width=120)
        entry.insert(0, default)
        entry.pack(side="left", padx=10)
        return entry

    def _res_row(self, parent, label: str, value: str) -> ctk.CTkLabel:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=15, pady=3)
        ctk.CTkLabel(row, text=label + ":", font=ctk.CTkFont(size=11),
                     text_color="gray", width=220, anchor="w").pack(side="left")
        lbl = ctk.CTkLabel(row, text=value, font=ctk.CTkFont(size=13, weight="bold"), anchor="w")
        lbl.pack(side="left", padx=10)
        return lbl
