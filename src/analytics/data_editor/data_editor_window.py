"""
Editor de Dados — planilha interativa com geração por fórmulas.
"""
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import customtkinter as ctk

from .data_editor_utils import evaluate_formula, format_value, FORMULA_HELP

_pd = None


def _get_pd():
    global _pd
    if _pd is None:
        import pandas as pd
        _pd = pd
    return _pd


# ──────────────────────────────────────────────────────────────────────────────
# Diálogo de Geração de Dados
# ──────────────────────────────────────────────────────────────────────────────

# Definição de cada distribuição: (label, func, [(param_label, default), ...], descrição)
_DIST_DEFS = [
    ("Normal",          "NORMAL",    [("Média", "100"), ("Desvio Padrão", "15")],
     "Distribuição normal (Gaussiana). Ex: alturas, pesos, erros de medição."),
    ("Uniforme",        "UNIFORME",  [("Mínimo", "0"), ("Máximo", "100")],
     "Todos os valores com mesma probabilidade no intervalo [min, max]."),
    ("Log-Normal",      "LOGNORMAL", [("Média (log)", "0"), ("Sigma (log)", "0.5")],
     "Dados positivos com cauda longa. Ex: tempos de reparo, salários."),
    ("Exponencial",     "EXPONENCIAL",[("Escala (1/taxa)", "10")],
     "Tempo entre eventos. Ex: tempo entre falhas."),
    ("Triangular",      "TRIANGULAR",[("Mínimo", "5"), ("Moda", "10"), ("Máximo", "20")],
     "Distribuição com valor mais provável definido. Útil em simulações."),
    ("Weibull",         "WEIBULL",   [("Forma (a)", "1.5")],
     "Análise de confiabilidade e vida útil de componentes."),
    ("Poisson",         "POISSON",   [("Lambda (taxa)", "5")],
     "Contagem de eventos em intervalo fixo. Ex: defeitos por peça."),
    ("Binomial",        "BINOMIAL",  [("Nº Tentativas", "10"), ("Probabilidade p", "0.3")],
     "Nº de sucessos em N tentativas independentes."),
    ("Inteiro Aleatório","RANDINT",  [("Mínimo", "1"), ("Máximo", "10")],
     "Inteiros aleatórios no intervalo [min, max] (inclusive)."),
    ("Sequência",       "SEQUENCIA", [("Início", "1"), ("Fim", "100"), ("Passo", "1")],
     "Gera uma sequência aritmética de valores."),
    ("Espaço Linear",   "LINSPACE",  [("Início", "0"), ("Fim", "1")],
     "N pontos igualmente espaçados entre início e fim."),
    ("Repetir Valor",   "REPETIR",   [("Valor", "0")],
     "Repete o mesmo valor em todas as linhas."),
    ("Fórmula Livre",   "custom",    [],
     "Digite qualquer expressão. Use nomes de colunas como variáveis."),
]
_DIST_NAMES = [d[0] for d in _DIST_DEFS]


class _GenerateDialog(ctk.CTkToplevel):
    """Diálogo para gerar dados para uma coluna — campos de parâmetros por distribuição."""

    def __init__(self, parent, col_names: list, n_rows: int, columns_data: dict, on_apply):
        super().__init__(parent)
        self.title("Gerar Dados")
        self.resizable(True, True)
        self.geometry("520x560")
        self.minsize(480, 480)
        self.transient(parent)
        self.grab_set()

        self._col_names = col_names
        self._n_rows = max(n_rows, 1) if n_rows > 0 else 50
        self._columns_data = columns_data
        self._on_apply = on_apply
        self._param_entries: list[ctk.CTkEntry] = []

        self._build()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build(self):
        # Cabeçalho
        hdr = ctk.CTkFrame(self, fg_color="#1a1a1a", corner_radius=0)
        hdr.pack(fill="x")
        ctk.CTkLabel(hdr, text="🔢 Gerar Dados",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(pady=14)

        body = ctk.CTkScrollableFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=16, pady=8)
        self._body = body

        PAD = dict(padx=4, pady=5)

        # ── Nome da coluna ────────────────────────────────────────────────────
        self._section(body, "Coluna de destino")

        dest_row = ctk.CTkFrame(body, fg_color="transparent")
        dest_row.pack(fill="x", **PAD)

        col_options = ["(nova coluna)"] + self._col_names
        self._dest_var = ctk.StringVar(value=col_options[0])
        dest_cb = ctk.CTkComboBox(dest_row, variable=self._dest_var,
                                  values=col_options, width=180,
                                  command=self._on_dest_change)
        dest_cb.pack(side="left", padx=(0, 8))

        self._col_name_entry = ctk.CTkEntry(dest_row, placeholder_text="Nome da nova coluna",
                                            width=200)
        self._col_name_entry.pack(side="left")

        # ── Quantidade de linhas ──────────────────────────────────────────────
        n_row = ctk.CTkFrame(body, fg_color="transparent")
        n_row.pack(fill="x", **PAD)
        ctk.CTkLabel(n_row, text="Quantidade de valores:", width=180, anchor="w").pack(side="left")
        self._n_entry = ctk.CTkEntry(n_row, width=100)
        self._n_entry.insert(0, str(self._n_rows))
        self._n_entry.pack(side="left", padx=8)

        # ── Tipo de distribuição ──────────────────────────────────────────────
        self._section(body, "Tipo de dado")

        self._dist_var = ctk.StringVar(value=_DIST_NAMES[0])
        dist_cb = ctk.CTkComboBox(body, variable=self._dist_var,
                                  values=_DIST_NAMES, width=280,
                                  command=self._on_dist_change,
                                  font=ctk.CTkFont(size=12))
        dist_cb.pack(anchor="w", **PAD)

        self._desc_lbl = ctk.CTkLabel(body, text="", font=ctk.CTkFont(size=10),
                                      text_color="gray", wraplength=460, justify="left")
        self._desc_lbl.pack(anchor="w", padx=4)

        # ── Painel de parâmetros (recriado ao mudar distribuição) ─────────────
        self._section(body, "Parâmetros")
        self._params_frame = ctk.CTkFrame(body, fg_color="transparent")
        self._params_frame.pack(fill="x", **PAD)

        # ── Fórmula livre (visível só quando "Fórmula Livre") ─────────────────
        self._free_frame = ctk.CTkFrame(body, fg_color="transparent")
        self._free_formula_entry = ctk.CTkEntry(
            self._free_frame, placeholder_text="Ex: ColA * 2 + NORMAL(0, 1, 50)",
            width=460, height=36, font=ctk.CTkFont(size=11))
        self._free_formula_entry.pack(fill="x", pady=2)
        ctk.CTkLabel(self._free_frame,
                     text="Use nomes de colunas diretamente como variáveis.\n"
                          "Funções: NORMAL, UNIFORME, SEQUENCIA, SQRT, LOG, ABS…",
                     font=ctk.CTkFont(size=10), text_color="gray",
                     justify="left").pack(anchor="w")

        # ── Preview ───────────────────────────────────────────────────────────
        self._section(body, "Pré-visualização")
        self._preview_lbl = ctk.CTkLabel(body, text="Clique em 'Visualizar' para ver os primeiros valores.",
                                         font=ctk.CTkFont(size=11), text_color="gray",
                                         anchor="w", justify="left", wraplength=460)
        self._preview_lbl.pack(anchor="w", padx=4, pady=2)

        # Popula parâmetros da distribuição inicial
        self._on_dist_change()

        # ── Rodapé com botões (fora do scroll, sempre visível) ────────────────
        footer = ctk.CTkFrame(self, fg_color="#1a1a1a", corner_radius=0)
        footer.pack(fill="x", side="bottom")

        ctk.CTkButton(footer, text="👁 Visualizar", command=self._preview,
                      fg_color="gray30", hover_color="gray20",
                      width=130, height=38).pack(side="left", padx=(12, 6), pady=10)

        ctk.CTkButton(footer, text="✅ Gerar Dados", command=self._apply,
                      fg_color="#27AE60", hover_color="#1E8449",
                      font=ctk.CTkFont(size=13, weight="bold"),
                      width=160, height=38).pack(side="left", padx=6, pady=10)

        ctk.CTkButton(footer, text="Cancelar", command=self.destroy,
                      fg_color="gray25", hover_color="gray20",
                      width=100, height=38).pack(side="right", padx=12, pady=10)

    def _section(self, parent, title: str):
        ctk.CTkLabel(parent, text=title,
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#2E86DE").pack(anchor="w", padx=4, pady=(12, 2))

    # ── Eventos ───────────────────────────────────────────────────────────────

    def _on_dest_change(self, *_):
        is_new = self._dest_var.get() == "(nova coluna)"
        if is_new:
            self._col_name_entry.pack(side="left")
        else:
            self._col_name_entry.pack_forget()

    def _on_dist_change(self, *_):
        name = self._dist_var.get()
        defn = next((d for d in _DIST_DEFS if d[0] == name), _DIST_DEFS[0])
        _, func, params, desc = defn

        self._desc_lbl.configure(text=desc)

        # Limpa painel de parâmetros anterior
        for w in self._params_frame.winfo_children():
            w.destroy()
        self._param_entries.clear()
        self._free_frame.pack_forget()

        if func == "custom":
            self._free_frame.pack(fill="x", in_=self._params_frame)
        else:
            for label, default in params:
                row = ctk.CTkFrame(self._params_frame, fg_color="transparent")
                row.pack(fill="x", pady=3)
                ctk.CTkLabel(row, text=f"{label}:", width=160, anchor="w",
                             font=ctk.CTkFont(size=12)).pack(side="left")
                entry = ctk.CTkEntry(row, width=140, font=ctk.CTkFont(size=12))
                entry.insert(0, default)
                entry.pack(side="left", padx=8)
                self._param_entries.append(entry)

    # ── Geração ───────────────────────────────────────────────────────────────

    def _get_n(self) -> int:
        try:
            return max(1, int(self._n_entry.get()))
        except ValueError:
            return self._n_rows

    def _build_formula(self) -> str:
        name = self._dist_var.get()
        defn = next((d for d in _DIST_DEFS if d[0] == name), None)
        if defn is None:
            raise ValueError("Distribuição não encontrada.")
        _, func, params, _ = defn
        n = self._get_n()

        if func == "custom":
            expr = self._free_formula_entry.get().strip().lstrip('=')
            if not expr:
                raise ValueError("Digite uma fórmula no campo de expressão.")
            return expr

        values = []
        for i, entry in enumerate(self._param_entries):
            val = entry.get().strip()
            if not val:
                raise ValueError(f"Preencha o parâmetro '{params[i][0]}'.")
            values.append(val)

        # Sequência não usa n como último argumento
        if func == "SEQUENCIA":
            return f"{func}({', '.join(values)})"

        return f"{func}({', '.join(values)}, {n})"

    def _eval(self) -> list:
        formula = self._build_formula()
        n = self._get_n()
        return evaluate_formula(formula, self._columns_data, n)

    def _preview(self):
        try:
            vals = self._eval()
            shown = [format_value(v) for v in vals[:8]]
            suffix = f"  … ({len(vals)} valores no total)" if len(vals) > 8 else f"  ({len(vals)} valores)"
            self._preview_lbl.configure(
                text="  ".join(shown) + suffix,
                text_color="#4CAF50")
        except Exception as e:
            self._preview_lbl.configure(text=f"Erro: {e}", text_color="#FF5555")

    def _apply(self):
        try:
            vals = self._eval()
        except Exception as e:
            messagebox.showerror("Erro", str(e), parent=self)
            return

        dest = self._dest_var.get()
        if dest == "(nova coluna)":
            col_name = self._col_name_entry.get().strip()
            if not col_name:
                col_name = f"Col{len(self._col_names) + 1}"
        else:
            col_name = dest

        self._on_apply(col_name, vals)
        self.destroy()


# ──────────────────────────────────────────────────────────────────────────────
# Janela principal do Editor de Dados
# ──────────────────────────────────────────────────────────────────────────────

class DataEditorWindow(ctk.CTkToplevel):
    """
    Editor de dados estilo planilha.
    Permite digitar, editar, e gerar dados por fórmulas.
    """

    COL_WIDTH = 110
    ROW_HEIGHT = 24

    def __init__(self, parent, initial_df=None, on_use_data=None):
        """
        Args:
            parent: janela pai
            initial_df: DataFrame opcional para pré-carregar
            on_use_data: callback(df) quando o usuário clica "Usar nos Análises"
        """
        super().__init__(parent)
        self.title("Editor de Dados")
        self.resizable(True, True)
        self.minsize(900, 580)
        self.geometry("1150x700")
        self.transient(parent)

        self._on_use_data = on_use_data
        self._columns: list[str] = []        # nomes das colunas
        self._data: list[list] = []          # lista de listas [row][col]
        self._cell_entry: tk.Entry | None = None
        self._selected_col_idx: int | None = None
        self._col_formulas: dict[str, str] = {}  # col_name → formula

        self._apply_treeview_style()
        self._build_ui()

        if initial_df is not None:
            self._load_dataframe(initial_df)
        else:
            self._init_empty(cols=3, rows=20)

    # ── Estilo dark do Treeview ───────────────────────────────────────────────

    def _apply_treeview_style(self):
        style = ttk.Style()
        try:
            style.theme_use('default')
        except Exception:
            pass
        style.configure('Editor.Treeview',
            background='#2b2b2b', foreground='#e0e0e0',
            rowheight=self.ROW_HEIGHT, fieldbackground='#2b2b2b',
            font=('Segoe UI', 10))
        style.configure('Editor.Treeview.Heading',
            background='#1a1a1a', foreground='#ffffff',
            font=('Segoe UI', 10, 'bold'), relief='flat')
        style.map('Editor.Treeview',
            background=[('selected', '#1E5BA8')],
            foreground=[('selected', 'white')])
        style.map('Editor.Treeview.Heading',
            background=[('active', '#2E86DE')])

    # ── Construção da UI ──────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Toolbar ──────────────────────────────────────────────────────────
        toolbar = ctk.CTkFrame(self, fg_color="#1a1a1a", height=48)
        toolbar.pack(fill="x", side="top")
        toolbar.pack_propagate(False)

        def tb_btn(text, cmd, color="#2E86DE", hcolor="#1E5BA8", width=130):
            return ctk.CTkButton(toolbar, text=text, command=cmd,
                                 fg_color=color, hover_color=hcolor,
                                 width=width, height=32,
                                 font=ctk.CTkFont(size=11))

        tb_btn("➕ Coluna", self._add_column_dialog).pack(side="left", padx=(10, 4), pady=8)
        tb_btn("✏️ Renomear Col.", self._rename_selected_column,
               "gray30", "gray20", 130).pack(side="left", padx=4, pady=8)
        tb_btn("🗑 Del. Coluna", self._delete_selected_column,
               "#c0392b", "#922b21", 120).pack(side="left", padx=4, pady=8)

        ctk.CTkFrame(toolbar, width=2, fg_color="gray35").pack(side="left", fill="y", pady=6, padx=6)

        tb_btn("➕ Linhas", self._add_rows_dialog, "gray30", "gray20", 100).pack(side="left", padx=4, pady=8)
        tb_btn("🗑 Del. Linhas", self._delete_selected_rows, "#c0392b", "#922b21", 110).pack(side="left", padx=4, pady=8)

        ctk.CTkFrame(toolbar, width=2, fg_color="gray35").pack(side="left", fill="y", pady=6, padx=6)

        tb_btn("🔢 Gerar Dados", self._open_generate_dialog, "#7B2D8B", "#5C2167", 130).pack(side="left", padx=4, pady=8)
        tb_btn("🧹 Limpar Tudo", self._clear_all, "gray30", "gray20", 120).pack(side="left", padx=4, pady=8)

        ctk.CTkFrame(toolbar, width=2, fg_color="gray35").pack(side="left", fill="y", pady=6, padx=6)

        tb_btn("📂 Abrir CSV/Excel", self._import_file, "gray30", "gray20", 140).pack(side="left", padx=4, pady=8)
        tb_btn("💾 Exportar CSV", self._export_csv, "gray30", "gray20", 130).pack(side="left", padx=4, pady=8)

        # Usar nos Análises (lado direito)
        tb_btn("✅ Usar nos Análises", self._use_in_analysis, "#27AE60", "#1E8449", 160).pack(side="right", padx=10, pady=8)

        # ── Barra de fórmula ──────────────────────────────────────────────────
        fbar = ctk.CTkFrame(self, fg_color="#111111", height=32)
        fbar.pack(fill="x", side="top")
        fbar.pack_propagate(False)
        ctk.CTkLabel(fbar, text="fx", font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="#2E86DE", width=28).pack(side="left", padx=(8, 0))
        self._formula_var = tk.StringVar()
        self._formula_bar = ctk.CTkEntry(fbar, textvariable=self._formula_var,
                                         font=ctk.CTkFont(size=11), height=26,
                                         fg_color="#1a1a1a", border_color="#333")
        self._formula_bar.pack(side="left", fill="x", expand=True, padx=6, pady=3)
        self._formula_bar.bind('<Return>', self._formula_bar_commit)

        # ── Grid ──────────────────────────────────────────────────────────────
        grid_frame = ctk.CTkFrame(self, fg_color="transparent")
        grid_frame.pack(fill="both", expand=True, padx=6, pady=4)

        # Scrollbars
        self._vsb = ttk.Scrollbar(grid_frame, orient="vertical")
        self._hsb = ttk.Scrollbar(grid_frame, orient="horizontal")

        self._tree = ttk.Treeview(
            grid_frame,
            style='Editor.Treeview',
            selectmode='extended',
            yscrollcommand=self._vsb.set,
            xscrollcommand=self._hsb.set,
            show='headings',
        )
        self._vsb.configure(command=self._tree.yview)
        self._hsb.configure(command=self._tree.xview)

        self._vsb.pack(side="right", fill="y")
        self._hsb.pack(side="bottom", fill="x")
        self._tree.pack(fill="both", expand=True)

        # Bindings
        self._tree.bind('<Double-1>', self._on_cell_double_click)
        self._tree.bind('<ButtonRelease-1>', self._on_cell_select)
        self._tree.bind('<Return>', self._on_return_key)
        self._tree.bind('<Delete>', lambda e: self._clear_selected_cells())
        self._tree.bind('<Button-3>', self._on_right_click)
        self._tree.bind('<Control-z>', lambda e: None)  # placeholder undo

        # Barra de status
        self._status_var = tk.StringVar(value="Pronto")
        status = ctk.CTkLabel(self, textvariable=self._status_var,
                              font=ctk.CTkFont(size=10), text_color="gray",
                              anchor="w")
        status.pack(fill="x", padx=10, pady=(0, 4))

    # ── Dados internos ────────────────────────────────────────────────────────

    def _init_empty(self, cols: int = 3, rows: int = 20):
        self._columns = [f"Col{i+1}" for i in range(cols)]
        self._data = [[""] * cols for _ in range(rows)]
        self._rebuild_tree()

    def _load_dataframe(self, df):
        pd = _get_pd()
        self._columns = list(df.columns)
        self._data = []
        for _, row in df.iterrows():
            self._data.append([format_value(v) for v in row])
        self._rebuild_tree()
        self._set_status(f"Carregado: {len(self._data)} linhas × {len(self._columns)} colunas")

    def get_dataframe(self):
        pd = _get_pd()
        rows = []
        for row in self._data:
            record = {}
            for j, col in enumerate(self._columns):
                val = row[j] if j < len(row) else ''
                try:
                    record[col] = float(val) if val not in ('', None) else None
                except (ValueError, TypeError):
                    record[col] = val
            rows.append(record)
        return pd.DataFrame(rows, columns=self._columns)

    # ── Treeview ──────────────────────────────────────────────────────────────

    def _rebuild_tree(self):
        # Destroys existing entries
        self._tree.delete(*self._tree.get_children())

        # Configure columns
        col_ids = [f'c{i}' for i in range(len(self._columns))]
        self._tree['columns'] = col_ids

        for i, (cid, name) in enumerate(zip(col_ids, self._columns)):
            self._tree.heading(cid, text=name,
                               command=lambda idx=i: self._on_header_click(idx))
            self._tree.column(cid, width=self.COL_WIDTH, minwidth=60, stretch=True, anchor='center')

        # Insert rows
        for r_idx, row in enumerate(self._data):
            vals = []
            for j in range(len(self._columns)):
                vals.append(row[j] if j < len(row) else '')
            iid = self._tree.insert('', 'end', iid=f'r{r_idx}', values=vals)

        self._update_status_bar()

    def _refresh_row(self, r_idx: int):
        row = self._data[r_idx]
        vals = [row[j] if j < len(row) else '' for j in range(len(self._columns))]
        self._tree.item(f'r{r_idx}', values=vals)

    def _refresh_all(self):
        for r_idx in range(len(self._data)):
            self._refresh_row(r_idx)

    # ── Edição de células ─────────────────────────────────────────────────────

    def _on_cell_select(self, event):
        """Atualiza barra de fórmula ao selecionar célula."""
        region = self._tree.identify_region(event.x, event.y)
        if region != 'cell':
            return
        col_id = self._tree.identify_column(event.x)
        item = self._tree.identify_row(event.y)
        if not item or not col_id:
            return

        col_idx = int(col_id.replace('#', '')) - 1
        r_idx = self._row_index(item)
        if r_idx is None or col_idx >= len(self._columns):
            return

        val = self._data[r_idx][col_idx] if col_idx < len(self._data[r_idx]) else ''
        self._formula_var.set(str(val))
        self._selected_col_idx = col_idx

    def _on_return_key(self, event):
        """Abre edição na célula selecionada via teclado."""
        sel = self._tree.selection()
        if not sel:
            return
        item = sel[0]
        col_idx = self._selected_col_idx if self._selected_col_idx is not None else 0
        self._start_cell_edit(item, col_idx)

    def _on_cell_double_click(self, event):
        """Abre Entry overlay para edição inline via clique duplo."""
        region = self._tree.identify_region(event.x, event.y)
        if region == 'heading':
            col_id = self._tree.identify_column(event.x)
            idx = int(col_id.replace('#', '')) - 1
            self._rename_column_by_idx(idx)
            return
        if region != 'cell':
            return
        col_id = self._tree.identify_column(event.x)
        item = self._tree.identify_row(event.y)
        if not item or not col_id:
            return
        col_idx = int(col_id.replace('#', '')) - 1
        self._start_cell_edit(item, col_idx)

    def _start_cell_edit(self, item: str, col_idx: int):
        r_idx = self._row_index(item)
        if r_idx is None or col_idx >= len(self._columns):
            return

        try:
            bbox = self._tree.bbox(item, f'c{col_idx}')
        except Exception:
            return
        if not bbox:
            return

        x, y, w, h = bbox

        # Destroy existing entry
        if self._cell_entry:
            self._cell_entry.destroy()
            self._cell_entry = None

        entry = tk.Entry(self._tree,
                         font=('Segoe UI', 10),
                         bg='#1a4f7a', fg='white',
                         insertbackground='white',
                         relief='flat', bd=0)
        entry.place(x=x, y=y, width=w, height=h)

        current = self._data[r_idx][col_idx] if col_idx < len(self._data[r_idx]) else ''
        entry.insert(0, str(current))
        entry.select_range(0, 'end')
        entry.focus_set()
        self._cell_entry = entry

        def commit(event=None):
            val = entry.get()
            self._set_cell(r_idx, col_idx, val)
            entry.destroy()
            self._cell_entry = None
            self._formula_var.set(val)
            # Move to next row on Enter
            if event and event.keysym == 'Return':
                next_item = f'r{r_idx + 1}'
                if self._tree.exists(next_item):
                    self._tree.selection_set(next_item)
                    self._tree.see(next_item)
            elif event and event.keysym == 'Tab':
                self._move_focus(r_idx, col_idx, 0, 1)

        def cancel(event=None):
            entry.destroy()
            self._cell_entry = None

        entry.bind('<Return>', commit)
        entry.bind('<Tab>', commit)
        entry.bind('<Escape>', cancel)
        entry.bind('<FocusOut>', commit)

    def _set_cell(self, r_idx: int, col_idx: int, val: str):
        while len(self._data) <= r_idx:
            self._data.append([""] * len(self._columns))
        while len(self._data[r_idx]) <= col_idx:
            self._data[r_idx].append("")

        # Fórmula de coluna inteira?
        if val.startswith('='):
            self._col_formulas[self._columns[col_idx]] = val[1:]
        self._data[r_idx][col_idx] = val
        self._refresh_row(r_idx)
        self._update_status_bar()

    def _move_focus(self, r_idx, col_idx, dr, dc):
        new_r = r_idx + dr
        new_c = col_idx + dc
        if new_c >= len(self._columns):
            new_c = 0
            new_r += 1
        item = f'r{new_r}'
        if self._tree.exists(item):
            self._tree.selection_set(item)
            self._tree.see(item)
            self._selected_col_idx = new_c

    def _formula_bar_commit(self, event=None):
        """Aplica o conteúdo da barra de fórmula na célula selecionada."""
        sel = self._tree.selection()
        if not sel or self._selected_col_idx is None:
            return
        item = sel[0]
        r_idx = self._row_index(item)
        if r_idx is None:
            return
        val = self._formula_var.get()
        self._set_cell(r_idx, self._selected_col_idx, val)

    def _on_header_click(self, col_idx: int):
        self._selected_col_idx = col_idx
        self._formula_var.set(self._col_formulas.get(self._columns[col_idx], ''))

    def _on_right_click(self, event):
        menu = tk.Menu(self._tree, tearoff=0,
                       bg='#2b2b2b', fg='white',
                       activebackground='#2E86DE', activeforeground='white')
        menu.add_command(label="✏️  Renomear Coluna", command=self._rename_selected_column)
        menu.add_command(label="🗑  Excluir Coluna", command=self._delete_selected_column)
        menu.add_separator()
        menu.add_command(label="➕  Inserir Linha Acima", command=self._insert_row_above)
        menu.add_command(label="🗑  Excluir Linhas Selecionadas", command=self._delete_selected_rows)
        menu.add_separator()
        menu.add_command(label="🔢  Gerar Dados", command=self._open_generate_dialog)
        menu.add_command(label="🧹  Limpar Células", command=self._clear_selected_cells)
        menu.post(event.x_root, event.y_root)

    # ── Gerenciamento de colunas ───────────────────────────────────────────────

    def _add_column_dialog(self):
        name = simpledialog.askstring("Nova Coluna", "Nome da coluna:", parent=self)
        if not name:
            return
        if name in self._columns:
            messagebox.showerror("Erro", f"Coluna '{name}' já existe.", parent=self)
            return
        self._columns.append(name)
        for row in self._data:
            row.append("")
        self._rebuild_tree()
        self._set_status(f"Coluna '{name}' adicionada.")

    def _rename_selected_column(self):
        if self._selected_col_idx is None or self._selected_col_idx >= len(self._columns):
            messagebox.showinfo("Aviso", "Clique em um cabeçalho de coluna primeiro.", parent=self)
            return
        self._rename_column_by_idx(self._selected_col_idx)

    def _rename_column_by_idx(self, idx: int):
        old = self._columns[idx]
        new = simpledialog.askstring("Renomear Coluna", f"Novo nome para '{old}':", initialvalue=old, parent=self)
        if not new or new == old:
            return
        if new in self._columns:
            messagebox.showerror("Erro", f"Coluna '{new}' já existe.", parent=self)
            return
        if old in self._col_formulas:
            self._col_formulas[new] = self._col_formulas.pop(old)
        self._columns[idx] = new
        self._rebuild_tree()

    def _delete_selected_column(self):
        if self._selected_col_idx is None or len(self._columns) <= 1:
            messagebox.showinfo("Aviso", "Não é possível excluir a última coluna.", parent=self)
            return
        name = self._columns[self._selected_col_idx]
        if not messagebox.askyesno("Confirmar", f"Excluir coluna '{name}'?", parent=self):
            return
        idx = self._selected_col_idx
        self._columns.pop(idx)
        for row in self._data:
            if idx < len(row):
                row.pop(idx)
        self._col_formulas.pop(name, None)
        self._selected_col_idx = None
        self._rebuild_tree()

    # ── Gerenciamento de linhas ────────────────────────────────────────────────

    def _add_rows_dialog(self):
        n_str = simpledialog.askstring("Adicionar Linhas", "Quantas linhas adicionar?",
                                       initialvalue="10", parent=self)
        if not n_str:
            return
        try:
            n = max(1, int(n_str))
        except ValueError:
            return
        for _ in range(n):
            self._data.append([""] * len(self._columns))
        self._rebuild_tree()
        self._set_status(f"{n} linhas adicionadas. Total: {len(self._data)}")

    def _insert_row_above(self):
        sel = self._tree.selection()
        idx = self._row_index(sel[0]) if sel else len(self._data)
        self._data.insert(idx, [""] * len(self._columns))
        self._rebuild_tree()

    def _delete_selected_rows(self):
        sel = self._tree.selection()
        if not sel:
            return
        indices = sorted([self._row_index(i) for i in sel if self._row_index(i) is not None], reverse=True)
        for idx in indices:
            if 0 <= idx < len(self._data):
                self._data.pop(idx)
        if not self._data:
            self._data.append([""] * len(self._columns))
        self._rebuild_tree()
        self._set_status(f"{len(indices)} linha(s) excluída(s).")

    def _clear_selected_cells(self):
        sel = self._tree.selection()
        if not sel or self._selected_col_idx is None:
            return
        for item in sel:
            r_idx = self._row_index(item)
            if r_idx is not None and self._selected_col_idx < len(self._data[r_idx]):
                self._data[r_idx][self._selected_col_idx] = ""
        self._refresh_all()

    def _clear_all(self):
        if not messagebox.askyesno("Limpar Tudo", "Apagar todos os dados?", parent=self):
            return
        self._col_formulas.clear()
        self._init_empty(cols=len(self._columns), rows=20)

    # ── Geração de dados ──────────────────────────────────────────────────────

    def _open_generate_dialog(self):
        cols_data = self._columns_as_dict()
        n = len(self._data)

        def on_apply(col_name: str, values: list):
            if col_name not in self._columns:
                self._columns.append(col_name)
                for row in self._data:
                    row.append("")

            col_idx = self._columns.index(col_name)

            # Expande linhas se necessário
            while len(self._data) < len(values):
                self._data.append([""] * len(self._columns))

            for i, val in enumerate(values):
                if i >= len(self._data):
                    break
                while len(self._data[i]) <= col_idx:
                    self._data[i].append("")
                self._data[i][col_idx] = format_value(val)

            self._rebuild_tree()
            self._set_status(f"Coluna '{col_name}' gerada com {len(values)} valores.")

        _GenerateDialog(self, list(self._columns), n, cols_data, on_apply)

    def _columns_as_dict(self) -> dict:
        """Retorna colunas atuais como dict {nome: [valores]}."""
        result = {}
        for j, col in enumerate(self._columns):
            vals = []
            for row in self._data:
                v = row[j] if j < len(row) else ''
                vals.append(v)
            result[col] = vals
        return result

    # ── Import / Export ───────────────────────────────────────────────────────

    def _import_file(self):
        path = filedialog.askopenfilename(
            title="Abrir arquivo",
            filetypes=[("Excel/CSV", "*.xlsx *.xls *.csv"), ("Todos", "*.*")],
            parent=self)
        if not path:
            return
        pd = _get_pd()
        try:
            if path.endswith('.csv'):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
            self._load_dataframe(df)
        except Exception as e:
            messagebox.showerror("Erro", str(e), parent=self)

    def _export_csv(self):
        path = filedialog.asksaveasfilename(
            title="Salvar CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            parent=self)
        if not path:
            return
        try:
            df = self.get_dataframe()
            df.to_csv(path, index=False)
            self._set_status(f"Exportado: {path}")
        except Exception as e:
            messagebox.showerror("Erro", str(e), parent=self)

    def _use_in_analysis(self):
        df = self.get_dataframe()
        if df.empty:
            messagebox.showinfo("Aviso", "Nenhum dado para usar.", parent=self)
            return
        if self._on_use_data:
            self._on_use_data(df)
            messagebox.showinfo("Sucesso",
                f"Dados enviados para análise!\n{len(df)} linhas × {len(df.columns)} colunas",
                parent=self)
        else:
            messagebox.showinfo("Dados Prontos",
                f"DataFrame gerado: {len(df)} linhas × {len(df.columns)} colunas\n\n"
                "Feche esta janela e os dados estarão disponíveis.", parent=self)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _row_index(self, item: str) -> int | None:
        try:
            return int(item.replace('r', ''))
        except (ValueError, AttributeError):
            return None

    def _set_status(self, msg: str):
        self._status_var.set(msg)

    def _update_status_bar(self):
        rows = len(self._data)
        cols = len(self._columns)
        non_empty = sum(1 for row in self._data for v in row if v not in ('', None))
        self._status_var.set(f"{rows} linhas × {cols} colunas  |  {non_empty} células preenchidas")
