"""
InlineSpreadsheet — planilha embutida com edição e paste do Excel (estilo JMP)
"""
import tkinter as tk
from tkinter import ttk, simpledialog
import customtkinter as ctk
import pandas as pd


class InlineSpreadsheet(ctk.CTkFrame):
    """
    Spreadsheet embutido com edição inline, paste do Excel e menu de contexto.
    Comportamento similar ao data table do JMP.
    """
    _MIN_ROWS = 25
    _MIN_COLS = 8
    _COL_WIDTH = 110
    _ROW_HEIGHT = 26
    _ROW_NUM_WIDTH = 42

    # Teclas que não devem disparar início de edição por digitação direta
    _NAV_KEYSYMS = {
        "Up", "Down", "Left", "Right", "Tab", "Return", "Escape", "F2", "Delete",
        "Home", "End", "Prior", "Next", "Shift_L", "Shift_R", "Control_L", "Control_R",
        "Alt_L", "Alt_R", "Caps_Lock",
    }

    def __init__(self, parent, on_data_change=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.on_data_change = on_data_change
        self._col_names: list[str] = []
        self._editing_entry: tk.Entry | None = None
        self._editing_info: tuple | None = None
        self._ctx_col_idx: int = 1

        # Célula/coluna ativa (seleção estilo Excel)
        self._sel_mode: str = "cell"          # "cell" | "column" | "rows" | "range"
        self._active_row_id: str | None = None
        self._active_col_idx: int | None = None

        # Intervalo retangular selecionado por arrasto do mouse (linha_topo, linha_baixo)
        self._range_rows: tuple[str, str] | None = None
        self._range_cols: tuple[int, int] | None = None
        self._drag_anchor: tuple[str, int] | None = None

        self._setup_style()
        self._setup_ui()
        self._setup_context_menu()
        self._setup_bindings()
        self._init_empty_grid()

        first = self.tree.get_children()
        if first:
            self._set_active(first[0], 2)

    # ── Setup ─────────────────────────────────────────────────────────────────

    _ROW_BG_EVEN = "#FFFFFF"
    _ROW_BG_ODD = "#F5F8FC"
    _GRID_LINE = "#DCE3ED"

    def _setup_style(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("Sheet.Treeview",
                    background=self._ROW_BG_EVEN,
                    foreground="#1A2733",
                    fieldbackground=self._ROW_BG_EVEN,
                    rowheight=self._ROW_HEIGHT,
                    font=("Segoe UI", 10),
                    borderwidth=1,
                    relief="solid",
                    bordercolor=self._GRID_LINE)
        s.configure("Sheet.Treeview.Heading",
                    background="#EAF0F8",
                    foreground="#2D3748",
                    font=("Segoe UI", 10, "bold"),
                    relief="flat", borderwidth=1,
                    bordercolor=self._GRID_LINE)
        s.map("Sheet.Treeview.Heading",
              background=[("active", "#D6E3F0")])
        s.map("Sheet.Treeview",
              background=[("selected", "#CFE4FA")],
              foreground=[("selected", "#0F2A44")])

    def _setup_ui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        wrap = tk.Frame(self, bg="white")
        wrap.grid(row=0, column=0, sticky="nsew")
        wrap.grid_rowconfigure(0, weight=1)
        wrap.grid_columnconfigure(0, weight=1)

        self.tree = ttk.Treeview(wrap, style="Sheet.Treeview",
                                  show="headings", selectmode="extended")
        self.tree.tag_configure("evenrow", background=self._ROW_BG_EVEN)
        self.tree.tag_configure("oddrow", background=self._ROW_BG_ODD)
        vsb = ttk.Scrollbar(wrap, orient="vertical",   command=self.tree.yview)
        hsb = ttk.Scrollbar(wrap, orient="horizontal", command=self.tree.xview)

        def _on_yscroll(*args):
            vsb.set(*args)
            self._reposition_highlight()

        def _on_xscroll(*args):
            hsb.set(*args)
            self._reposition_highlight()

        self.tree.configure(yscrollcommand=_on_yscroll, xscrollcommand=_on_xscroll)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        # Borda de destaque da célula/coluna ativa (4 tarjas finas sobrepostas)
        accent = "#2E86DE"
        self._hl_top = tk.Frame(self.tree, bg=accent)
        self._hl_bottom = tk.Frame(self.tree, bg=accent)
        self._hl_left = tk.Frame(self.tree, bg=accent)
        self._hl_right = tk.Frame(self.tree, bg=accent)
        self._hl_frames = (self._hl_top, self._hl_bottom, self._hl_left, self._hl_right)

        self.tree.bind("<Configure>", lambda e: self._reposition_highlight(), add="+")

    def _setup_context_menu(self):
        m = tk.Menu(
            self, tearoff=0,
            font=("Segoe UI", 10),
            bg="#FFFFFF", fg="#1A2733",
            activebackground="#2E86DE", activeforeground="#FFFFFF",
            disabledforeground="#A5B2C2",
            bd=0, relief="flat",
            activeborderwidth=4,
        )
        m.add_command(label="📋  Copiar",            accelerator="Ctrl+C", command=self._copy)
        m.add_command(label="📥  Colar",             accelerator="Ctrl+V", command=self._paste)
        m.add_command(label="✂️  Recortar",          accelerator="Ctrl+X", command=self._cut)
        m.add_separator()
        m.add_command(label="⬆️  Inserir linha acima",   command=lambda: self._insert_row(above=True))
        m.add_command(label="⬇️  Inserir linha abaixo",  command=lambda: self._insert_row(above=False))
        m.add_command(label="🗑️  Excluir linha(s)",      command=self._delete_selected_rows)
        m.add_separator()
        m.add_command(label="⬅️  Inserir coluna à esquerda", command=lambda: self._insert_col(left=True))
        m.add_command(label="➡️  Inserir coluna à direita",  command=lambda: self._insert_col(left=False))
        m.add_command(label="🗑️  Excluir coluna",            command=self._delete_col)
        m.add_command(label="✏️  Renomear coluna",           command=self._rename_col)
        m.add_separator()
        m.add_command(label="🔼  Ordenar crescente",  command=lambda: self._sort_col(ascending=True))
        m.add_command(label="🔽  Ordenar decrescente", command=lambda: self._sort_col(ascending=False))
        m.add_separator()
        m.add_command(label="🧹  Limpar células",     command=self._clear_selected_cells)
        m.add_command(label="🔲  Selecionar tudo", accelerator="Ctrl+A", command=self._select_all)
        self.ctx_menu = m

    def _setup_bindings(self):
        t = self.tree
        t.bind("<Double-1>",   self._on_double_click)
        t.bind("<Button-1>",   self._on_left_click, add="+")
        t.bind("<B1-Motion>",       self._on_mouse_drag)
        t.bind("<ButtonRelease-1>", self._on_mouse_release)
        t.bind("<Button-3>",   self._show_context_menu)
        t.bind("<Control-v>",  self._paste)
        t.bind("<Control-V>",  self._paste)
        t.bind("<Control-c>",  self._copy)
        t.bind("<Control-C>",  self._copy)
        t.bind("<Control-x>",  self._cut)
        t.bind("<Control-X>",  self._cut)
        t.bind("<Control-a>",  self._select_all)
        t.bind("<Control-A>",  self._select_all)
        t.bind("<Delete>",     self._clear_selected_cells)
        t.bind("<Return>",     self._on_return_key)
        t.bind("<F2>",         self._on_f2_key)

        # ── Navegação estilo Excel ────────────────────────────────────────
        t.bind("<Up>",    lambda e: self._move_active(-1, 0))
        t.bind("<Down>",  lambda e: self._move_active(1, 0))
        t.bind("<Left>",  lambda e: self._move_active(0, -1))
        t.bind("<Right>", lambda e: self._move_active(0, 1))
        t.bind("<Tab>",       lambda e: self._tab_move(forward=True))
        t.bind("<Shift-Tab>", lambda e: self._tab_move(forward=False))
        t.bind("<Home>",         self._move_home)
        t.bind("<End>",          self._move_end)
        t.bind("<Control-Home>", self._move_ctrl_home)
        t.bind("<Control-End>",  self._move_ctrl_end)
        t.bind("<Prior>", lambda e: self._move_page(forward=False))  # Page Up
        t.bind("<Next>",  lambda e: self._move_page(forward=True))   # Page Down
        t.bind("<Control-space>", self._select_active_column)
        t.bind("<Shift-space>",   self._select_active_row)

        # Digitar diretamente sobre a célula ativa já inicia a edição
        t.bind("<Key>", self._on_key_type)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_dataframe(self, df: pd.DataFrame):
        """Carrega um DataFrame no spreadsheet."""
        self._abort_edit()
        self._clear_tree()
        self._set_columns(list(df.columns))

        for i, (_, row) in enumerate(df.iterrows(), 1):
            vals = [str(i)] + [("" if pd.isna(v) else str(v)) for v in row]
            self.tree.insert("", "end", values=vals, tags=(self._row_tag(i),))

        n = len(df)
        for i in range(self._MIN_ROWS):
            pos = n + i + 1
            self.tree.insert("", "end",
                             values=[str(pos)] + [""] * len(df.columns),
                             tags=(self._row_tag(pos),))

    def get_dataframe(self) -> pd.DataFrame:
        """Retorna o conteúdo atual como DataFrame (ignora linhas vazias)."""
        if not self._col_names:
            return pd.DataFrame()

        rows = []
        for item in self.tree.get_children():
            raw = list(self.tree.item(item, "values"))
            data = raw[1:]                         # skip row number
            data = (data + [""] * len(self._col_names))[:len(self._col_names)]
            if any(str(v).strip() for v in data):
                rows.append(data)

        if not rows:
            return pd.DataFrame(columns=self._col_names)

        df = pd.DataFrame(rows, columns=self._col_names)

        # Remove colunas totalmente vazias (placeholders que o usuário nunca preencheu)
        filled_cols = [c for c in df.columns if df[c].astype(str).str.strip().any()]
        df = df[filled_cols]

        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="raise")
            except (ValueError, TypeError):
                pass
        return df

    def commit_pending_edit(self):
        """Confirma a edição de célula em andamento, se houver, antes de ler os dados."""
        if self._editing_entry:
            self._commit_edit()

    # ── Columns ───────────────────────────────────────────────────────────────

    def _set_columns(self, names: list[str]):
        self._col_names = list(names)
        all_cols = ["__row__"] + list(names)
        self.tree["columns"] = all_cols

        self.tree.heading("__row__", text="#")
        self.tree.column("__row__", width=self._ROW_NUM_WIDTH, anchor="center",
                         stretch=False, minwidth=self._ROW_NUM_WIDTH)

        for name in names:
            self.tree.heading(name, text=name, anchor="w")
            self.tree.column(name, width=self._COL_WIDTH, anchor="w",
                             minwidth=50, stretch=True)

    def _init_empty_grid(self):
        cols = [f"C{i+1}" for i in range(self._MIN_COLS)]
        self._set_columns(cols)
        for i in range(self._MIN_ROWS):
            self.tree.insert("", "end", values=[str(i+1)] + [""] * len(cols),
                             tags=(self._row_tag(i + 1),))

    def _extend_to_width(self, needed_data_cols: int):
        """Garante que a tabela tem pelo menos needed_data_cols colunas de dados."""
        if needed_data_cols <= len(self._col_names):
            return
        extra = needed_data_cols - len(self._col_names)
        new_names = self._col_names + [f"C{len(self._col_names)+i+1}" for i in range(extra)]

        all_rows = [list(self.tree.item(it, "values")) for it in self.tree.get_children()]
        self._clear_tree()
        self._set_columns(new_names)
        for i, vals in enumerate(all_rows, 1):
            row_num = vals[0] if vals else ""
            data = vals[1:] if len(vals) > 1 else []
            data = (data + [""] * len(new_names))[:len(new_names)]
            self.tree.insert("", "end", values=[row_num] + data, tags=(self._row_tag(i),))

    def _rebuild_with_cols(self, new_names: list[str], all_data_rows: list[list]):
        """Reconstrói o Treeview com novas colunas e dados fornecidos."""
        self._clear_tree()
        self._set_columns(new_names)
        for i, data in enumerate(all_data_rows, 1):
            data = (data + [""] * len(new_names))[:len(new_names)]
            self.tree.insert("", "end", values=[str(i)] + data, tags=(self._row_tag(i),))

    # ── Editing ───────────────────────────────────────────────────────────────

    def _on_double_click(self, event):
        region = self.tree.identify_region(event.x, event.y)
        col    = self.tree.identify_column(event.x)
        row_id = self.tree.identify_row(event.y)

        if region == "heading":
            col_idx = int(col.replace("#", ""))
            if col_idx > 1:                        # not row-number column
                self._rename_col_at(col_idx)
        elif region == "cell" and row_id and col:
            self._start_edit(row_id, col)

    def _on_return_key(self, event):
        if self._active_row_id and self._active_col_idx:
            self._start_edit(self._active_row_id, f"#{self._active_col_idx}")
            return
        sel = self.tree.selection()
        if sel:
            self._start_edit(sel[0], "#2")

    def _on_f2_key(self, event):
        self._on_return_key(event)

    def _start_edit(self, row_id: str, col: str):
        self._abort_edit()

        col_idx = int(col.replace("#", ""))        # 1-based; 1 = row number col
        if col_idx == 1:
            return

        bbox = self.tree.bbox(row_id, col)
        if not bbox:
            return
        x, y, w, h = bbox

        vals = list(self.tree.item(row_id, "values"))
        current = vals[col_idx - 1] if col_idx - 1 < len(vals) else ""

        e = tk.Entry(self.tree, font=("Segoe UI", 9), relief="solid", bd=1,
                     highlightthickness=1, highlightcolor="#2E86DE",
                     highlightbackground="#2E86DE")
        e.place(x=x, y=y, width=w, height=h)
        e.insert(0, current)
        e.select_range(0, "end")
        e.focus_set()

        self._editing_entry = e
        self._editing_info  = (row_id, col_idx)

        e.bind("<Return>",   lambda ev: self._commit_edit(move_down=True))
        e.bind("<Tab>",      lambda ev: self._commit_edit(move_right=True))
        e.bind("<Escape>",   lambda ev: self._abort_edit())
        e.bind("<FocusOut>", lambda ev: self._commit_edit())

    def _commit_edit(self, move_down=False, move_right=False):
        if not self._editing_entry or not self._editing_entry.winfo_exists():
            return
        if not self._editing_info:
            return

        row_id, col_idx = self._editing_info
        new_val = self._editing_entry.get()

        try:
            vals = list(self.tree.item(row_id, "values"))
            while len(vals) <= col_idx - 1:
                vals.append("")
            vals[col_idx - 1] = new_val
            self.tree.item(row_id, values=vals)
        except Exception:
            pass

        self._editing_entry.destroy()
        self._editing_entry = None
        self._editing_info  = None
        self._notify_change()

        if move_down:
            self._set_active(row_id, col_idx)
            self._move_active(1, 0)
        elif move_right:
            self._set_active(row_id, col_idx)
            self._tab_move(forward=True)
        else:
            self._set_active(row_id, col_idx)

    def _abort_edit(self):
        if self._editing_entry and self._editing_entry.winfo_exists():
            self._editing_entry.destroy()
        self._editing_entry = None
        self._editing_info  = None

    # ── Copy / Paste / Cut ───────────────────────────────────────────────────

    def _copy(self, event=None):
        if self._sel_mode == "cell" and self._active_row_id:
            vals = list(self.tree.item(self._active_row_id, "values"))
            idx = self._active_col_idx - 1
            text = vals[idx] if idx < len(vals) else ""
            self.clipboard_clear()
            self.clipboard_append(str(text))
            return

        if self._sel_mode == "column" and self._active_col_idx:
            idx = self._active_col_idx - 1
            lines = []
            for it in self.tree.get_children():
                vals = list(self.tree.item(it, "values"))
                lines.append(str(vals[idx]) if idx < len(vals) else "")
            self.clipboard_clear()
            self.clipboard_append("\n".join(lines))
            return

        if self._sel_mode == "range" and self._range_rows and self._range_cols:
            lines = []
            for it in self._range_row_ids():
                vals = list(self.tree.item(it, "values"))
                row_vals = [
                    (str(vals[ci - 1]) if ci - 1 < len(vals) else "")
                    for ci in range(self._range_cols[0], self._range_cols[1] + 1)
                ]
                lines.append("\t".join(row_vals))
            self.clipboard_clear()
            self.clipboard_append("\n".join(lines))
            return

        sel = self.tree.selection()
        if not sel:
            return
        lines = []
        for item in sel:
            vals = list(self.tree.item(item, "values"))[1:]
            lines.append("\t".join(str(v) for v in vals))
        self.clipboard_clear()
        self.clipboard_append("\n".join(lines))

    def _cut(self, event=None):
        self._copy()
        self._clear_selected_cells()

    def _paste(self, event=None):
        try:
            text = self.clipboard_get()
        except Exception:
            return
        if not text.strip():
            return

        raw_rows = [line.split("\t") for line in text.strip().split("\n")]

        # Ponto de início a partir da célula ativa (captura antes de qualquer
        # rebuild de colunas, já que ele reseta a seleção)
        sel = self.tree.selection()
        all_items = list(self.tree.get_children())
        if self._sel_mode == "range" and self._range_rows and self._range_rows[0] in all_items:
            start_idx = all_items.index(self._range_rows[0])
            start_col_offset = max(0, self._range_cols[0] - 2)
        elif self._sel_mode == "cell" and self._active_row_id in all_items:
            start_idx = all_items.index(self._active_row_id)
            start_col_offset = max(0, (self._active_col_idx or 2) - 2)
        else:
            start_idx = (all_items.index(sel[0]) if sel else 0)
            start_col_offset = 0

        # Detect if first row is a header
        has_header = (
            len(raw_rows) > 1
            and all(not self._is_number(v) for v in raw_rows[0] if v.strip())
            and any(self._is_number(v) for v in raw_rows[1] if v.strip())
        )

        if has_header:
            header = [v.strip() for v in raw_rows[0]]
            data_rows = raw_rows[1:]
            # Rebuild columns with new header
            existing_data = [
                list(self.tree.item(it, "values"))[1:]
                for it in self.tree.get_children()
            ]
            # Extend existing_data to match new header width
            needed = max(len(header), len(self._col_names))
            new_names = (header + [f"C{i+1}" for i in range(len(header), needed)])[:needed]
            self._rebuild_with_cols(new_names, existing_data)
            start_col_offset = 0  # header substitui as colunas: cola a partir da 1ª
        else:
            data_rows = raw_rows

        max_cols = max((len(r) for r in data_rows), default=0)
        self._extend_to_width(start_col_offset + max_cols)
        all_items = list(self.tree.get_children())   # refresh after possible rebuild

        for r_off, row_vals in enumerate(data_rows):
            item_idx = start_idx + r_off
            while item_idx >= len(all_items):
                n = len(all_items) + 1
                new_it = self.tree.insert("", "end",
                                          values=[str(n)] + [""] * len(self._col_names))
                all_items.append(new_it)

            item = all_items[item_idx]
            vals = list(self.tree.item(item, "values"))

            for c_off, v in enumerate(row_vals):
                ci = 1 + start_col_offset + c_off   # index in values (0 = row num)
                while len(vals) <= ci:
                    vals.append("")
                vals[ci] = v.strip()

            self.tree.item(item, values=vals)

        self._update_row_numbers()
        self._notify_change()

    # ── Row operations ────────────────────────────────────────────────────────

    def _insert_row(self, above=True):
        sel = self.tree.selection()
        all_items = list(self.tree.get_children())
        if sel:
            ref_idx = all_items.index(sel[0])
            pos = ref_idx if above else ref_idx + 1
        else:
            pos = len(all_items)

        empty = [""] * (len(self._col_names) + 1)
        if pos >= len(all_items):
            self.tree.insert("", "end", values=empty)
        else:
            self.tree.insert("", pos, values=empty)

        self._update_row_numbers()
        self._notify_change()

    def _delete_selected_rows(self):
        sel = self.tree.selection()
        if not sel:
            return
        if self._active_row_id in sel:
            self._active_row_id = None
            self._hide_highlight()
        for it in sel:
            self.tree.delete(it)
        # Keep minimum rows
        n = len(self.tree.get_children())
        for _ in range(max(0, self._MIN_ROWS - n)):
            self.tree.insert("", "end", values=[""] * (len(self._col_names) + 1))
        self._update_row_numbers()
        self._notify_change()

    # ── Column operations ─────────────────────────────────────────────────────

    def _insert_col(self, left=True):
        ci = self._ctx_col_idx          # 1-based Treeview col (1 = row num)
        data_ci = ci - 2 if ci > 1 else 0   # 0-based index into _col_names
        insert_at = data_ci if left else data_ci + 1
        insert_at = max(0, min(insert_at, len(self._col_names)))

        new_name = f"C{len(self._col_names)+1}"
        new_names = self._col_names[:]
        new_names.insert(insert_at, new_name)

        all_data = []
        for it in self.tree.get_children():
            row = list(self.tree.item(it, "values"))[1:]
            row.insert(insert_at, "")
            all_data.append(row)

        self._rebuild_with_cols(new_names, all_data)
        self._notify_change()

    def _delete_col(self):
        ci = self._ctx_col_idx
        if ci <= 1 or len(self._col_names) <= 1:
            return
        data_ci = ci - 2           # 0-based

        new_names = [c for i, c in enumerate(self._col_names) if i != data_ci]
        all_data = []
        for it in self.tree.get_children():
            row = list(self.tree.item(it, "values"))[1:]
            row = [v for i, v in enumerate(row) if i != data_ci]
            all_data.append(row)

        self._rebuild_with_cols(new_names, all_data)
        self._notify_change()

    def _rename_col(self):
        self._rename_col_at(self._ctx_col_idx)

    def _rename_col_at(self, col_idx: int):
        data_ci = col_idx - 2
        if data_ci < 0 or data_ci >= len(self._col_names):
            return

        old_name = self._col_names[data_ci]
        new_name = simpledialog.askstring(
            "Renomear Coluna",
            f"Novo nome para '{old_name}':",
            initialvalue=old_name,
            parent=self
        )
        if new_name and new_name.strip() and new_name.strip() != old_name:
            new_name = new_name.strip()
            # Treeview columns are keyed by name, must rebuild
            new_names = self._col_names[:]
            new_names[data_ci] = new_name
            all_data = [
                list(self.tree.item(it, "values"))[1:]
                for it in self.tree.get_children()
            ]
            self._rebuild_with_cols(new_names, all_data)
            self._notify_change()

    def _sort_col(self, ascending=True):
        ci = self._ctx_col_idx
        if ci <= 1:
            return

        all_items = list(self.tree.get_children())

        def _key(it):
            vals = list(self.tree.item(it, "values"))
            v = vals[ci - 1] if ci - 1 < len(vals) else ""
            try:
                return (0, float(str(v).replace(",", ".")))
            except (ValueError, TypeError):
                return (1, str(v).lower())

        for idx, it in enumerate(sorted(all_items, key=_key, reverse=not ascending)):
            self.tree.move(it, "", idx)

        self._update_row_numbers()

    # ── Selection ─────────────────────────────────────────────────────────────

    def _clear_selected_cells(self, event=None):
        if self._sel_mode == "cell" and self._active_row_id:
            vals = list(self.tree.item(self._active_row_id, "values"))
            idx = self._active_col_idx - 1
            if idx < len(vals):
                vals[idx] = ""
                self.tree.item(self._active_row_id, values=vals)
            self._notify_change()
            return

        if self._sel_mode == "column" and self._active_col_idx:
            idx = self._active_col_idx - 1
            for it in self.tree.get_children():
                vals = list(self.tree.item(it, "values"))
                if idx < len(vals):
                    vals[idx] = ""
                    self.tree.item(it, values=vals)
            self._notify_change()
            return

        if self._sel_mode == "range" and self._range_rows and self._range_cols:
            for it in self._range_row_ids():
                vals = list(self.tree.item(it, "values"))
                for ci in range(self._range_cols[0], self._range_cols[1] + 1):
                    idx = ci - 1
                    if idx < len(vals):
                        vals[idx] = ""
                self.tree.item(it, values=vals)
            self._notify_change()
            return

        for it in self.tree.selection():
            vals = list(self.tree.item(it, "values"))
            new_vals = [vals[0]] + [""] * (len(vals) - 1)
            self.tree.item(it, values=new_vals)
        self._notify_change()

    def _select_all(self, event=None):
        self.tree.selection_set(self.tree.get_children())
        self._sel_mode = "rows"
        self._active_col_idx = None
        self._hide_highlight()

    # ── Navegação e seleção estilo Excel ─────────────────────────────────────

    def _on_left_click(self, event):
        region = self.tree.identify_region(event.x, event.y)
        shift = bool(event.state & 0x0001)
        ctrl = bool(event.state & 0x0004)

        if region == "heading":
            col = self.tree.identify_column(event.x)
            try:
                col_idx = int(col.replace("#", ""))
            except ValueError:
                return
            if col_idx == 1:
                self._select_all()
                return
            self._sel_mode = "column"
            self._active_row_id = None
            self._active_col_idx = col_idx
            self.tree.selection_remove(*self.tree.selection())
            self._highlight_column(col_idx)
            return

        if region != "cell":
            return

        row_id = self.tree.identify_row(event.y)
        col = self.tree.identify_column(event.x)
        if not row_id or not col:
            return
        col_idx = int(col.replace("#", ""))

        if col_idx == 1 or ctrl:
            self._sel_mode = "rows"
            self._active_row_id = None
            self._active_col_idx = None
            self._drag_anchor = None
            self._hide_highlight()
            return

        if shift and self._active_row_id and self._active_col_idx:
            self._set_range(self._active_row_id, self._active_col_idx, row_id, col_idx)
            self._drag_anchor = (self._active_row_id, self._active_col_idx)
            return

        self._drag_anchor = (row_id, col_idx)
        self._set_active(row_id, col_idx)

    def _on_mouse_drag(self, event):
        if not self._drag_anchor:
            return
        row_id = self.tree.identify_row(event.y)
        col = self.tree.identify_column(event.x)
        if not row_id or not col:
            return
        try:
            col_idx = int(col.replace("#", ""))
        except ValueError:
            return
        if col_idx == 1:
            col_idx = 2

        self.tree.see(row_id)
        anchor_row, anchor_col = self._drag_anchor
        if row_id == anchor_row and col_idx == anchor_col:
            self._set_active(row_id, col_idx)
            return
        self._set_range(anchor_row, anchor_col, row_id, col_idx)

    def _on_mouse_release(self, event):
        self._drag_anchor = None

    def _any_visible_bbox(self, children, col_idx: int):
        for it in children:
            bbox = self.tree.bbox(it, f"#{col_idx}")
            if bbox:
                return bbox
        return None

    def _set_range(self, row_a: str, col_a: int, row_b: str, col_b: int):
        children = list(self.tree.get_children())
        if row_a not in children or row_b not in children:
            return
        ia, ib = children.index(row_a), children.index(row_b)
        r0, r1 = min(ia, ib), max(ia, ib)
        c0, c1 = min(col_a, col_b), max(col_a, col_b)

        self._sel_mode = "range"
        self._range_rows = (children[r0], children[r1])
        self._range_cols = (c0, c1)
        self._active_row_id = row_b
        self._active_col_idx = col_b
        self.tree.selection_remove(*self.tree.selection())
        self._draw_range_highlight()

    def _draw_range_highlight(self):
        if not self._range_rows or not self._range_cols:
            return
        row_top, row_bottom = self._range_rows
        c0, c1 = self._range_cols
        children = list(self.tree.get_children())

        bbox_top = self.tree.bbox(row_top, f"#{c0}")
        bbox_bottom = self.tree.bbox(row_bottom, f"#{c1}")
        xb_left = bbox_top or self._any_visible_bbox(children, c0)
        xb_right = bbox_bottom or self._any_visible_bbox(children, c1)
        if not xb_left or not xb_right:
            self._hide_highlight()
            return

        x = min(xb_left[0], xb_right[0])
        w = max(xb_left[0] + xb_left[2], xb_right[0] + xb_right[2]) - x
        y = bbox_top[1] if bbox_top else 0
        y_bottom = (bbox_bottom[1] + bbox_bottom[3]) if bbox_bottom else self.tree.winfo_height()
        h = max(0, y_bottom - y)
        self._place_highlight(x, y, w, h)

    def _range_row_ids(self) -> list:
        """Retorna os iids das linhas cobertas pelo intervalo selecionado."""
        if not self._range_rows:
            return []
        children = list(self.tree.get_children())
        row_top, row_bottom = self._range_rows
        if row_top not in children or row_bottom not in children:
            return []
        i0, i1 = children.index(row_top), children.index(row_bottom)
        return children[i0:i1 + 1]

    def _set_active(self, row_id: str, col_idx: int):
        self._sel_mode = "cell"
        self._active_row_id = row_id
        self._active_col_idx = col_idx
        self.tree.selection_set(row_id)
        self.tree.focus(row_id)
        self.tree.see(row_id)
        self._highlight_cell(row_id, col_idx)

    def _active_row_index(self, children: list) -> int:
        if self._active_row_id and self._active_row_id in children:
            return children.index(self._active_row_id)
        sel = self.tree.selection()
        if sel and sel[0] in children:
            return children.index(sel[0])
        return 0

    def _move_active(self, d_row: int, d_col: int):
        """Movimento por seta: sem wrap, estende a grade ao passar do limite."""
        if self._editing_entry:
            self._commit_edit()

        children = list(self.tree.get_children())
        if not children:
            return "break"

        row_idx = self._active_row_index(children)
        col_idx = self._active_col_idx or 2

        if d_col:
            col_idx += d_col
            if col_idx > len(self._col_names) + 1:
                self._append_empty_col()
                # o rebuild de colunas recria os itens com novos iids
                children = list(self.tree.get_children())
                row_idx = self._active_row_index(children)
                col_idx = len(self._col_names) + 1
            col_idx = max(2, col_idx)

        if d_row:
            row_idx += d_row
            if row_idx >= len(children):
                self._append_empty_row()
                children = list(self.tree.get_children())
                row_idx = len(children) - 1
            row_idx = max(0, row_idx)

        self._set_active(children[row_idx], col_idx)
        return "break"

    def _tab_move(self, forward: bool):
        if self._editing_entry:
            self._commit_edit()

        children = list(self.tree.get_children())
        if not children:
            return "break"

        row_idx = self._active_row_index(children)
        col_idx = self._active_col_idx or 2
        n_cols = len(self._col_names)

        col_idx += 1 if forward else -1
        if col_idx > n_cols + 1:
            col_idx = 2
            row_idx += 1
            if row_idx >= len(children):
                self._append_empty_row()
                children = list(self.tree.get_children())
        elif col_idx < 2:
            col_idx = n_cols + 1
            row_idx -= 1
            if row_idx < 0:
                row_idx = 0
                col_idx = 2

        self._set_active(children[row_idx], col_idx)
        return "break"

    def _move_home(self, event=None):
        children = list(self.tree.get_children())
        if not children:
            return "break"
        row_idx = self._active_row_index(children)
        self._set_active(children[row_idx], 2)
        return "break"

    def _move_end(self, event=None):
        children = list(self.tree.get_children())
        if not children:
            return "break"
        row_idx = self._active_row_index(children)
        self._set_active(children[row_idx], len(self._col_names) + 1)
        return "break"

    def _move_ctrl_home(self, event=None):
        children = list(self.tree.get_children())
        if not children:
            return "break"
        self._set_active(children[0], 2)
        return "break"

    def _move_ctrl_end(self, event=None):
        children = list(self.tree.get_children())
        if not children:
            return "break"
        self._set_active(children[-1], len(self._col_names) + 1)
        return "break"

    def _move_page(self, forward: bool):
        children = list(self.tree.get_children())
        if not children:
            return "break"
        row_idx = self._active_row_index(children)
        row_idx += 15 if forward else -15
        row_idx = max(0, min(row_idx, len(children) - 1))
        col_idx = self._active_col_idx or 2
        self._set_active(children[row_idx], col_idx)
        return "break"

    def _select_active_column(self, event=None):
        col_idx = self._active_col_idx or 2
        self._sel_mode = "column"
        self._active_row_id = None
        self._active_col_idx = col_idx
        self.tree.selection_remove(*self.tree.selection())
        self._highlight_column(col_idx)
        return "break"

    def _select_active_row(self, event=None):
        children = list(self.tree.get_children())
        if not children:
            return "break"
        row_id = self._active_row_id or (children[self._active_row_index(children)])
        self._sel_mode = "rows"
        self._active_col_idx = None
        self.tree.selection_set(row_id)
        self._hide_highlight()
        return "break"

    def _on_key_type(self, event):
        if self._editing_entry:
            return
        if event.keysym in self._NAV_KEYSYMS:
            return
        if event.state & 0x0004:            # Ctrl pressionado
            return
        if not event.char or not event.char.isprintable():
            return
        if self._sel_mode != "cell" or not self._active_row_id or not self._active_col_idx:
            return

        self._start_edit(self._active_row_id, f"#{self._active_col_idx}")
        if self._editing_entry:
            self._editing_entry.delete(0, "end")
            self._editing_entry.insert(0, event.char)
        return "break"

    def _append_empty_row(self) -> str:
        empty = [""] * (len(self._col_names) + 1)
        new_id = self.tree.insert("", "end", values=empty)
        self._update_row_numbers()
        return new_id

    def _append_empty_col(self):
        new_name = f"C{len(self._col_names)+1}"
        new_names = self._col_names + [new_name]
        all_data = []
        for it in self.tree.get_children():
            row = list(self.tree.item(it, "values"))[1:]
            row.append("")
            all_data.append(row)
        active_row_pos = None
        if self._active_row_id in self.tree.get_children():
            active_row_pos = list(self.tree.get_children()).index(self._active_row_id)
        self._rebuild_with_cols(new_names, all_data)
        if active_row_pos is not None:
            children = list(self.tree.get_children())
            if active_row_pos < len(children):
                self._active_row_id = children[active_row_pos]

    # ── Highlight (borda da célula/coluna ativa) ─────────────────────────────

    def _place_highlight(self, x: int, y: int, w: int, h: int, thickness: int = 2):
        self._hl_top.place(x=x, y=y, width=w, height=thickness)
        self._hl_bottom.place(x=x, y=y + h - thickness, width=w, height=thickness)
        self._hl_left.place(x=x, y=y, width=thickness, height=h)
        self._hl_right.place(x=x + w - thickness, y=y, width=thickness, height=h)

    def _hide_highlight(self):
        for f in self._hl_frames:
            f.place_forget()

    def _highlight_cell(self, row_id: str, col_idx: int):
        bbox = self.tree.bbox(row_id, f"#{col_idx}")
        if not bbox:
            self._hide_highlight()
            return
        x, y, w, h = bbox
        self._place_highlight(x, y, w, h)

    def _highlight_column(self, col_idx: int):
        children = self.tree.get_children()
        bbox = None
        for it in children:
            bbox = self.tree.bbox(it, f"#{col_idx}")
            if bbox:
                break
        if not bbox:
            self._hide_highlight()
            return
        x, y0, w, _ = bbox
        height = max(0, self.tree.winfo_height() - y0)
        self._place_highlight(x, y0, w, height)

    def _reposition_highlight(self):
        if self._sel_mode == "cell" and self._active_row_id:
            self._highlight_cell(self._active_row_id, self._active_col_idx)
        elif self._sel_mode == "column" and self._active_col_idx:
            self._highlight_column(self._active_col_idx)
        elif self._sel_mode == "range" and self._range_rows and self._range_cols:
            self._draw_range_highlight()

    # ── Context menu ──────────────────────────────────────────────────────────

    def _show_context_menu(self, event):
        col = self.tree.identify_column(event.x)
        if col:
            try:
                self._ctx_col_idx = int(col.replace("#", ""))
            except ValueError:
                self._ctx_col_idx = 1

        row_id = self.tree.identify_row(event.y)
        if row_id and row_id not in self.tree.selection():
            self.tree.selection_set(row_id)
        if row_id and self._ctx_col_idx > 1:
            self._set_active(row_id, self._ctx_col_idx)

        try:
            self.ctx_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.ctx_menu.grab_release()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _row_tag(self, position_1_based: int) -> str:
        return "oddrow" if position_1_based % 2 else "evenrow"

    def _update_row_numbers(self):
        for i, it in enumerate(self.tree.get_children(), 1):
            vals = list(self.tree.item(it, "values"))
            if vals:
                vals[0] = str(i)
                self.tree.item(it, values=vals, tags=(self._row_tag(i),))

    def _clear_tree(self):
        for it in self.tree.get_children():
            self.tree.delete(it)
        self._sel_mode = "cell"
        self._active_row_id = None
        self._active_col_idx = None
        self._range_rows = None
        self._range_cols = None
        self._drag_anchor = None
        self._hide_highlight()

    def _notify_change(self):
        if self.on_data_change:
            try:
                self.on_data_change(self.get_dataframe())
            except Exception:
                pass

    @staticmethod
    def _is_number(s: str) -> bool:
        try:
            float(str(s).replace(",", ".").strip())
            return True
        except (ValueError, TypeError):
            return False
