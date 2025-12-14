"""
Variability Analysis Window
Allows user to create variability charts with multiple X factors and one Y response
Similar to JMP Variability Chart
"""
import customtkinter as ctk
from tkinter import messagebox
from src.utils.lazy_imports import get_pandas, get_numpy, get_matplotlib_figure, get_matplotlib_backend, get_matplotlib

from .variability_utils import (
    create_nested_variability_chart
)

# Lazy-loaded libraries
_pd = None
_np = None
_plt = None
_Figure = None
_FigureCanvasTkAgg = None

def _ensure_libs():
    """Ensure heavy libraries are loaded"""
    global _pd, _np, _plt, _Figure, _FigureCanvasTkAgg
    if _pd is None:
        _pd = get_pandas()
        _np = get_numpy()
        _plt = get_matplotlib()
        _Figure = get_matplotlib_figure()
        _FigureCanvasTkAgg = get_matplotlib_backend()
    return _pd, _np, _plt, _Figure, _FigureCanvasTkAgg


class VariabilityWindow(ctk.CTkToplevel):
    def __init__(self, parent, df):
        super().__init__(parent)
        
        # Load heavy libraries (lazy)
        pd, np, plt, Figure, FigureCanvasTkAgg = _ensure_libs()
        self.pd = pd
        self.np = np
        self.plt = plt
        self.Figure = Figure
        self.FigureCanvasTkAgg = FigureCanvasTkAgg
        
        self.df = df
        self.selected_x_columns = []
        self.selected_y_column = None
        
        # Window configuration
        self.title("Análise de Variabilidade")
        
        # Allow resizing
        self.resizable(True, True)
        self.minsize(1000, 700)
        
        # Start maximized (full screen)
        self.state('zoomed')  # Windows maximized
        
        # Make modal
        self.transient(parent)
        self.grab_set()
        
        self.create_widgets()
    
    def get_selected_x_columns_ordered(self):
        """Get selected X columns in their current order"""
        selected = [col for col in self.df.columns if self.x_checkboxes[col].get()]
        return selected
    
    def move_x_column_up(self, col):
        """Move X column up in the order"""
        if not self.x_checkboxes[col].get():
            return  # Only move if selected
        
        selected = self.get_selected_x_columns_ordered()
        if col not in selected or selected.index(col) == 0:
            return  # Already at top or not selected
        
        # Swap order in the actual dataframe columns list
        all_cols = list(self.df.columns)
        current_idx = all_cols.index(col)
        
        # Find the previous selected column
        prev_selected_cols = [c for c in all_cols[:current_idx] if self.x_checkboxes[c].get()]
        if prev_selected_cols:
            prev_col = prev_selected_cols[-1]
            prev_idx = all_cols.index(prev_col)
            
            # Swap in the list
            all_cols[current_idx], all_cols[prev_idx] = all_cols[prev_idx], all_cols[current_idx]
            
            # Update dataframe column order
            self.df = self.df[all_cols]
            
            # Rebuild X checkboxes
            self.rebuild_x_checkboxes()
            self.update_separator_level_options()
    
    def move_x_column_down(self, col):
        """Move X column down in the order"""
        if not self.x_checkboxes[col].get():
            return  # Only move if selected
        
        selected = self.get_selected_x_columns_ordered()
        if col not in selected or selected.index(col) == len(selected) - 1:
            return  # Already at bottom or not selected
        
        # Swap order in the actual dataframe columns list
        all_cols = list(self.df.columns)
        current_idx = all_cols.index(col)
        
        # Find the next selected column
        next_selected_cols = [c for c in all_cols[current_idx+1:] if self.x_checkboxes[c].get()]
        if next_selected_cols:
            next_col = next_selected_cols[0]
            next_idx = all_cols.index(next_col)
            
            # Swap in the list
            all_cols[current_idx], all_cols[next_idx] = all_cols[next_idx], all_cols[current_idx]
            
            # Update dataframe column order
            self.df = self.df[all_cols]
            
            # Rebuild X checkboxes
            self.rebuild_x_checkboxes()
            self.update_separator_level_options()
    
    def rebuild_x_checkboxes(self):
        """Rebuild X checkboxes after reordering"""
        # Store current states
        checkbox_states = {col: var.get() for col, var in self.x_checkboxes.items()}
        
        # Clear existing frames
        for col, widgets in self.x_order_frames.items():
            widgets['frame'].destroy()
        
        self.x_checkboxes.clear()
        self.x_order_frames.clear()
        
        # Recreate checkboxes in new order using stored reference
        for col in self.df.columns:
            col_frame = ctk.CTkFrame(self.x_scroll_frame, fg_color="transparent")
            col_frame.pack(fill="x", pady=2)
            
            var = ctk.BooleanVar(value=checkbox_states.get(col, False))
            cb = ctk.CTkCheckBox(
                col_frame,
                text=col,
                variable=var,
                font=ctk.CTkFont(size=11),
                command=self.update_separator_level_options
            )
            cb.pack(side="left", padx=(0, 5))
            
            # Order control buttons
            order_frame = ctk.CTkFrame(col_frame, fg_color="transparent")
            order_frame.pack(side="left")
            
            up_btn = ctk.CTkButton(
                order_frame,
                text="↑",
                width=25,
                height=20,
                font=ctk.CTkFont(size=12, weight="bold"),
                command=lambda c=col: self.move_x_column_up(c)
            )
            up_btn.pack(side="left", padx=1)
            
            down_btn = ctk.CTkButton(
                order_frame,
                text="↓",
                width=25,
                height=20,
                font=ctk.CTkFont(size=12, weight="bold"),
                command=lambda c=col: self.move_x_column_down(c)
            )
            down_btn.pack(side="left", padx=1)
            
            # Order indicator
            order_label = ctk.CTkLabel(
                col_frame,
                text="",
                font=ctk.CTkFont(size=9),
                text_color="gray"
            )
            order_label.pack(side="left", padx=(5, 0))
            
            self.x_checkboxes[col] = var
            self.x_order_frames[col] = {
                'frame': col_frame,
                'checkbox': cb,
                'order_frame': order_frame,
                'up_btn': up_btn,
                'down_btn': down_btn,
                'order_label': order_label
            }
        
        self.update_x_column_order_display()
    
    def update_x_column_order_display(self):
        """Update the order display indicators"""
        selected = self.get_selected_x_columns_ordered()
        
        for col, widgets in self.x_order_frames.items():
            if col in selected:
                order_num = selected.index(col) + 1
                widgets['order_label'].configure(text=f"(Nível {order_num})")
                widgets['order_frame'].pack(side="left")
            else:
                widgets['order_label'].configure(text="")
                # Hide order buttons when not selected
                # widgets['order_frame'].pack_forget()
    
    def toggle_separator_options(self):
        """Show/hide separator level options based on checkbox state"""
        if self.show_separators_var.get():
            self.separator_levels_frame.pack(fill="x", padx=10, pady=(5, 10))
        else:
            self.separator_levels_frame.pack_forget()
    
    def update_separator_level_options(self):
        """Update separator level options based on selected X columns"""
        # Clear existing checkboxes
        for widget in self.separator_checkboxes_frame.winfo_children():
            widget.destroy()
        self.separator_level_vars.clear()
        
        # Get selected X columns in order
        x_columns = self.get_selected_x_columns_ordered()
        
        if len(x_columns) > 0:
            # Create checkbox for each level
            for i, col in enumerate(x_columns):
                var = ctk.BooleanVar(value=(i == 0))  # First level selected by default
                cb = ctk.CTkCheckBox(
                    self.separator_checkboxes_frame,
                    text=f"Nível {i+1} ({col})",
                    variable=var,
                    font=ctk.CTkFont(size=11)
                )
                cb.pack(side="left", padx=5)
                self.separator_level_vars[i] = var
        else:
            # Show message when no X columns selected
            ctk.CTkLabel(
                self.separator_checkboxes_frame,
                text="Selecione fatores X primeiro",
                font=ctk.CTkFont(size=10),
                text_color="gray"
            ).pack(side="left", padx=5)
        
        # Update order display
        self.update_x_column_order_display()
    
    def create_widgets(self):
        # Main container with scrollable frame
        self.main_container = ctk.CTkScrollableFrame(self)
        self.main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title = ctk.CTkLabel(
            self.main_container,
            text="📊 Análise de Variabilidade",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=(0, 20))
        
        # Description
        desc = ctk.CTkLabel(
            self.main_container,
            text="Crie gráficos de variabilidade com múltiplos fatores X e uma resposta Y\nSimilar ao Variability Chart do JMP",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        desc.pack(pady=(0, 20))
        
        # Configuration Frame
        config_frame = ctk.CTkFrame(self.main_container)
        config_frame.pack(fill="x", pady=(0, 20))
        
        # Column Selection Frame
        selection_frame = ctk.CTkFrame(config_frame)
        selection_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Left side - X Columns (multiple selection)
        x_frame = ctk.CTkFrame(selection_frame)
        x_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        ctk.CTkLabel(
            x_frame,
            text="Fatores X (selecione múltiplos):",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        ctk.CTkLabel(
            x_frame,
            text="Use Ctrl+Click para múltipla seleção",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        ).pack(pady=(0, 5))
        
        # Scrollable frame for X columns
        self.x_scroll_frame = ctk.CTkScrollableFrame(x_frame, height=200)
        self.x_scroll_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        self.x_checkboxes = {}
        self.x_order_frames = {}  # Store frames for reordering
        for col in self.df.columns:
            # Container frame for checkbox + order buttons
            col_frame = ctk.CTkFrame(self.x_scroll_frame, fg_color="transparent")
            col_frame.pack(fill="x", pady=2)
            
            var = ctk.BooleanVar(value=False)
            cb = ctk.CTkCheckBox(
                col_frame,
                text=col,
                variable=var,
                font=ctk.CTkFont(size=11),
                command=self.update_separator_level_options
            )
            cb.pack(side="left", padx=(0, 5))
            
            # Order control buttons (only visible when checked)
            order_frame = ctk.CTkFrame(col_frame, fg_color="transparent")
            order_frame.pack(side="left")
            
            up_btn = ctk.CTkButton(
                order_frame,
                text="↑",
                width=25,
                height=20,
                font=ctk.CTkFont(size=12, weight="bold"),
                command=lambda c=col: self.move_x_column_up(c)
            )
            up_btn.pack(side="left", padx=1)
            
            down_btn = ctk.CTkButton(
                order_frame,
                text="↓",
                width=25,
                height=20,
                font=ctk.CTkFont(size=12, weight="bold"),
                command=lambda c=col: self.move_x_column_down(c)
            )
            down_btn.pack(side="left", padx=1)
            
            # Order indicator
            order_label = ctk.CTkLabel(
                col_frame,
                text="",
                font=ctk.CTkFont(size=9),
                text_color="gray"
            )
            order_label.pack(side="left", padx=(5, 0))
            
            self.x_checkboxes[col] = var
            self.x_order_frames[col] = {
                'frame': col_frame,
                'checkbox': cb,
                'order_frame': order_frame,
                'up_btn': up_btn,
                'down_btn': down_btn,
                'order_label': order_label
            }
        
        self.update_x_column_order_display()
        
        # Right side - Y Column (single selection)
        y_frame = ctk.CTkFrame(selection_frame)
        y_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        ctk.CTkLabel(
            y_frame,
            text="Variável Resposta Y (selecione uma):",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        # Scrollable frame for Y column
        y_scroll_frame = ctk.CTkScrollableFrame(y_frame, height=200)
        y_scroll_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        self.y_var = ctk.StringVar(value="")
        for col in self.df.columns:
            rb = ctk.CTkRadioButton(
                y_scroll_frame,
                text=col,
                variable=self.y_var,
                value=col,
                font=ctk.CTkFont(size=11)
            )
            rb.pack(anchor="w", pady=2)
        
        # Options Frame
        options_frame = ctk.CTkFrame(config_frame)
        options_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            options_frame,
            text="Opções do Gráfico:",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", pady=(10, 5))
        
        options_inner = ctk.CTkFrame(options_frame)
        options_inner.pack(fill="x", padx=10, pady=(0, 10))
        
        self.show_mean_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            options_inner,
            text="Mostrar Linhas de Média",
            variable=self.show_mean_var,
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=10)
        
        self.show_separators_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            options_inner,
            text="Linhas de Separação",
            variable=self.show_separators_var,
            font=ctk.CTkFont(size=12),
            command=self.toggle_separator_options
        ).pack(side="left", padx=10)
        
        # Separator levels frame
        self.separator_levels_frame = ctk.CTkFrame(options_frame)
        self.separator_levels_frame.pack(fill="x", padx=10, pady=(5, 10))
        
        ctk.CTkLabel(
            self.separator_levels_frame,
            text="Níveis de Separação:",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(side="left", padx=(10, 10))
        
        ctk.CTkLabel(
            self.separator_levels_frame,
            text="Escolha em quais níveis hierárquicos mostrar linhas de divisão:",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        ).pack(side="left", padx=(0, 10))
        
        # Container for level checkboxes (will be populated dynamically)
        self.separator_checkboxes_frame = ctk.CTkFrame(self.separator_levels_frame)
        self.separator_checkboxes_frame.pack(side="left", padx=10)
        
        self.separator_level_vars = {}
        
        # Specification Limits Frame
        spec_limits_frame = ctk.CTkFrame(config_frame)
        spec_limits_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            spec_limits_frame,
            text="Limites de Especificação:",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", pady=(10, 5))
        
        ctk.CTkLabel(
            spec_limits_frame,
            text="Deixe em branco se não houver limites definidos",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        ).pack(anchor="w", pady=(0, 5))
        
        limits_inner = ctk.CTkFrame(spec_limits_frame)
        limits_inner.pack(fill="x", padx=10, pady=(0, 10))
        
        # LSL (Lower Specification Limit)
        lsl_frame = ctk.CTkFrame(limits_inner, fg_color="transparent")
        lsl_frame.pack(side="left", padx=10)
        
        ctk.CTkLabel(
            lsl_frame,
            text="LSL (Limite Inferior):",
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=(0, 5))
        
        self.lsl_entry = ctk.CTkEntry(
            lsl_frame,
            width=120,
            placeholder_text="Ex: 3.5",
            font=ctk.CTkFont(size=12)
        )
        self.lsl_entry.pack(side="left")
        
        # USL (Upper Specification Limit)
        usl_frame = ctk.CTkFrame(limits_inner, fg_color="transparent")
        usl_frame.pack(side="left", padx=10)
        
        ctk.CTkLabel(
            usl_frame,
            text="USL (Limite Superior):",
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=(0, 5))
        
        self.usl_entry = ctk.CTkEntry(
            usl_frame,
            width=120,
            placeholder_text="Ex: 4.5",
            font=ctk.CTkFont(size=12)
        )
        self.usl_entry.pack(side="left")
        
        # Generate button
        generate_btn = ctk.CTkButton(
            config_frame,
            text="🔄 Gerar Gráfico de Variabilidade",
            command=self.generate_analysis,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40
        )
        generate_btn.pack(pady=20)
        
        # Results container (initially empty)
        self.results_container = ctk.CTkFrame(self.main_container)
        self.results_container.pack(fill="both", expand=True, pady=(0, 20))
    
    def generate_analysis(self):
        """Generate variability analysis"""
        try:
            # Get selected columns in order
            x_columns = self.get_selected_x_columns_ordered()
            y_column = self.y_var.get()
            
            # Validate selection
            if not x_columns:
                messagebox.showerror(
                    "Erro",
                    "Por favor, selecione pelo menos um fator X."
                )
                return
            
            if not y_column:
                messagebox.showerror(
                    "Erro",
                    "Por favor, selecione uma variável resposta Y."
                )
                return
            
            # Check if Y column is in X columns
            if y_column in x_columns:
                messagebox.showerror(
                    "Erro",
                    "A variável Y não pode ser a mesma que um dos fatores X."
                )
                return
            
            # Clear previous results
            for widget in self.results_container.winfo_children():
                widget.destroy()
            
            # Get options
            show_mean = self.show_mean_var.get()
            show_separators = self.show_separators_var.get()
            
            # Get specification limits
            lsl = None
            usl = None
            
            try:
                lsl_text = self.lsl_entry.get().strip()
                if lsl_text:
                    lsl = float(lsl_text)
            except ValueError:
                messagebox.showwarning(
                    "Aviso",
                    "LSL inválido. Usando apenas o limite superior."
                )
            
            try:
                usl_text = self.usl_entry.get().strip()
                if usl_text:
                    usl = float(usl_text)
            except ValueError:
                messagebox.showwarning(
                    "Aviso",
                    "USL inválido. Usando apenas o limite inferior."
                )
            
            # Validate limits
            if lsl is not None and usl is not None and lsl >= usl:
                messagebox.showerror(
                    "Erro",
                    "O Limite Inferior (LSL) deve ser menor que o Limite Superior (USL)."
                )
                return
            
            # Filter dataframe to only selected columns
            cols_to_use = x_columns + [y_column]
            df_filtered = self.df[cols_to_use].dropna()
            
            if len(df_filtered) == 0:
                messagebox.showerror(
                    "Erro",
                    "Não há dados válidos nas colunas selecionadas."
                )
                return
            
            # Create chart
            chart_frame = ctk.CTkFrame(self.results_container)
            chart_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            ctk.CTkLabel(
                chart_frame,
                text="📊 Gráfico de Variabilidade Hierárquico",
                font=ctk.CTkFont(size=16, weight="bold")
            ).pack(pady=(10, 5))
            
            # Get selected separator levels
            separator_levels = None
            if show_separators and self.separator_level_vars:
                separator_levels = [i for i, var in self.separator_level_vars.items() if var.get()]
                if not separator_levels:
                    separator_levels = None  # If none selected, use default
            
            # Generate hierarchical chart
            fig = create_nested_variability_chart(
                df_filtered,
                x_columns,
                y_column,
                show_mean_lines=show_mean,
                show_group_separators=show_separators,
                separator_levels=separator_levels,
                lsl=lsl,
                usl=usl
            )
            
            # Embed chart in tkinter
            canvas = self.FigureCanvasTkAgg(fig, master=chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
            
            # Export button
            export_frame = ctk.CTkFrame(self.results_container)
            export_frame.pack(fill="x", padx=10, pady=10)
            
            ctk.CTkButton(
                export_frame,
                text="� Exportar Gráfico (PNG)",
                command=lambda: self.export_chart(fig),
                font=ctk.CTkFont(size=12)
            ).pack(padx=5)
            
        except Exception as e:
            messagebox.showerror(
                "Erro",
                f"Erro ao gerar análise:\n{str(e)}"
            )
    
    def export_statistics(self, stats_df):
        """Export statistics to CSV"""
        from tkinter import filedialog
        
        file_path = filedialog.asksaveasfilename(
            title="Salvar Estatísticas",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                stats_df.to_csv(file_path, index=False)
                messagebox.showinfo(
                    "Sucesso",
                    f"Estatísticas exportadas com sucesso para:\n{file_path}"
                )
            except Exception as e:
                messagebox.showerror(
                    "Erro",
                    f"Erro ao exportar estatísticas:\n{str(e)}"
                )
    
    def export_chart(self, fig):
        """Export chart to PNG"""
        from tkinter import filedialog
        
        file_path = filedialog.asksaveasfilename(
            title="Salvar Gráfico",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo(
                    "Sucesso",
                    f"Gráfico exportado com sucesso para:\n{file_path}"
                )
            except Exception as e:
                messagebox.showerror(
                    "Erro",
                    f"Erro ao exportar gráfico:\n{str(e)}"
                )
