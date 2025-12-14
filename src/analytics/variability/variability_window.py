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
        
        # Get selected X columns
        x_columns = [col for col, var in self.x_checkboxes.items() if var.get()]
        
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
        x_scroll_frame = ctk.CTkScrollableFrame(x_frame, height=200)
        x_scroll_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        self.x_checkboxes = {}
        for col in self.df.columns:
            var = ctk.BooleanVar(value=False)
            cb = ctk.CTkCheckBox(
                x_scroll_frame,
                text=col,
                variable=var,
                font=ctk.CTkFont(size=11),
                command=self.update_separator_level_options
            )
            cb.pack(anchor="w", pady=2)
            self.x_checkboxes[col] = var
        
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
            # Get selected columns
            x_columns = [col for col, var in self.x_checkboxes.items() if var.get()]
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
                separator_levels=separator_levels
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
