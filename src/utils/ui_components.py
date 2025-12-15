"""
UI Utilities - Componentes reutilizáveis de interface
"""
import customtkinter as ctk
from tkinter import ttk, filedialog, messagebox
from typing import List, Dict, Any
from datetime import datetime


def create_horizontal_stats_table(parent_frame: ctk.CTkFrame, 
                                  columns: List[str], 
                                  rows_data: List[Dict[str, str]], 
                                  title: str = "Model Adjustement Information"):
    """
    Cria uma tabela horizontal de estatísticas no estilo Minitab/profissional
    
    Args:
        parent_frame: Frame pai onde a tabela será inserida
        columns: Lista de nomes das colunas
        rows_data: Lista de dicts com os dados de cada linha
        title: Título da tabela
    """
    # Clear previous content
    for widget in parent_frame.winfo_children():
        widget.destroy()
    
    # Title
    ctk.CTkLabel(
        parent_frame,
        text=title,
        font=ctk.CTkFont(size=14, weight="bold")
    ).pack(pady=8)
    
    # Table container
    table_container = ctk.CTkFrame(parent_frame)
    table_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    
    # Configure ttk style
    style = ttk.Style()
    style.theme_use("default")
    style.configure("Custom.Treeview",
                   background="#2b2b2b",
                   foreground="white",
                   fieldbackground="#2b2b2b",
                   borderwidth=0,
                   rowheight=30)
    style.configure("Custom.Treeview.Heading",
                   background="#1f538d",
                   foreground="white",
                   borderwidth=1,
                   relief="solid",
                   font=('TkDefaultFont', 9, 'bold'))
    style.map("Custom.Treeview.Heading",
             background=[('active', '#2E86DE')])
    style.map("Custom.Treeview",
             background=[('selected', '#2E86DE')])
    
    # Create Treeview
    tree = ttk.Treeview(
        table_container, 
        columns=columns, 
        show='headings',
        style="Custom.Treeview",
        height=len(rows_data)
    )
    
    # Configure columns
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=120, anchor="center")
    
    # Insert rows
    for row_data in rows_data:
        values = [row_data.get(col, "") for col in columns]
        tree.insert("", "end", values=values)
    
    # Add scrollbar if needed
    scrollbar = ttk.Scrollbar(table_container, orient="horizontal", command=tree.xview)
    tree.configure(xscrollcommand=scrollbar.set)
    
    tree.pack(fill="both", expand=True, padx=5, pady=5)
    scrollbar.pack(fill="x", padx=5)
    
    return tree


def create_vertical_stats_table(parent_frame: ctk.CTkFrame,
                                stats_dict: Dict[str, str],
                                title: str = "Estatísticas"):
    """
    Cria uma tabela vertical simples de estatísticas
    
    Args:
        parent_frame: Frame pai
        stats_dict: Dicionário com chave: valor
        title: Título
    """
    # Clear previous
    for widget in parent_frame.winfo_children():
        widget.destroy()
    
    # Title
    ctk.CTkLabel(
        parent_frame,
        text=title,
        font=ctk.CTkFont(size=14, weight="bold")
    ).pack(pady=8)
    
    # Grid container
    grid_frame = ctk.CTkFrame(parent_frame, fg_color="transparent")
    grid_frame.pack(padx=10, pady=(0, 10), fill="both", expand=True)
    
    row = 0
    for key, value in stats_dict.items():
        ctk.CTkLabel(
            grid_frame,
            text=f"{key}:",
            font=ctk.CTkFont(weight="bold"),
            anchor="w",
            width=150
        ).grid(row=row, column=0, sticky="w", padx=5, pady=3)
        
        ctk.CTkLabel(
            grid_frame,
            text=value,
            anchor="w",
            width=150
        ).grid(row=row, column=1, sticky="w", padx=5, pady=3)
        
        row += 1


def create_minitab_style_table(parent_frame: ctk.CTkFrame,
                               headers: List[str],
                               data_rows: List[List[Any]],
                               title: str = "Results",
                               column_widths: List[int] = None):
    """
    Cria tabela estilo Minitab com múltiplas linhas
    
    Args:
        parent_frame: Frame pai
        headers: Lista de cabeçalhos
        data_rows: Lista de listas com dados
        title: Título da tabela
        column_widths: Lista de larguras para cada coluna
    """
    # Clear previous
    for widget in parent_frame.winfo_children():
        widget.destroy()
    
    # Title
    ctk.CTkLabel(
        parent_frame,
        text=title,
        font=ctk.CTkFont(size=14, weight="bold")
    ).pack(pady=8)
    
    # Table container
    table_container = ctk.CTkFrame(parent_frame)
    table_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    
    # Configure style
    style = ttk.Style()
    style.theme_use("default")
    style.configure("Minitab.Treeview",
                   background="#2b2b2b",
                   foreground="white",
                   fieldbackground="#2b2b2b",
                   borderwidth=0,
                   rowheight=28)
    style.configure("Minitab.Treeview.Heading",
                   background="#1f538d",
                   foreground="white",
                   borderwidth=1,
                   relief="solid",
                   font=('TkDefaultFont', 9, 'bold'))
    style.map("Minitab.Treeview.Heading",
             background=[('active', '#2E86DE')])
    style.map("Minitab.Treeview",
             background=[('selected', '#2E86DE')])
    
    # Create Treeview
    tree = ttk.Treeview(
        table_container,
        columns=headers,
        show='headings',
        style="Minitab.Treeview",
        height=min(len(data_rows), 15)
    )
    
    # Configure columns
    if column_widths is None:
        column_widths = [120] * len(headers)
    
    for idx, header in enumerate(headers):
        tree.heading(header, text=header)
        tree.column(header, width=column_widths[idx], anchor="center")
    
    # Insert data
    for row in data_rows:
        tree.insert("", "end", values=row)
    
    # Scrollbars
    vsb = ttk.Scrollbar(table_container, orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(table_container, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    
    # Pack
    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")
    
    table_container.grid_rowconfigure(0, weight=1)
    table_container.grid_columnconfigure(0, weight=1)
    
    return tree


def create_variable_selector(parent_frame: ctk.CTkFrame, 
                             title: str,
                             variables: List[str],
                             selection_mode: str = "single",
                             description: str = None) -> Dict:
    """
    Cria um seletor padronizado de variáveis
    
    Args:
        parent_frame: Frame pai
        title: Título do seletor
        variables: Lista de variáveis disponíveis
        selection_mode: "single" para dropdown, "multiple" para checkboxes
        description: Texto descritivo opcional
    
    Returns:
        Dict com componentes criados e variáveis de controle
    """
    import tkinter as tk
    
    # Container frame
    selector_frame = ctk.CTkFrame(parent_frame)
    selector_frame.pack(fill="x", padx=20, pady=10)
    
    # Title
    ctk.CTkLabel(
        selector_frame,
        text=title,
        font=ctk.CTkFont(size=14, weight="bold")
    ).pack(anchor="w", pady=(10, 5), padx=10)
    
    # Description (optional)
    if description:
        ctk.CTkLabel(
            selector_frame,
            text=description,
            font=ctk.CTkFont(size=10),
            text_color="gray"
        ).pack(anchor="w", pady=(0, 5), padx=10)
    
    result = {
        "frame": selector_frame,
        "variables": {},
        "mode": selection_mode
    }
    
    if selection_mode == "single":
        # Dropdown for single selection
        var = tk.StringVar()
        dropdown = ctk.CTkOptionMenu(
            selector_frame,
            variable=var,
            values=variables,
            width=300,
            font=ctk.CTkFont(size=12),
            dropdown_font=ctk.CTkFont(size=11)
        )
        dropdown.pack(padx=10, pady=(0, 10))
        
        if variables:
            var.set(variables[0])
        
        result["variable"] = var
        result["widget"] = dropdown
        
    elif selection_mode == "multiple":
        # Scrollable frame with checkboxes
        scroll_frame = ctk.CTkScrollableFrame(selector_frame, height=150)
        scroll_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        for col in variables:
            var = tk.BooleanVar()
            cb = ctk.CTkCheckBox(
                scroll_frame,
                text=col,
                variable=var,
                font=ctk.CTkFont(size=12)
            )
            cb.pack(anchor="w", padx=10, pady=2)
            result["variables"][col] = var
        
        result["scroll_frame"] = scroll_frame
    
    return result


def create_action_button(parent_frame: ctk.CTkFrame,
                        text: str,
                        command: callable,
                        icon: str = "📊") -> ctk.CTkButton:
    """
    Cria um botão de ação padronizado
    
    Args:
        parent_frame: Frame pai
        text: Texto do botão
        command: Função a ser executada
        icon: Emoji/ícone do botão
    
    Returns:
        CTkButton criado
    """
    button = ctk.CTkButton(
        parent_frame,
        text=f"{icon} {text}",
        command=command,
        font=ctk.CTkFont(size=14, weight="bold"),
        height=40,
        corner_radius=8,
        fg_color="#1f538d",
        hover_color="#2E86DE"
    )
    button.pack(pady=20)
    
    return button


def create_section_title(parent_frame: ctk.CTkFrame,
                        title: str,
                        icon: str = "📊") -> ctk.CTkLabel:
    """
    Cria um título de seção padronizado
    
    Args:
        parent_frame: Frame pai
        title: Texto do título
        icon: Emoji/ícone
    
    Returns:
        CTkLabel criado
    """
    label = ctk.CTkLabel(
        parent_frame,
        text=f"{icon} {title}",
        font=ctk.CTkFont(size=16, weight="bold")
    )
    label.pack(pady=(10, 5))
    
    return label


def add_chart_export_button(parent_frame: ctk.CTkFrame, figure, default_filename: str = "grafico"):
    """
    Adiciona botão de exportar gráfico em PNG
    
    Args:
        parent_frame: Frame onde o botão será adicionado
        figure: Objeto Figure do matplotlib
        default_filename: Nome padrão do arquivo (sem extensão)
    
    Returns:
        Button widget
    """
    def export_chart():
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"{default_filename}_{timestamp}.png"
            
            filepath = filedialog.asksaveasfilename(
                defaultextension=".png",
                initialfile=default_name,
                filetypes=[
                    ("PNG Image", "*.png"),
                    ("PDF Document", "*.pdf"),
                    ("SVG Vector", "*.svg"),
                    ("All Files", "*.*")
                ],
                title="Exportar Gráfico"
            )
            
            if filepath:
                figure.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
                messagebox.showinfo("Sucesso", f"Gráfico exportado com sucesso!\n{filepath}")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao exportar gráfico:\n{str(e)}")
    
    export_btn = ctk.CTkButton(
        parent_frame,
        text="💾 Exportar Gráfico",
        command=export_chart,
        height=28,
        fg_color="#27AE60",
        hover_color="#1E8449"
    )
    
    return export_btn
