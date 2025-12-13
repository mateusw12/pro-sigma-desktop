"""
UI Utilities - Componentes reutilizáveis de interface
"""
import customtkinter as ctk
from tkinter import ttk
from typing import List, Dict, Any


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
