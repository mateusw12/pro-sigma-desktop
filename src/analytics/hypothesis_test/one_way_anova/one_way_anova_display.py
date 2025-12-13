"""
One-Way ANOVA Display
"""
import customtkinter as ctk
from src.utils.lazy_imports import get_numpy, get_matplotlib_figure, get_matplotlib_backend
from typing import Dict


def display_anova_results(parent, results: Dict):
    """Display One-Way ANOVA results"""
    anova_data = results.get("oneWayAnova", {})
    
    if not anova_data:
        ctk.CTkLabel(
            parent,
            text="Nenhum resultado disponível",
            font=ctk.CTkFont(size=14)
        ).pack(pady=20)
        return
    
    # Create tabs for each response
    if len(anova_data) > 1:
        tabview = ctk.CTkTabview(parent)
        tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        for response_name, result_data in anova_data.items():
            tab = tabview.add(response_name)
            display_anova_for_response(tab, result_data, response_name)
    else:
        response_name = list(anova_data.keys())[0]
        result_data = anova_data[response_name]
        display_anova_for_response(parent, result_data, response_name)


def display_anova_for_response(parent, result_data: Dict, response_name: str):
    """Display ANOVA results for a single response"""
    content_frame = ctk.CTkFrame(parent)
    content_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Left - Chart
    left_frame = ctk.CTkFrame(content_frame)
    left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
    
    # Right - Tables
    right_frame = ctk.CTkFrame(content_frame)
    right_frame.pack(side="left", fill="both", expand=True)
    
    # Display chart
    display_anova_chart(left_frame, result_data, response_name)
    
    # Display tables
    display_anova_table(right_frame, result_data)
    display_summary_of_fit_table(right_frame, result_data)


def display_anova_chart(parent, result_data: Dict, response_name: str):
    """Display scatter plot with mean line"""
    # Carrega bibliotecas lazy
    np = get_numpy()
    Figure = get_matplotlib_figure()
    FigureCanvasTkAgg = get_matplotlib_backend()
    
    chart_frame = ctk.CTkFrame(parent)
    chart_frame.pack(fill="both", expand=True)
    
    ctk.CTkLabel(
        chart_frame,
        text=f"Distribuição por Grupo - {response_name}",
        font=ctk.CTkFont(size=14, weight="bold")
    ).pack(pady=5)
    
    data_frame = result_data.get("dataFrame", [])
    
    if not data_frame:
        ctk.CTkLabel(chart_frame, text="Sem dados para exibir").pack(pady=20)
        return
    
    # Prepare data
    categories = []
    values = []
    
    for item in data_frame:
        categories.append(str(item['x']))
        values.append(item['y'])
    
    # Get unique categories and calculate mean
    unique_categories = sorted(set(categories))
    global_mean = result_data.get("summaryOfFit", {}).get("media", np.mean(values))
    
    # Create chart
    fig = Figure(figsize=(6, 5), dpi=100)
    ax = fig.add_subplot(111)
    
    # Map categories to numeric values
    category_map = {cat: i for i, cat in enumerate(unique_categories)}
    x_numeric = [category_map[cat] for cat in categories]
    
    # Plot scatter points
    ax.scatter(x_numeric, values, alpha=0.6, s=50, color='#3498db')
    
    # Plot mean line
    ax.axhline(global_mean, color='red', linestyle='--', linewidth=2, 
               label=f'Média = {global_mean:.4f}')
    
    ax.set_xticks(range(len(unique_categories)))
    ax.set_xticklabels(unique_categories, rotation=45, ha='right')
    ax.set_xlabel('Grupos', fontsize=10, fontweight='bold')
    ax.set_ylabel(response_name, fontsize=10, fontweight='bold')
    ax.set_title('Distribuição dos Valores por Grupo', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    canvas = FigureCanvasTkAgg(fig, chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)


def display_anova_table(parent, result_data: Dict):
    """Display ANOVA table"""
    table_frame = ctk.CTkFrame(parent)
    table_frame.pack(fill="x", pady=(0, 20))
    
    ctk.CTkLabel(
        table_frame,
        text="Análise de Variância",
        font=ctk.CTkFont(size=16, weight="bold")
    ).pack(pady=10)
    
    anova = result_data.get("anova", {})
    
    # Create table
    table_data_frame = ctk.CTkFrame(table_frame)
    table_data_frame.pack(fill="x", padx=10, pady=(0, 10))
    
    # Headers
    headers = ["Termo", "GL", "SQ", "QM", "F-Ratio", "Prob > F"]
    for col, header in enumerate(headers):
        ctk.CTkLabel(
            table_data_frame,
            text=header,
            font=ctk.CTkFont(weight="bold"),
            width=90
        ).grid(row=0, column=col, padx=3, pady=5)
    
    # Data rows
    graus = anova.get("grausLiberdade", {})
    s_quadrados = anova.get("sQuadrados", {})
    m_quadrados = anova.get("mQuadrados", {})
    f_ratio = anova.get("fRatio", 0)
    prob_f = anova.get("probF", 0)
    
    rows_data = [
        ("Modelo", graus.get("modelo", 0), s_quadrados.get("modelo", 0), 
         m_quadrados.get("modelo", 0), f_ratio, prob_f),
        ("Erro", graus.get("erro", 0), s_quadrados.get("erro", 0), 
         m_quadrados.get("erro", 0), "", ""),
        ("Total", graus.get("total", 0), s_quadrados.get("total", 0), 
         "", "", "")
    ]
    
    for row_idx, (termo, gl, sq, qm, f_val, p_val) in enumerate(rows_data, start=1):
        # Color significant p-values
        is_significant = isinstance(p_val, (int, float)) and p_val < 0.05
        text_color = "red" if is_significant else None
        
        ctk.CTkLabel(table_data_frame, text=termo, width=90).grid(row=row_idx, column=0, padx=3, pady=3)
        ctk.CTkLabel(table_data_frame, text=f"{gl}" if gl != "" else "", width=90).grid(row=row_idx, column=1, padx=3, pady=3)
        ctk.CTkLabel(table_data_frame, text=f"{sq:.4f}" if sq != "" else "", width=90).grid(row=row_idx, column=2, padx=3, pady=3)
        ctk.CTkLabel(table_data_frame, text=f"{qm:.4f}" if qm != "" else "", width=90).grid(row=row_idx, column=3, padx=3, pady=3)
        ctk.CTkLabel(table_data_frame, text=f"{f_val:.4f}" if f_val != "" else "", width=90).grid(row=row_idx, column=4, padx=3, pady=3)
        
        p_text = ""
        if isinstance(p_val, (int, float)):
            p_text = f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001*"
        ctk.CTkLabel(table_data_frame, text=p_text, width=90, text_color=text_color).grid(row=row_idx, column=5, padx=3, pady=3)


def display_summary_of_fit_table(parent, result_data: Dict):
    """Display Summary of Fit table"""
    table_frame = ctk.CTkFrame(parent)
    table_frame.pack(fill="x", pady=(0, 20))
    
    ctk.CTkLabel(
        table_frame,
        text="Resumo do Ajuste",
        font=ctk.CTkFont(size=16, weight="bold")
    ).pack(pady=10)
    
    summary = result_data.get("summaryOfFit", {})
    
    # Create table
    table_data_frame = ctk.CTkFrame(table_frame)
    table_data_frame.pack(fill="x", padx=10, pady=(0, 10))
    
    # Headers
    headers = ["Termo", "R²", "R² Ajustado", "RMSE", "Média", "Observações"]
    for col, header in enumerate(headers):
        ctk.CTkLabel(
            table_data_frame,
            text=header,
            font=ctk.CTkFont(weight="bold"),
            width=90
        ).grid(row=0, column=col, padx=3, pady=5)
    
    # Data row - use grid directly without intermediate frame
    ctk.CTkLabel(table_data_frame, text="Valor", width=90).grid(row=1, column=0, padx=3, pady=3)
    ctk.CTkLabel(table_data_frame, text=f"{summary.get('rQuadrado', 0):.4f}", width=90).grid(row=1, column=1, padx=3, pady=3)
    ctk.CTkLabel(table_data_frame, text=f"{summary.get('rQuadradoAjustado', 0):.4f}", width=90).grid(row=1, column=2, padx=3, pady=3)
    ctk.CTkLabel(table_data_frame, text=f"{summary.get('rmse', 0):.4f}", width=90).grid(row=1, column=3, padx=3, pady=3)
    ctk.CTkLabel(table_data_frame, text=f"{summary.get('media', 0):.4f}", width=90).grid(row=1, column=4, padx=3, pady=3)
    ctk.CTkLabel(table_data_frame, text=f"{summary.get('observacoes', 0):.0f}", width=90).grid(row=1, column=5, padx=3, pady=3)
