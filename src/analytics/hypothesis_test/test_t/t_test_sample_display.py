"""
t-Test Sample Display
"""
import customtkinter as ctk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Dict


def display_t_test_sample_results(parent, results: Dict):
    """Display t-test sample results"""
    result = results.get("result", {})
    
    if not result:
        ctk.CTkLabel(
            parent,
            text="Nenhum resultado disponível",
            font=ctk.CTkFont(size=14)
        ).pack(pady=20)
        return
    
    test_type = result.get("type", "")
    
    if test_type == "one-sample":
        display_one_sample_results(parent, result)
    elif test_type == "paired-sample":
        display_paired_sample_results(parent, result)


def display_one_sample_results(parent, result: Dict):
    """Display one-sample t-test results"""
    content_frame = ctk.CTkFrame(parent)
    content_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Left - Chart
    left_frame = ctk.CTkFrame(content_frame)
    left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
    
    # Right - Table
    right_frame = ctk.CTkFrame(content_frame)
    right_frame.pack(side="left", fill="both", expand=True)
    
    # Display histogram
    display_one_sample_chart(left_frame, result)
    
    # Display summary table
    display_one_sample_table(right_frame, result)


def display_one_sample_chart(parent, result: Dict):
    """Display histogram for one-sample test"""
    chart_frame = ctk.CTkFrame(parent)
    chart_frame.pack(fill="both", expand=True)
    
    column = result.get("column", "")
    y_values = result.get("yValues", {}).get(column, [])
    
    ctk.CTkLabel(
        chart_frame,
        text=f"Histograma - {column}",
        font=ctk.CTkFont(size=14, weight="bold")
    ).pack(pady=5)
    
    if not y_values:
        ctk.CTkLabel(chart_frame, text="Sem dados para exibir").pack(pady=20)
        return
    
    # Create histogram
    fig = Figure(figsize=(6, 5), dpi=100)
    ax = fig.add_subplot(111)
    
    ax.hist(y_values, bins='auto', color='#3498db', alpha=0.7, edgecolor='black')
    
    # Add mean line
    mean = result.get("mean", 0)
    ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Média = {mean:.4f}')
    
    ax.set_xlabel('Valores', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequência', fontsize=10, fontweight='bold')
    ax.set_title('Distribuição dos Valores', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    canvas = FigureCanvasTkAgg(fig, chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)


def display_one_sample_table(parent, result: Dict):
    """Display summary table for one-sample test"""
    table_frame = ctk.CTkFrame(parent)
    table_frame.pack(fill="both", expand=True, pady=10)
    
    ctk.CTkLabel(
        table_frame,
        text="Resumo do Teste",
        font=ctk.CTkFont(size=16, weight="bold")
    ).pack(pady=10)
    
    # Create table
    table_data_frame = ctk.CTkFrame(table_frame)
    table_data_frame.pack(fill="x", padx=10, pady=(0, 10))
    
    # Headers
    headers = ["Métrica", "Valor"]
    for col, header in enumerate(headers):
        ctk.CTkLabel(
            table_data_frame,
            text=header,
            font=ctk.CTkFont(weight="bold"),
            width=150
        ).grid(row=0, column=col, padx=5, pady=5, sticky="w")
    
    # Data rows
    mean = result.get("mean", 0)
    std = result.get("std", 0)
    t_calc = result.get("tCalculate", 0)
    p_value = result.get("pValue", 0)
    
    is_significant = p_value < 0.05
    
    rows_data = [
        ("Média", f"{mean:.5f}"),
        ("Desvio Padrão", f"{std:.5f}"),
        ("t-Calculado", f"{t_calc:.5f}"),
        ("p-Value", f"{p_value:.5f}"),
    ]
    
    for row_idx, (label, value) in enumerate(rows_data, start=1):
        # Color p-value if significant
        text_color = "red" if row_idx == 4 and is_significant else None
        
        ctk.CTkLabel(table_data_frame, text=label, width=150).grid(row=row_idx, column=0, padx=5, pady=3, sticky="w")
        ctk.CTkLabel(table_data_frame, text=value, width=150, text_color=text_color).grid(row=row_idx, column=1, padx=5, pady=3, sticky="w")
    
    # Add conclusion
    conclusion = "Significativo (p < 0.05)" if is_significant else "Não Significativo"
    conclusion_color = "red" if is_significant else "green"
    
    conclusion_frame = ctk.CTkFrame(table_frame)
    conclusion_frame.pack(fill="x", padx=10, pady=10)
    
    ctk.CTkLabel(
        conclusion_frame,
        text="Conclusão:",
        font=ctk.CTkFont(weight="bold"),
        width=150
    ).pack(side="left", padx=5)
    
    ctk.CTkLabel(
        conclusion_frame,
        text=conclusion,
        text_color=conclusion_color,
        font=ctk.CTkFont(weight="bold")
    ).pack(side="left", padx=5)


def display_paired_sample_results(parent, result: Dict):
    """Display paired-sample t-test results"""
    content_frame = ctk.CTkFrame(parent)
    content_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Left - Chart
    left_frame = ctk.CTkFrame(content_frame)
    left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
    
    # Right - Table
    right_frame = ctk.CTkFrame(content_frame)
    right_frame.pack(side="left", fill="both", expand=True)
    
    # Display histogram
    display_paired_sample_chart(left_frame, result)
    
    # Display summary table
    display_paired_sample_table(right_frame, result)


def display_paired_sample_chart(parent, result: Dict):
    """Display histogram for paired-sample test"""
    chart_frame = ctk.CTkFrame(parent)
    chart_frame.pack(fill="both", expand=True)
    
    ctk.CTkLabel(
        chart_frame,
        text="Histograma das Amostras",
        font=ctk.CTkFont(size=14, weight="bold")
    ).pack(pady=5)
    
    y_values = result.get("yValues", {})
    
    if not y_values:
        ctk.CTkLabel(chart_frame, text="Sem dados para exibir").pack(pady=20)
        return
    
    # Create histogram with multiple series
    fig = Figure(figsize=(6, 5), dpi=100)
    ax = fig.add_subplot(111)
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    for idx, (col_name, values) in enumerate(y_values.items()):
        color = colors[idx % len(colors)]
        ax.hist(values, bins='auto', alpha=0.5, label=col_name, color=color, edgecolor='black')
    
    ax.set_xlabel('Valores', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequência', fontsize=10, fontweight='bold')
    ax.set_title('Distribuição das Amostras', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    canvas = FigureCanvasTkAgg(fig, chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)


def display_paired_sample_table(parent, result: Dict):
    """Display summary table for paired-sample test"""
    table_frame = ctk.CTkFrame(parent)
    table_frame.pack(fill="both", expand=True, pady=10)
    
    ctk.CTkLabel(
        table_frame,
        text="Comparações Pareadas",
        font=ctk.CTkFont(size=16, weight="bold")
    ).pack(pady=10)
    
    pairs = result.get("pairs", {})
    
    if not pairs:
        ctk.CTkLabel(table_frame, text="Sem resultados disponíveis").pack(pady=20)
        return
    
    # Create scrollable frame
    table_scroll = ctk.CTkScrollableFrame(table_frame, height=400)
    table_scroll.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    
    # Headers
    headers = ["Par", "Média Dif.", "Desvio Padrão", "t-Calculado", "p-Value", "n"]
    for col, header in enumerate(headers):
        ctk.CTkLabel(
            table_scroll,
            text=header,
            font=ctk.CTkFont(weight="bold"),
            width=100
        ).grid(row=0, column=col, padx=3, pady=5)
    
    # Data rows
    for row_idx, (pair_name, pair_data) in enumerate(pairs.items(), start=1):
        is_significant = pair_data.get("pValue", 1) < 0.05
        text_color = "red" if is_significant else None
        
        ctk.CTkLabel(table_scroll, text=pair_name, width=100).grid(row=row_idx, column=0, padx=3, pady=3)
        ctk.CTkLabel(table_scroll, text=f"{pair_data['mean']:.5f}", width=100).grid(row=row_idx, column=1, padx=3, pady=3)
        ctk.CTkLabel(table_scroll, text=f"{pair_data['std']:.5f}", width=100).grid(row=row_idx, column=2, padx=3, pady=3)
        ctk.CTkLabel(table_scroll, text=f"{pair_data['tCalculate']:.5f}", width=100).grid(row=row_idx, column=3, padx=3, pady=3)
        
        p_value_text = f"{pair_data['pValue']:.5f}" if pair_data['pValue'] >= 0.0001 else "< 0.0001*"
        ctk.CTkLabel(table_scroll, text=p_value_text, width=100, text_color=text_color).grid(row=row_idx, column=4, padx=3, pady=3)
        
        ctk.CTkLabel(table_scroll, text=f"{pair_data['n']}", width=100).grid(row=row_idx, column=5, padx=3, pady=3)
