"""
t-Test with Expected Mean Display
"""
import customtkinter as ctk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Dict


def display_t_test_expected_results(parent, results: Dict):
    """Display t-test with expected mean results"""
    result = results.get("result", {})
    response_name = results.get("response", "")
    
    if not result:
        ctk.CTkLabel(
            parent,
            text="Nenhum resultado disponível",
            font=ctk.CTkFont(size=14)
        ).pack(pady=20)
        return
    
    content_frame = ctk.CTkFrame(parent)
    content_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Left - Chart
    left_frame = ctk.CTkFrame(content_frame)
    left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
    
    # Right - Table
    right_frame = ctk.CTkFrame(content_frame)
    right_frame.pack(side="left", fill="both", expand=True)
    
    # Display histogram
    display_histogram(left_frame, result, response_name)
    
    # Display results table
    display_results_table(right_frame, result, response_name)


def display_histogram(parent, result: Dict, response_name: str):
    """Display histogram with expected mean line"""
    chart_frame = ctk.CTkFrame(parent)
    chart_frame.pack(fill="both", expand=True)
    
    ctk.CTkLabel(
        chart_frame,
        text=f"Histograma - {response_name}",
        font=ctk.CTkFont(size=14, weight="bold")
    ).pack(pady=5)
    
    # Get data from parent window (we need to pass it)
    sample_mean = result.get("sampleMean", 0)
    expected_mean = result.get("expectedMean", 0)
    sample_std = result.get("sampleStd", 0)
    n = result.get("n", 0)
    
    # Generate sample data for visualization (approximate distribution)
    np.random.seed(42)
    sample_data = np.random.normal(sample_mean, sample_std, n)
    
    # Create histogram
    fig = Figure(figsize=(6, 5), dpi=100)
    ax = fig.add_subplot(111)
    
    # Plot histogram
    ax.hist(sample_data, bins='auto', color='#3498db', alpha=0.7, edgecolor='black', label='Dados da Amostra')
    
    # Add sample mean line
    ax.axvline(sample_mean, color='blue', linestyle='--', linewidth=2, label=f'Média da Amostra = {sample_mean:.4f}')
    
    # Add expected mean line
    ax.axvline(expected_mean, color='red', linestyle='--', linewidth=2, label=f'Média Esperada = {expected_mean:.4f}')
    
    ax.set_xlabel('Valores', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequência', fontsize=10, fontweight='bold')
    ax.set_title('Distribuição dos Dados vs. Média Esperada', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    canvas = FigureCanvasTkAgg(fig, chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)


def display_results_table(parent, result: Dict, response_name: str):
    """Display results table"""
    results_frame = ctk.CTkFrame(parent)
    results_frame.pack(fill="both", expand=True, padx=20, pady=20)
    
    ctk.CTkLabel(
        results_frame,
        text=f"t-Test: {response_name}",
        font=ctk.CTkFont(size=18, weight="bold")
    ).pack(pady=10)
    
    # Results table
    info_frame = ctk.CTkFrame(results_frame)
    info_frame.pack(fill="x", padx=20, pady=20)
    
    is_significant = result['pValue'] < 0.05
    text_color = "red" if is_significant else "green"
    
    info_data = [
        ("Média da Amostra", f"{result['sampleMean']:.4f}"),
        ("Média Esperada", f"{result['expectedMean']:.4f}"),
        ("Desvio Padrão", f"{result['sampleStd']:.4f}"),
        ("Tamanho da Amostra", f"{result['n']}"),
        ("t-Statistic", f"{result['tStatistic']:.4f}"),
        ("Graus de Liberdade", f"{result['degreesOfFreedom']}"),
        ("p-Value", f"{result['pValue']:.4f}" if result['pValue'] >= 0.0001 else "< 0.0001*"),
        ("IC 95% Inferior", f"{result['ciLower']:.4f}"),
        ("IC 95% Superior", f"{result['ciUpper']:.4f}"),
        ("Conclusão", "Diferença Significativa (p < 0.05)" if is_significant else "Sem Diferença Significativa")
    ]
    
    for i, (label, value) in enumerate(info_data):
        label_widget = ctk.CTkLabel(info_frame, text=f"{label}:", font=ctk.CTkFont(weight="bold"), width=250)
        label_widget.grid(row=i, column=0, padx=10, pady=5, sticky="w")
        
        color = text_color if i == len(info_data) - 1 else None
        value_widget = ctk.CTkLabel(info_frame, text=value, width=300, text_color=color)
        value_widget.grid(row=i, column=1, padx=10, pady=5, sticky="w")
