"""
Ishikawa Diagram Utilities
Funções para criar e exportar Diagramas de Ishikawa (Espinha de Peixe)
"""

import io
from typing import Dict, List, Optional


def create_ishikawa_diagram(
    effect: str,
    categories: Dict[str, List[str]],
    title: Optional[str] = None,
    figsize: tuple = (14, 8)
):
    """
    Cria um Diagrama de Ishikawa (Espinha de Peixe)
    
    Args:
        effect: Texto do efeito principal (cabeça do peixe)
        categories: Dicionário com categorias e suas causas
                   Ex: {'Método': ['Causa1', 'Causa2'], 'Material': ['Causa3']}
        title: Título opcional do diagrama
        figsize: Tamanho da figura (largura, altura)
    
    Returns:
        matplotlib.figure.Figure: Figura do diagrama
    """
    # Lazy import do matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Título
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Linha principal (espinha central)
    spine_y = 50
    spine_start_x = 10
    spine_end_x = 75
    
    ax.plot([spine_start_x, spine_end_x], [spine_y, spine_y], 
            'k-', linewidth=3, zorder=1)
    
    # Cabeça do peixe (efeito)
    head_x = spine_end_x + 5
    head_y = spine_y
    head_width = 18
    head_height = 16
    
    # Desenha retângulo arredondado para o efeito
    effect_box = FancyBboxPatch(
        (head_x, head_y - head_height/2), 
        head_width, head_height,
        boxstyle="round,pad=0.1",
        linewidth=2.5,
        edgecolor='darkblue',
        facecolor='lightblue',
        zorder=3
    )
    ax.add_patch(effect_box)
    
    # Texto do efeito
    ax.text(head_x + head_width/2, head_y, effect,
            ha='center', va='center', fontsize=11, fontweight='bold',
            wrap=True, zorder=4)
    
    # Conexão espinha -> cabeça
    arrow = FancyArrowPatch(
        (spine_end_x, spine_y), (head_x, head_y),
        arrowstyle='->', mutation_scale=25, linewidth=2.5,
        color='black', zorder=2
    )
    ax.add_patch(arrow)
    
    # Organiza categorias
    cat_list = list(categories.keys())
    num_categories = len(cat_list)
    
    if num_categories == 0:
        return fig
    
    # Posições das categorias (alternar superior/inferior)
    positions_top = []
    positions_bottom = []
    
    # Distribui categorias igualmente ao longo da espinha
    spacing = (spine_end_x - spine_start_x - 10) / max(num_categories, 1)
    
    for i, category in enumerate(cat_list):
        x_pos = spine_start_x + 10 + (i * spacing)
        
        if i % 2 == 0:  # Categorias pares vão para cima
            positions_top.append((category, x_pos))
        else:  # Categorias ímpares vão para baixo
            positions_bottom.append((category, x_pos))
    
    # Desenha categorias superiores
    for category, x_pos in positions_top:
        causes = categories[category]
        _draw_category_branch(ax, x_pos, spine_y, category, causes, position='top')
    
    # Desenha categorias inferiores
    for category, x_pos in positions_bottom:
        causes = categories[category]
        _draw_category_branch(ax, x_pos, spine_y, category, causes, position='bottom')
    
    plt.tight_layout()
    return fig


def _draw_category_branch(ax, x_pos, spine_y, category, causes, position='top'):
    """
    Desenha um ramo de categoria (superior ou inferior)
    
    Args:
        ax: Eixo do matplotlib
        x_pos: Posição X na espinha central
        spine_y: Posição Y da espinha central
        category: Nome da categoria
        causes: Lista de causas
        position: 'top' ou 'bottom'
    """
    from matplotlib.patches import FancyBboxPatch
    
    # Configurações
    branch_length = 20
    category_offset = 8
    cause_spacing = 4
    
    if position == 'top':
        angle_sign = 1
        branch_end_y = spine_y + branch_length
        text_va = 'bottom'
    else:
        angle_sign = -1
        branch_end_y = spine_y - branch_length
        text_va = 'top'
    
    # Linha do ramo principal (diagonal)
    ax.plot([x_pos, x_pos], [spine_y, branch_end_y],
            'k-', linewidth=2, zorder=1)
    
    # Caixa da categoria
    cat_box_width = 12
    cat_box_height = 5
    cat_box_x = x_pos - cat_box_width/2
    
    if position == 'top':
        cat_box_y = branch_end_y
    else:
        cat_box_y = branch_end_y - cat_box_height
    
    category_box = FancyBboxPatch(
        (cat_box_x, cat_box_y), 
        cat_box_width, cat_box_height,
        boxstyle="round,pad=0.05",
        linewidth=1.5,
        edgecolor='darkgreen',
        facecolor='lightgreen',
        zorder=2
    )
    ax.add_patch(category_box)
    
    # Texto da categoria
    ax.text(x_pos, cat_box_y + cat_box_height/2, category,
            ha='center', va='center', fontsize=9, fontweight='bold',
            zorder=3)
    
    # Desenha causas
    if causes:
        cause_start_y = spine_y + (branch_length * 0.2 * angle_sign)
        
        for i, cause in enumerate(causes[:5]):  # Máximo 5 causas por categoria
            cause_y = cause_start_y + (i * cause_spacing * angle_sign)
            cause_x_end = x_pos + 8
            
            # Linha da causa
            ax.plot([x_pos, cause_x_end], [cause_y, cause_y],
                    'gray', linewidth=1, linestyle='--', zorder=0)
            
            # Texto da causa
            ax.text(cause_x_end + 0.5, cause_y, cause,
                    ha='left', va='center', fontsize=7.5, 
                    style='italic', color='dimgray', zorder=1)


def export_diagram_to_png(fig, file_path: str, dpi: int = 300):
    """
    Exporta o diagrama para arquivo PNG
    
    Args:
        fig: Figura do matplotlib
        file_path: Caminho do arquivo de saída
        dpi: Resolução da imagem
    """
    fig.savefig(file_path, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')


def export_diagram_to_buffer(fig, dpi: int = 150):
    """
    Exporta o diagrama para um buffer em memória
    
    Args:
        fig: Figura do matplotlib
        dpi: Resolução da imagem
    
    Returns:
        io.BytesIO: Buffer com a imagem PNG
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    return buf


def validate_ishikawa_data(effect: str, categories: Dict[str, List[str]]) -> tuple[bool, str]:
    """
    Valida os dados para criação do diagrama
    
    Args:
        effect: Efeito principal
        categories: Dicionário de categorias e causas
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not effect or not effect.strip():
        return False, "O efeito principal não pode estar vazio"
    
    if not categories:
        return False, "Adicione pelo menos uma categoria"
    
    if len(categories) > 8:
        return False, "Máximo de 8 categorias permitidas"
    
    for category, causes in categories.items():
        if not category or not category.strip():
            return False, "Nome de categoria não pode estar vazio"
        
        if len(causes) > 5:
            return False, f"A categoria '{category}' tem mais de 5 causas (máximo permitido)"
    
    return True, ""
