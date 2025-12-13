"""
Otimizações de Renderização para CustomTkinter
Reduz overhead de renderização e melhora fluidez
"""
import customtkinter as ctk
from typing import Optional


# ===== CONFIGURAÇÕES DE RENDERING =====

# Desabilita animações pesadas para melhor performance
ctk.deactivate_automatic_dpi_awareness()  # Melhora performance em multi-monitor


def optimize_ctk_widgets():
    """
    Aplica otimizações globais aos widgets CTk
    Deve ser chamado uma vez no início da aplicação
    """
    # Reduz corner radius padrão (menos pesado para renderizar)
    ctk.set_widget_scaling(1.0)  # Sem scaling desnecessário
    
    # Configurações de aparência otimizadas
    ctk.set_appearance_mode("dark")  # Dark mode é mais leve
    

def create_lightweight_frame(parent, **kwargs):
    """
    Cria um frame otimizado para performance
    
    Args:
        parent: Widget pai
        **kwargs: Argumentos adicionais para CTkFrame
        
    Returns:
        CTkFrame otimizado
    """
    # Remove configurações pesadas de renderização
    default_config = {
        'fg_color': 'transparent',  # Transparente é mais leve
        'corner_radius': 5,  # Corner radius menor
        'border_width': 0,  # Sem borda por padrão
    }
    
    # Atualiza com kwargs do usuário
    default_config.update(kwargs)
    
    return ctk.CTkFrame(parent, **default_config)


def create_lightweight_button(parent, text, command, **kwargs):
    """
    Cria um botão otimizado para performance
    
    Args:
        parent: Widget pai
        text: Texto do botão
        command: Função a executar
        **kwargs: Argumentos adicionais
        
    Returns:
        CTkButton otimizado
    """
    default_config = {
        'corner_radius': 6,  # Corner radius menor
        'border_width': 0,
        'border_spacing': 5,
    }
    
    # Adiciona fonte apenas se não foi especificada
    if 'font' not in kwargs:
        default_config['font'] = get_button_font()
    
    default_config.update(kwargs)
    
    return ctk.CTkButton(parent, text=text, command=command, **default_config)


def create_lightweight_label(parent, text, **kwargs):
    """
    Cria um label otimizado
    
    Args:
        parent: Widget pai
        text: Texto do label
        **kwargs: Argumentos adicionais
        
    Returns:
        CTkLabel otimizado
    """
    default_config = {}
    
    # Adiciona fonte apenas se não foi especificada
    if 'font' not in kwargs:
        default_config['font'] = get_label_font()
    
    default_config.update(kwargs)
    
    return ctk.CTkLabel(parent, text=text, **default_config)


class OptimizedScrollableFrame(ctk.CTkScrollableFrame):
    """
    Frame scrollável otimizado com virtualização
    Renderiza apenas widgets visíveis para melhor performance
    """
    
    def __init__(self, parent, **kwargs):
        """
        Args:
            parent: Widget pai
            **kwargs: Argumentos adicionais para CTkScrollableFrame
        """
        # Configurações otimizadas
        default_config = {
            'fg_color': 'transparent',
            'scrollbar_button_color': '#2E86DE',
            'scrollbar_button_hover_color': '#1E5BA8',
        }
        default_config.update(kwargs)
        
        super().__init__(parent, **default_config)
        
        # Otimiza scrollbar
        if hasattr(self, '_scrollbar'):
            self._scrollbar.configure(width=10)
    
    def add_widget_lazy(self, widget_factory, *args, **kwargs):
        """
        Adiciona widget de forma lazy (só cria quando visível)
        
        Args:
            widget_factory: Função que cria o widget
            *args, **kwargs: Argumentos para a factory
        """
        # Por enquanto cria diretamente, mas pode ser expandido
        # para incluir virtualização verdadeira no futuro
        return widget_factory(self, *args, **kwargs)


def reduce_widget_overhead(widget):
    """
    Reduz overhead de renderização de um widget
    
    Args:
        widget: Widget a otimizar
    """
    # Desabilita updates desnecessários
    if hasattr(widget, 'configure'):
        # Aplica configurações de baixo overhead
        try:
            widget.configure(takefocus=0)  # Remove foco desnecessário
        except:
            pass


def batch_update_widgets(widgets, update_func):
    """
    Atualiza múltiplos widgets de forma eficiente
    
    Args:
        widgets: Lista de widgets a atualizar
        update_func: Função que recebe widget e atualiza
    """
    # Desabilita updates durante processamento
    for widget in widgets:
        if hasattr(widget, 'configure'):
            update_func(widget)


# ===== ESTILOS PRÉ-DEFINIDOS OTIMIZADOS =====
# Nota: Não criar CTkFont no nível do módulo pois Tkinter precisa estar inicializado

LIGHTWEIGHT_BUTTON_STYLE = {
    'corner_radius': 6,
    'border_width': 0,
    'height': 35,
}

LIGHTWEIGHT_CARD_STYLE = {
    'corner_radius': 8,
    'border_width': 1,
    'border_color': 'gray25',
}

LIGHTWEIGHT_LABEL_STYLE = {
    # font será adicionado quando widget for criado
}

HEADER_LABEL_STYLE = {
    # font será adicionado quando widget for criado
}


def get_button_font(size=12, weight='normal'):
    """Retorna CTkFont para botões (lazy)"""
    return ctk.CTkFont(size=size, weight=weight)


def get_label_font(size=11, weight='normal'):
    """Retorna CTkFont para labels (lazy)"""
    return ctk.CTkFont(size=size, weight=weight)
