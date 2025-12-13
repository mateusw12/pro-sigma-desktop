"""
Utilitários de Performance para UI
"""
import time
from functools import wraps
from typing import Callable


class ResizeOptimizer:
    """
    Otimizador de eventos de redimensionamento
    Evita múltiplas reconstruções durante resize
    """
    
    def __init__(self, debounce_ms: int = 150):
        """
        Args:
            debounce_ms: Tempo de espera em milissegundos antes de executar
        """
        self.debounce_ms = debounce_ms
        self.last_call_time = 0
        self.pending_call = None
        self.widget = None
    
    def debounce(self, func: Callable):
        """
        Decorator para adicionar debounce a funções
        
        Args:
            func: Função a ser "debounced"
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time() * 1000  # Converte para ms
            
            # Cancela chamada pendente anterior
            if self.pending_call:
                try:
                    self.widget.after_cancel(self.pending_call)
                except:
                    pass
            
            # Agenda nova chamada
            def delayed_call():
                self.last_call_time = time.time() * 1000
                func(*args, **kwargs)
            
            # Se tiver widget (tkinter), usa after
            if hasattr(args[0], 'after'):
                self.widget = args[0]
                self.pending_call = self.widget.after(self.debounce_ms, delayed_call)
            else:
                # Fallback: executa imediatamente
                delayed_call()
        
        return wrapper


class LazyLoader:
    """
    Carregador lazy para widgets pesados
    Carrega widgets apenas quando necessário
    """
    
    def __init__(self):
        self.loaded_widgets = {}
    
    def load_widget(self, widget_id: str, create_func: Callable, *args, **kwargs):
        """
        Carrega widget apenas se ainda não foi carregado
        
        Args:
            widget_id: ID único do widget
            create_func: Função para criar o widget
            *args, **kwargs: Argumentos para a função de criação
        
        Returns:
            Widget criado ou em cache
        """
        if widget_id not in self.loaded_widgets:
            self.loaded_widgets[widget_id] = create_func(*args, **kwargs)
        
        return self.loaded_widgets[widget_id]
    
    def clear_cache(self, widget_id: str = None):
        """
        Limpa cache de widgets
        
        Args:
            widget_id: ID específico ou None para limpar tudo
        """
        if widget_id:
            if widget_id in self.loaded_widgets:
                del self.loaded_widgets[widget_id]
        else:
            self.loaded_widgets.clear()


def optimize_frame_resize(frame):
    """
    Aplica otimizações de performance a um frame
    
    Args:
        frame: Frame CTk para otimizar
    """
    # Desabilita propagação de tamanho onde apropriado
    if hasattr(frame, 'pack_propagate'):
        # Não chama por padrão - deixa para casos específicos
        pass
    
    # Configura update_idletasks de forma otimizada
    original_update = frame.update_idletasks
    last_update = [0]  # Lista para modificar no closure
    
    def optimized_update():
        current = time.time()
        # Limita updates a no máximo 1 a cada 50ms
        if current - last_update[0] > 0.05:
            last_update[0] = current
            original_update()
    
    frame.update_idletasks = optimized_update
    
    return frame


def batch_widget_creation(widget_list, parent, batch_size=10):
    """
    Cria widgets em lotes para evitar travamento da UI
    
    Args:
        widget_list: Lista de tuplas (widget_class, kwargs)
        parent: Widget pai
        batch_size: Número de widgets por lote
    
    Returns:
        Lista de widgets criados
    """
    created_widgets = []
    
    def create_batch(start_idx):
        end_idx = min(start_idx + batch_size, len(widget_list))
        
        for i in range(start_idx, end_idx):
            widget_class, kwargs = widget_list[i]
            widget = widget_class(parent, **kwargs)
            created_widgets.append(widget)
        
        if end_idx < len(widget_list):
            # Agenda próximo lote
            parent.after(10, lambda: create_batch(end_idx))
    
    # Inicia criação
    if widget_list:
        create_batch(0)
    
    return created_widgets


# Instância global de otimizador
resize_optimizer = ResizeOptimizer(debounce_ms=150)
lazy_loader = LazyLoader()
