"""
Sistema de Lazy Import para bibliotecas pesadas
Melhora significativamente o tempo de inicialização
"""
import sys
from typing import Any, Callable


class LazyModule:
    """
    Wrapper para importação lazy de módulos
    O módulo só é importado quando realmente usado
    """
    
    def __init__(self, module_name: str, import_func: Callable = None):
        """
        Args:
            module_name: Nome do módulo para import
            import_func: Função customizada de import (opcional)
        """
        self.module_name = module_name
        self.import_func = import_func
        self._module = None
    
    def _load_module(self):
        """Carrega o módulo se ainda não foi carregado"""
        if self._module is None:
            if self.import_func:
                self._module = self.import_func()
            else:
                self._module = __import__(self.module_name, fromlist=[''])
        return self._module
    
    def __getattr__(self, name: str) -> Any:
        """Intercepta acessos a atributos e carrega o módulo se necessário"""
        module = self._load_module()
        return getattr(module, name)
    
    def __call__(self, *args, **kwargs):
        """Permite chamar o módulo diretamente se for uma função"""
        module = self._load_module()
        return module(*args, **kwargs)


# ===== LAZY IMPORTS DE BIBLIOTECAS PESADAS =====

def _import_matplotlib():
    """Import customizado para matplotlib"""
    import matplotlib
    # Configura backend antes de importar pyplot
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    return plt

def _import_matplotlib_figure():
    """Import customizado para matplotlib.figure"""
    from matplotlib.figure import Figure
    return Figure

def _import_matplotlib_backend():
    """Import customizado para matplotlib backend"""
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    return FigureCanvasTkAgg


# Instâncias lazy dos módulos pesados
lazy_numpy = LazyModule('numpy')
lazy_pandas = LazyModule('pandas')
lazy_scipy_stats = LazyModule('scipy.stats')
lazy_scipy_spatial_distance = LazyModule('scipy.spatial.distance')
lazy_scipy_optimize = LazyModule('scipy.optimize')
lazy_scipy_special = LazyModule('scipy.special')
lazy_pyDOE2 = LazyModule('pyDOE2')
lazy_matplotlib = LazyModule('matplotlib.pyplot', _import_matplotlib)
lazy_matplotlib_figure = LazyModule('matplotlib.figure', _import_matplotlib_figure)
lazy_matplotlib_backend = LazyModule('matplotlib.backends.backend_tkagg', _import_matplotlib_backend)
lazy_statsmodels_api = LazyModule('statsmodels.api')
lazy_statsmodels_formula = LazyModule('statsmodels.formula.api')


def get_numpy():
    """Retorna numpy, carregando apenas quando necessário"""
    return lazy_numpy._load_module()


def get_pandas():
    """Retorna pandas, carregando apenas quando necessário"""
    return lazy_pandas._load_module()


def get_scipy_stats():
    """Retorna scipy.stats, carregando apenas quando necessário"""
    return lazy_scipy_stats._load_module()


def get_scipy_spatial_distance():
    """Retorna scipy.spatial.distance, carregando apenas quando necessário"""
    return lazy_scipy_spatial_distance._load_module()


def get_scipy_optimize():
    """Retorna scipy.optimize, carregando apenas quando necessário"""
    return lazy_scipy_optimize._load_module()


def get_scipy_special():
    """Retorna scipy.special, carregando apenas quando necessário"""
    return lazy_scipy_special._load_module()


def get_pyDOE2():
    """Retorna pyDOE2, carregando apenas quando necessário"""
    return lazy_pyDOE2._load_module()


def get_matplotlib():
    """Retorna matplotlib.pyplot, carregando apenas quando necessário"""
    return lazy_matplotlib._load_module()


def get_matplotlib_figure():
    """Retorna matplotlib Figure, carregando apenas quando necessário"""
    return lazy_matplotlib_figure._load_module()


def get_matplotlib_backend():
    """Retorna FigureCanvasTkAgg, carregando apenas quando necessário"""
    return lazy_matplotlib_backend._load_module()


def get_statsmodels_api():
    """Retorna statsmodels.api, carregando apenas quando necessário"""
    return lazy_statsmodels_api._load_module()


def get_statsmodels_formula():
    """Retorna statsmodels.formula.api, carregando apenas quando necessário"""
    return lazy_statsmodels_formula._load_module()


# ===== FUNÇÕES AUXILIARES =====

def preload_heavy_modules():
    """
    Pré-carrega módulos pesados em background
    Útil para carregar durante tela de splash ou idle
    """
    import threading
    
    def load_in_background():
        # Carrega módulos mais pesados primeiro
        get_matplotlib()
        get_scipy_stats()
        get_numpy()
        get_pandas()
    
    thread = threading.Thread(target=load_in_background, daemon=True)
    thread.start()


def is_module_loaded(module_lazy: LazyModule) -> bool:
    """
    Verifica se um módulo lazy já foi carregado
    
    Args:
        module_lazy: Instância de LazyModule
        
    Returns:
        True se já foi carregado, False caso contrário
    """
    return module_lazy._module is not None
