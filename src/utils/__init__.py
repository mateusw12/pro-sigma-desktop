"""
Utils Module - Utilitários do Pro Sigma
"""
from .file_history import FileHistory
from .performance_utils import ResizeOptimizer, LazyLoader, optimize_frame_resize, resize_optimizer, lazy_loader
from .performance_config import PERFORMANCE_CONFIG, THREAD_CONFIG, CACHE_CONFIG
from .lazy_imports import (
    lazy_numpy, lazy_pandas, lazy_scipy_stats, lazy_matplotlib,
    get_numpy, get_pandas, get_scipy_stats, get_matplotlib,
    get_matplotlib_figure, get_matplotlib_backend,
    preload_heavy_modules
)
from .render_optimization import (
    optimize_ctk_widgets,
    create_lightweight_frame,
    create_lightweight_button,
    create_lightweight_label,
    OptimizedScrollableFrame,
    LIGHTWEIGHT_BUTTON_STYLE,
    LIGHTWEIGHT_CARD_STYLE,
)
from .cache_system import (
    WidgetCache,
    DataCache,
    widget_cache,
    data_cache,
    cache_result,
)

__all__ = [
    'FileHistory',
    'ResizeOptimizer',
    'LazyLoader',
    'optimize_frame_resize',
    'resize_optimizer',
    'lazy_loader',
    'PERFORMANCE_CONFIG',
    'THREAD_CONFIG',
    'CACHE_CONFIG',
    'lazy_numpy',
    'lazy_pandas',
    'lazy_scipy_stats',
    'lazy_matplotlib',
    'get_numpy',
    'get_pandas',
    'get_scipy_stats',
    'get_matplotlib',
    'get_matplotlib_figure',
    'get_matplotlib_backend',
    'preload_heavy_modules',
    'optimize_ctk_widgets',
    'create_lightweight_frame',
    'create_lightweight_button',
    'create_lightweight_label',
    'OptimizedScrollableFrame',
    'LIGHTWEIGHT_BUTTON_STYLE',
    'LIGHTWEIGHT_CARD_STYLE',
    'WidgetCache',
    'DataCache',
    'widget_cache',
    'data_cache',
    'cache_result',
]
