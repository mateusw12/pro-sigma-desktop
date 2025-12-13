"""
Configurações de Performance para Pro Sigma
"""

# Configurações de otimização de interface
PERFORMANCE_CONFIG = {
    # Desabilita animações complexas para melhor performance
    'disable_animations': True,
    
    # Tempo de debounce para eventos de redimensionamento (ms)
    'resize_debounce': 100,
    
    # Número máximo de widgets visíveis simultaneamente no scroll
    'max_visible_widgets': 20,
    
    # Usa double buffering para evitar flickering
    'use_double_buffer': True,
    
    # Lazy loading de componentes pesados
    'lazy_load_charts': True,
    
    # Lazy import de bibliotecas pesadas (melhora tempo de inicialização)
    'lazy_imports': True,
    
    # Virtualização de scroll (renderiza apenas visíveis)
    'virtualize_scroll': True,
    
    # Cache de widgets criados
    'cache_widgets': True,
    
    # Otimizar redimensionamento
    'optimize_resize': True,
}

# Configurações de thread para operações pesadas
THREAD_CONFIG = {
    # Número máximo de threads para processamento
    'max_threads': 4,
    
    # Timeout para operações (segundos)
    'operation_timeout': 300,
}

# Configurações de cache
CACHE_CONFIG = {
    # Cache de dados importados
    'cache_imported_data': True,
    
    # Tamanho máximo do cache (MB)
    'max_cache_size': 500,
    
    # Tempo de expiração do cache (minutos)
    'cache_expiry': 60,
}
