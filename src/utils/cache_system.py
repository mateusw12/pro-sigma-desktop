"""
Sistema de Cache para Widgets e Dados
Melhora performance evitando recriação desnecessária
"""
from typing import Any, Dict, Optional, Callable
import time
from functools import wraps


class WidgetCache:
    """
    Cache para widgets criados
    Evita recriação de widgets pesados
    """
    
    def __init__(self, max_size: int = 100, ttl: int = 3600):
        """
        Args:
            max_size: Tamanho máximo do cache
            ttl: Time to live em segundos (padrão 1 hora)
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[Any]:
        """
        Obtém widget do cache
        
        Args:
            key: Chave do widget
            
        Returns:
            Widget ou None se não encontrado/expirado
        """
        if key in self._cache:
            entry = self._cache[key]
            
            # Verifica expiração
            if time.time() - entry['timestamp'] > self.ttl:
                del self._cache[key]
                return None
            
            return entry['widget']
        
        return None
    
    def set(self, key: str, widget: Any):
        """
        Armazena widget no cache
        
        Args:
            key: Chave do widget
            widget: Widget a armazenar
        """
        # Limpa cache se estiver cheio
        if len(self._cache) >= self.max_size:
            # Remove item mais antigo
            oldest_key = min(self._cache.keys(), 
                           key=lambda k: self._cache[k]['timestamp'])
            del self._cache[oldest_key]
        
        self._cache[key] = {
            'widget': widget,
            'timestamp': time.time()
        }
    
    def clear(self, key: Optional[str] = None):
        """
        Limpa cache
        
        Args:
            key: Chave específica ou None para limpar tudo
        """
        if key:
            if key in self._cache:
                del self._cache[key]
        else:
            self._cache.clear()
    
    def size(self) -> int:
        """Retorna tamanho atual do cache"""
        return len(self._cache)


class DataCache:
    """
    Cache para dados processados
    Evita reprocessamento de dados
    """
    
    def __init__(self, max_size_mb: int = 500, ttl: int = 3600):
        """
        Args:
            max_size_mb: Tamanho máximo em MB
            ttl: Time to live em segundos
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.max_size_mb = max_size_mb
        self.ttl = ttl
        self._current_size = 0
    
    def get(self, key: str) -> Optional[Any]:
        """
        Obtém dados do cache
        
        Args:
            key: Chave dos dados
            
        Returns:
            Dados ou None se não encontrado/expirado
        """
        if key in self._cache:
            entry = self._cache[key]
            
            # Verifica expiração
            if time.time() - entry['timestamp'] > self.ttl:
                self._remove_entry(key)
                return None
            
            # Atualiza timestamp (LRU)
            entry['timestamp'] = time.time()
            return entry['data']
        
        return None
    
    def set(self, key: str, data: Any, size_mb: float = 0):
        """
        Armazena dados no cache
        
        Args:
            key: Chave dos dados
            data: Dados a armazenar
            size_mb: Tamanho estimado em MB
        """
        # Remove entrada antiga se existir
        if key in self._cache:
            self._remove_entry(key)
        
        # Verifica se precisa limpar espaço
        while self._current_size + size_mb > self.max_size_mb and self._cache:
            # Remove item mais antigo
            oldest_key = min(self._cache.keys(), 
                           key=lambda k: self._cache[k]['timestamp'])
            self._remove_entry(oldest_key)
        
        self._cache[key] = {
            'data': data,
            'size': size_mb,
            'timestamp': time.time()
        }
        self._current_size += size_mb
    
    def _remove_entry(self, key: str):
        """Remove entrada do cache"""
        if key in self._cache:
            entry = self._cache[key]
            self._current_size -= entry['size']
            del self._cache[key]
    
    def clear(self):
        """Limpa todo o cache"""
        self._cache.clear()
        self._current_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do cache"""
        return {
            'entries': len(self._cache),
            'size_mb': self._current_size,
            'max_size_mb': self.max_size_mb,
            'usage_percent': (self._current_size / self.max_size_mb) * 100
        }


def cache_result(ttl: int = 300):
    """
    Decorator para cachear resultado de funções
    
    Args:
        ttl: Time to live em segundos
    """
    cache = {}
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Cria chave única baseada em argumentos
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Verifica cache
            if key in cache:
                entry = cache[key]
                if time.time() - entry['timestamp'] <= ttl:
                    return entry['result']
                else:
                    del cache[key]
            
            # Executa função e cacheia resultado
            result = func(*args, **kwargs)
            cache[key] = {
                'result': result,
                'timestamp': time.time()
            }
            
            return result
        
        return wrapper
    
    return decorator


# Instâncias globais de cache
widget_cache = WidgetCache(max_size=100, ttl=3600)
data_cache = DataCache(max_size_mb=500, ttl=3600)
