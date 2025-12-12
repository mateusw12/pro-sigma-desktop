"""
Gerenciador de Histórico de Arquivos do Pro Sigma
Salva e gerencia o histórico de arquivos importados
"""
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional


class FileHistory:
    """Gerencia o histórico de arquivos importados"""
    
    def __init__(self):
        """Inicializa o gerenciador de histórico"""
        self.app_data_dir = Path.home() / '.pro_sigma'
        self.history_file = self.app_data_dir / 'file_history.json'
        self.max_history = 50  # Máximo de arquivos no histórico
        
        # Cria diretório se não existir
        self.app_data_dir.mkdir(exist_ok=True)
        
        # Inicializa histórico
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict]:
        """
        Carrega o histórico de arquivos
        
        Returns:
            Lista de dicionários com informações dos arquivos
        """
        if not self.history_file.exists():
            return []
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    
    def _save_history(self):
        """Salva o histórico em disco"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Erro ao salvar histórico: {e}")
    
    def add_file(self, file_path: str, file_type: str, rows: int, cols: int):
        """
        Adiciona um arquivo ao histórico
        
        Args:
            file_path: Caminho completo do arquivo
            file_type: Tipo do arquivo (excel ou csv)
            rows: Número de linhas
            cols: Número de colunas
        """
        path = Path(file_path)
        
        # Verifica se o arquivo existe
        if not path.exists():
            return
        
        # Cria entrada do histórico
        entry = {
            'file_path': str(path.absolute()),
            'file_name': path.name,
            'file_type': file_type,
            'rows': rows,
            'cols': cols,
            'size_bytes': path.stat().st_size,
            'last_accessed': datetime.now().isoformat(),
            'access_count': 1
        }
        
        # Remove entrada antiga se o arquivo já existir
        self.history = [h for h in self.history if h['file_path'] != entry['file_path']]
        
        # Adiciona no início da lista
        self.history.insert(0, entry)
        
        # Limita o tamanho do histórico
        if len(self.history) > self.max_history:
            self.history = self.history[:self.max_history]
        
        # Salva
        self._save_history()
    
    def update_access(self, file_path: str):
        """
        Atualiza a data de acesso de um arquivo
        
        Args:
            file_path: Caminho do arquivo
        """
        for entry in self.history:
            if entry['file_path'] == file_path:
                entry['last_accessed'] = datetime.now().isoformat()
                entry['access_count'] = entry.get('access_count', 1) + 1
                break
        
        self._save_history()
    
    def remove_file(self, file_path: str):
        """
        Remove um arquivo do histórico
        
        Args:
            file_path: Caminho do arquivo
        """
        self.history = [h for h in self.history if h['file_path'] != file_path]
        self._save_history()
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Obtém o histórico de arquivos
        
        Args:
            limit: Número máximo de itens a retornar
            
        Returns:
            Lista de dicionários com informações dos arquivos
        """
        if limit:
            return self.history[:limit]
        return self.history
    
    def clear_history(self):
        """Limpa todo o histórico"""
        self.history = []
        self._save_history()
    
    def get_recent_files(self, count: int = 5) -> List[Dict]:
        """
        Obtém os arquivos mais recentes
        
        Args:
            count: Número de arquivos a retornar
            
        Returns:
            Lista com os arquivos mais recentes
        """
        return self.history[:count]
    
    def get_most_used_files(self, count: int = 5) -> List[Dict]:
        """
        Obtém os arquivos mais utilizados
        
        Args:
            count: Número de arquivos a retornar
            
        Returns:
            Lista com os arquivos mais utilizados
        """
        sorted_history = sorted(
            self.history, 
            key=lambda x: x.get('access_count', 1), 
            reverse=True
        )
        return sorted_history[:count]
    
    def file_exists(self, file_path: str) -> bool:
        """
        Verifica se um arquivo do histórico ainda existe
        
        Args:
            file_path: Caminho do arquivo
            
        Returns:
            True se o arquivo existe, False caso contrário
        """
        return Path(file_path).exists()
    
    def clean_missing_files(self):
        """Remove do histórico arquivos que não existem mais"""
        self.history = [h for h in self.history if self.file_exists(h['file_path'])]
        self._save_history()
    
    def format_size(self, size_bytes: int) -> str:
        """
        Formata o tamanho do arquivo
        
        Args:
            size_bytes: Tamanho em bytes
            
        Returns:
            String formatada (ex: "1.5 MB")
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def format_date(self, iso_date: str) -> str:
        """
        Formata data ISO para exibição
        
        Args:
            iso_date: Data em formato ISO
            
        Returns:
            String formatada (ex: "12/12/2025 14:30")
        """
        try:
            dt = datetime.fromisoformat(iso_date)
            return dt.strftime("%d/%m/%Y %H:%M")
        except:
            return "Data desconhecida"
