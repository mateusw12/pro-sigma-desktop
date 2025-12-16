"""
Gerenciador de Licenças do Pro Sigma
Responsável por validar e gerenciar as licenças dos usuários
"""
import json
import hashlib
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

class LicenseManager:
    """Gerencia a validação e armazenamento de licenças"""
    
    # Planos disponíveis
    PLANS = {
        'basic': {
            'name': 'Básico',
            'features': [
                'variability', 'process_capability', 'hypothesis_test',
                'distribution_test', 'cov_ems', 'normalization_test',
                'descriptive_stats', 'ishikawa', 'gage_rr', 'run_chart', 'pareto'
            ]
        },
        'intermediate': {
            'name': 'Intermediário',
            'features': [
                'variability', 'process_capability', 'hypothesis_test',
                'distribution_test', 'cov_ems',
                'text_analysis',  'control_charts', 
                'dashboard', 'monte_carlo', 'cov_ems', 'descriptive_stats',
                'ishikawa', 'gage_rr', 'run_chart', 'pareto', 'doe',
            ]
        },
        'pro': {
            'name': 'Pro',
            'features': [
                'variability', 'process_capability', 'hypothesis_test',
                'distribution_test', 'cov_ems',
                'text_analysis', 'normalization_test', 'control_charts', 
                'dashboard', 'monte_carlo',
                'simple_regression', 'multiple_regression', 'multivariate',
                'stackup', 'doe', 'space_filling', 'nonlinear', 'ccd',
                'neural_networks', 'tree_models', 'gage_rr', 'descriptive_stats',
                'ishikawa', 'run_chart', 'pareto', 'k_means'
            ]
        }
    }
    
    def __init__(self):
        """Inicializa o gerenciador de licenças"""
        self.app_data_dir = Path.home() / '.pro_sigma'
        self.license_file = self.app_data_dir / 'license.dat'
        self.secret_key = "ProSigma2025SecretKey"  # Em produção, usar algo mais seguro
        
        # Cria diretório se não existir
        self.app_data_dir.mkdir(exist_ok=True)
    
    def generate_license(self, plan: str, expiration_date: str) -> str:
        """
        Gera uma chave de licença
        
        Args:
            plan: Tipo de plano (basic, intermediate, pro)
            expiration_date: Data de expiração no formato YYYY-MM-DD
            
        Returns:
            Chave de licença codificada
        """
        if plan not in self.PLANS:
            raise ValueError(f"Plano inválido. Use: {', '.join(self.PLANS.keys())}")
        
        # Valida formato da data
        try:
            datetime.strptime(expiration_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Data deve estar no formato YYYY-MM-DD")
        
        # Cria payload
        license_data = {
            'plan': plan,
            'expiratedDate': expiration_date
        }
        
        # Converte para JSON
        json_data = json.dumps(license_data)
        
        # Cria hash de verificação
        hash_check = hashlib.sha256(
            (json_data + self.secret_key).encode()
        ).hexdigest()[:16]
        
        # Combina dados + hash
        full_data = json_data + '|' + hash_check
        
        # Codifica em base64
        license_key = base64.b64encode(full_data.encode()).decode()
        
        return license_key
    
    def validate_license(self, license_key: str) -> Dict:
        """
        Valida uma chave de licença
        
        Args:
            license_key: Chave de licença a ser validada
            
        Returns:
            Dict com informações da licença se válida
            
        Raises:
            ValueError: Se a licença for inválida
        """
        try:
            # Decodifica base64
            decoded = base64.b64decode(license_key.encode()).decode()
            
            # Separa dados e hash
            parts = decoded.split('|')
            if len(parts) != 2:
                raise ValueError("Formato de licença inválido")
            
            json_data, hash_check = parts
            
            # Verifica hash
            expected_hash = hashlib.sha256(
                (json_data + self.secret_key).encode()
            ).hexdigest()[:16]
            
            if hash_check != expected_hash:
                raise ValueError("Licença adulterada ou inválida")
            
            # Parse JSON
            license_data = json.loads(json_data)
            
            # Valida campos obrigatórios
            if 'plan' not in license_data or 'expiratedDate' not in license_data:
                raise ValueError("Licença incompleta")
            
            # Valida plano
            if license_data['plan'] not in self.PLANS:
                raise ValueError("Plano não reconhecido")
            
            # Valida data de expiração
            expiration_date = datetime.strptime(
                license_data['expiratedDate'], 
                '%Y-%m-%d'
            )
            
            if datetime.now() > expiration_date:
                raise ValueError("Licença expirada")
            
            # Adiciona informações do plano
            license_data['plan_name'] = self.PLANS[license_data['plan']]['name']
            license_data['features'] = self.PLANS[license_data['plan']]['features']
            license_data['is_valid'] = True
            
            return license_data
            
        except Exception as e:
            raise ValueError(f"Erro ao validar licença: {str(e)}")
    
    def save_license(self, license_key: str) -> Dict:
        """
        Salva a licença localmente após validação
        
        Args:
            license_key: Chave de licença
            
        Returns:
            Dados da licença validada
        """
        # Valida antes de salvar
        license_data = self.validate_license(license_key)
        
        # Salva a chave
        with open(self.license_file, 'w') as f:
            f.write(license_key)
        
        return license_data
    
    def load_license(self) -> Optional[Dict]:
        """
        Carrega e valida a licença salva
        
        Returns:
            Dict com dados da licença ou None se não houver licença válida
        """
        if not self.license_file.exists():
            return None
        
        try:
            with open(self.license_file, 'r') as f:
                license_key = f.read().strip()
            
            return self.validate_license(license_key)
        except:
            return None
    
    def has_valid_license(self) -> bool:
        """
        Verifica se existe uma licença válida salva
        
        Returns:
            True se existe licença válida, False caso contrário
        """
        return self.load_license() is not None
    
    def has_feature_access(self, feature: str) -> bool:
        """
        Verifica se o usuário tem acesso a uma funcionalidade
        
        Args:
            feature: Nome da funcionalidade
            
        Returns:
            True se tem acesso, False caso contrário
        """
        license_data = self.load_license()
        if not license_data:
            return False
        
        return feature in license_data.get('features', [])
    
    def remove_license(self):
        """Remove a licença salva"""
        if self.license_file.exists():
            self.license_file.unlink()


# Função auxiliar para gerar licenças de teste
def generate_test_licenses():
    """Gera licenças de teste para desenvolvimento"""
    lm = LicenseManager()
    
    licenses = {
        'basic': lm.generate_license('basic', '2026-12-31'),
        'intermediate': lm.generate_license('intermediate', '2026-12-31'),
        'pro': lm.generate_license('pro', '2026-12-31')
    }
    
    print("=== Licenças de Teste ===\n")
    for plan, key in licenses.items():
        print(f"{plan.upper()}:")
        print(key)
        print()
    
    return licenses


if __name__ == '__main__':
    # Gera licenças de teste
    generate_test_licenses()
