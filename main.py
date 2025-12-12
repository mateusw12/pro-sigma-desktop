"""
Aplicação Principal do Pro Sigma
"""
import customtkinter as ctk
from src.core.license_manager import LicenseManager
from src.ui.login_window import LoginWindow
from src.ui.home_page import HomePage


class ProSigmaApp(ctk.CTk):
    """Aplicação principal do Pro Sigma"""
    
    def __init__(self):
        super().__init__()
        
        # Configurações da janela principal
        self.title("Pro Sigma - Análise Estatística Six Sigma")
        self.geometry("1200x700")
        
        # Configuração de tema
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Centraliza a janela
        self.center_window()
        
        # Gerenciador de licenças
        self.license_manager = LicenseManager()
        self.license_data = None
        
        # Verifica licença existente
        self.check_license()
    
    def center_window(self):
        """Centraliza a janela na tela"""
        self.update_idletasks()
        width = 1200
        height = 700
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')
    
    def check_license(self):
        """Verifica se existe licença válida"""
        self.license_data = self.license_manager.load_license()
        
        if self.license_data:
            # Licença válida encontrada, carrega página inicial
            self.load_home_page()
        else:
            # Sem licença, mostra tela de login
            self.show_login()
    
    def show_login(self):
        """Mostra janela de login"""
        login_window = LoginWindow(self, self.on_login_success)
        self.wait_window(login_window)
    
    def on_login_success(self, license_data: dict):
        """
        Callback chamado após login bem-sucedido
        
        Args:
            license_data: Dados da licença validada
        """
        self.license_data = license_data
        self.load_home_page()
    
    def load_home_page(self):
        """Carrega a página inicial"""
        # Limpa janela
        for widget in self.winfo_children():
            widget.destroy()
        
        # Cria página inicial com callback para mudança de licença
        home_page = HomePage(
            self, 
            self.license_data,
            on_license_change=self.on_license_changed
        )
        home_page.pack(fill="both", expand=True)
    
    def on_license_changed(self, new_license_data: dict):
        """
        Callback chamado quando a licença é alterada
        
        Args:
            new_license_data: Dados da nova licença
        """
        self.license_data = new_license_data
        # Recarrega a página para atualizar as ferramentas disponíveis
        self.load_home_page()
    
    def run(self):
        """Inicia a aplicação"""
        self.mainloop()


def main():
    """Função principal"""
    app = ProSigmaApp()
    app.run()


if __name__ == '__main__':
    main()
