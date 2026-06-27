"""
Aplicação Principal do Pro Sigma
"""
import customtkinter as ctk
from src.core.license_manager import LicenseManager
from src.ui.login_window import LoginWindow
from src.ui.home_page import HomePage
from src.utils.lazy_imports import preload_heavy_modules
from src.utils.render_optimization import optimize_ctk_widgets


class ProSigmaApp(ctk.CTk):
    """Aplicação principal do Pro Sigma"""

    def __init__(self):
        super().__init__()

        # Aplica otimizações de renderização ANTES de criar widgets
        optimize_ctk_widgets()

        # Configurações da janela principal
        self.title("Pro Sigma - Análise Estatística Six Sigma")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.resizable(True, True)
        self.minsize(1000, 600)

        # Inicia sempre em tela cheia
        self.state('zoomed')

        # Pré-carrega módulos pesados em background enquanto o usuário vê a tela de login
        preload_heavy_modules()

        self.license_manager = LicenseManager()
        self.license_data = None

        self.check_license()

    def check_license(self):
        """Verifica se existe licença válida"""
        self.license_data = self.license_manager.load_license()

        if self.license_data:
            self.load_home_page()
        else:
            self.withdraw()
            self.show_login()

    def show_login(self):
        """Mostra janela de login"""
        login_window = LoginWindow(self, self.on_login_success)
        self.wait_window(login_window)

    def on_login_success(self, license_data: dict):
        """Callback chamado após login bem-sucedido"""
        self.license_data = license_data
        self.deiconify()
        self.state('zoomed')
        self.load_home_page()

    def load_home_page(self):
        """Carrega a página inicial"""
        for widget in self.winfo_children():
            widget.destroy()

        home_page = HomePage(
            self,
            self.license_data,
            on_license_change=self.on_license_changed
        )
        home_page.pack(fill="both", expand=True)

    def on_license_changed(self, new_license_data: dict):
        """Callback chamado quando a licença é alterada"""
        self.license_data = new_license_data
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
