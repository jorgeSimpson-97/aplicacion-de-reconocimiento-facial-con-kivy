from kivy.app import App
from kivy.uix.screenmanager import ScreenManager
from screens.pin_screen import PinScreen
from screens.main_menu import MainMenu
from screens.registrar_rostro import RegistrarRostro
from screens.reconocimiento_facial import Reconocimiento
from screens.CambiarPin import CambiarPinScreen
from screens.VerUsuarios import VerUsuariosScreen
import os
from kivy.utils import platform

class GestorPantallas(ScreenManager):
    pass

class AlarmaApp(App):
    def build(self):
        # En Android, solicita permisos necesarios
        if platform == 'android':
            from android.permissions import request_permissions, Permission
            request_permissions([
                Permission.CAMERA,
                Permission.ACCESS_FINE_LOCATION,
                Permission.ACCESS_COARSE_LOCATION
            ])

        sm = GestorPantallas()

        # Añadir pantallas de manera segura
        sm.add_widget(PinScreen(name="pin"))
        sm.add_widget(MainMenu(name="menu"))
        sm.add_widget(RegistrarRostro(name="registrar"))
        sm.add_widget(VerUsuariosScreen(name="ver_usuarios"))
        sm.add_widget(CambiarPinScreen(name="cambiar_pin"))

        # Verificar existencia del modelo antes de agregar pantalla Reconocimiento
        modelo_path = os.path.join("models", "mobilefacenet.tflite")
        if os.path.exists(modelo_path):
            sm.add_widget(Reconocimiento(name="reconocer"))
        else:
            print(f"⚠️ No se encontró el modelo TFLite en {modelo_path}. No se cargará la pantalla de reconocimiento.")

        sm.current = "pin"
        return sm

if __name__ == "__main__":
    AlarmaApp().run()
