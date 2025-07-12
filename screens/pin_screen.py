from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.app import App
from utils.pin_utils import guardar_pin, verificar_pin
import os


class PinScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Layout principal
        self.layout = BoxLayout(orientation='vertical', padding=40, spacing=10)

        # Widgets: etiqueta, entrada de PIN, botón, mensaje y botón de salida
        self.label = Label(text="Ingresa tu PIN")
        self.input = TextInput(password=True, multiline=False, input_filter='int')
        self.btn = Button(text="Acceder")
        self.msg = Label(text="", color=(1, 0, 0, 1))
        self.btn_salir = Button(text="Salir", size_hint=(1, 0.2))

        # Eventos
        self.btn.bind(on_release=self.verificar_o_guardar)
        self.btn_salir.bind(on_release=self.salir_app)

        # Añadir widgets
        self.layout.add_widget(self.label)
        self.layout.add_widget(self.input)
        self.layout.add_widget(self.btn)
        self.layout.add_widget(self.msg)
        self.layout.add_widget(self.btn_salir)
        self.add_widget(self.layout)

        # Si no existe el archivo de PIN, crea uno por defecto (1234)
        if not os.path.exists("data/pin.txt"):
            os.makedirs("data", exist_ok=True)
            guardar_pin("1234")

    def verificar_o_guardar(self, *args):
        pin = self.input.text.strip()
        if len(pin) < 4:
            self.msg.text = "El PIN debe tener al menos 4 dígitos"
            return

        # Verifica el PIN
        if verificar_pin(pin):
            self.manager.current = "menu"
        else:
            self.msg.text = "PIN incorrecto"

    def salir_app(self, instance):
        # Cierra la aplicación
        App.get_running_app().stop()
