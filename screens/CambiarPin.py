from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from utils.pin_utils import guardar_pin, verificar_pin

class CambiarPinScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.layout = BoxLayout(orientation='vertical', padding=40, spacing=10)

        self.label_actual = Label(text="Ingresa PIN actual:")
        self.input_actual = TextInput(password=True, multiline=False, input_filter='int')

        self.label_nuevo = Label(text="Ingresa nuevo PIN:")
        self.input_nuevo = TextInput(password=True, multiline=False, input_filter='int')

        self.label_confirmar = Label(text="Confirma nuevo PIN:")
        self.input_confirmar = TextInput(password=True, multiline=False, input_filter='int')

        self.msg = Label(text="", color=(1, 0, 0, 1))

        self.btn_cambiar = Button(text="Cambiar PIN")
        self.btn_cambiar.bind(on_release=self.cambiar_pin)

        self.btn_volver = Button(text="Volver al menú")
        self.btn_volver.bind(on_release=self.volver_al_menu)

        self.layout.add_widget(self.label_actual)
        self.layout.add_widget(self.input_actual)
        self.layout.add_widget(self.label_nuevo)
        self.layout.add_widget(self.input_nuevo)
        self.layout.add_widget(self.label_confirmar)
        self.layout.add_widget(self.input_confirmar)
        self.layout.add_widget(self.msg)
        self.layout.add_widget(self.btn_cambiar)
        self.layout.add_widget(self.btn_volver)

        self.add_widget(self.layout)

    def cambiar_pin(self, instance):
        pin_actual = self.input_actual.text.strip()
        pin_nuevo = self.input_nuevo.text.strip()
        pin_confirmar = self.input_confirmar.text.strip()

        if not verificar_pin(pin_actual):
            self.msg.text = "PIN actual incorrecto"
            return

        if len(pin_nuevo) < 4:
            self.msg.text = "El nuevo PIN debe tener al menos 4 dígitos"
            return

        if pin_nuevo != pin_confirmar:
            self.msg.text = "Los PINs no coinciden"
            return

        guardar_pin(pin_nuevo)
        self.msg.text = "PIN cambiado correctamente"
        self.input_actual.text = ""
        self.input_nuevo.text = ""
        self.input_confirmar.text = ""

    def volver_al_menu(self, instance):
        self.manager.current = "menu"
