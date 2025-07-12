from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.app import App
from kivy.properties import StringProperty
from kivy.graphics import Rectangle, Color
from kivy.metrics import dp
from kivy.clock import Clock
# Importa la función para enviar comandos al ESP32-CAM
from utils.send_command import send_command


class MainMenu(Screen):
    background_image = StringProperty('assets/background_menu.jpeg')  # Ruta por defecto

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Layout principal con canvas para el fondo
        self.main_layout = BoxLayout(orientation='vertical')

        # Canvas para la imagen de fondo
        with self.main_layout.canvas.before:
            Color(0.15, 0.15, 0.15, 1)  # Gris oscuro
            self.bg = Rectangle(
                source=self.background_image, 
                pos=self.main_layout.pos, 
                size=self.main_layout.size
            )
        self.main_layout.bind(pos=self.update_bg, size=self.update_bg)

        # Contenedor principal para el contenido
        self.content_layout = BoxLayout(
            orientation='vertical', 
            spacing=dp(20), 
            padding=dp(40),
            size_hint=(0.9, 0.9), 
            pos_hint={'center_x': 0.5, 'center_y': 0.5}
        )

        with self.content_layout.canvas.before:
            Color(0, 0, 0, 0.3)  # Fondo negro con opacidad
            self.content_bg = Rectangle(
                pos=self.content_layout.pos,
                size=self.content_layout.size
            )
        self.content_layout.bind(pos=self.update_content_bg, size=self.update_content_bg)

        # Layout horizontal para los dos botones principales
        top_buttons = BoxLayout(
            orientation='horizontal',
            spacing=dp(20),
            size_hint=(1, None),
            height=dp(50)
        )

        btn_registrar = Button(
            text="Registrar Rostro",
            background_normal='',
            background_color=(0.1, 0.6, 0.3, 1),
            bold=True
        )
        btn_reconocer = Button(
            text="Reconocimiento Facial",
            background_normal='',
            background_color=(0.1, 0.3, 0.6, 1),
            bold=True
        )

        btn_registrar.bind(on_release=self.ir_a_registrar)
        btn_reconocer.bind(on_release=self.ir_a_reconocer)

        top_buttons.add_widget(btn_registrar)
        top_buttons.add_widget(btn_reconocer)

        # Botón FACE_OK
        self.btn_face_ok = Button(
            text="ACTIVAR VEHÍCULO (FACE_OK)",
            size_hint=(1, None),
            height=dp(50),
            background_normal='',
            background_color=(0.8, 0.2, 0.2, 1),
            disabled=True,
            bold=True,
            font_size='16sp'
        )
        self.btn_face_ok.bind(on_release=self.enviar_face_ok)

        # Botones adicionales
        btn_cambiar_pin = Button(
            text="Cambiar PIN",
            size_hint=(1, None),
            height=dp(40),
            background_normal='',
            background_color=(0.3, 0.3, 0.5, 1)
        )
        btn_ver_usuarios = Button(
            text="Ver Usuarios",
            size_hint=(1, None),
            height=dp(40),
            background_normal='',
            background_color=(0.5, 0.3, 0.5, 1)
        )
        btn_salir = Button(
            text="Salir",
            size_hint=(1, None),
            height=dp(40),
            background_normal='',
            background_color=(0.5, 0.5, 0.5, 1)
        )

        btn_cambiar_pin.bind(on_release=self.ir_a_cambiar_pin)
        btn_ver_usuarios.bind(on_release=self.ir_a_ver_usuarios)
        btn_salir.bind(on_release=self.salir_app)

        # Añadir elementos al layout de contenido (sin BLE)
        self.content_layout.add_widget(top_buttons)
        self.content_layout.add_widget(self.btn_face_ok)
        self.content_layout.add_widget(btn_cambiar_pin)
        self.content_layout.add_widget(btn_ver_usuarios)
        self.content_layout.add_widget(btn_salir)

        # Agregar todo al layout principal
        self.main_layout.add_widget(self.content_layout)
        self.add_widget(self.main_layout)

    # Actualiza fondo de pantalla y contenido
    def update_bg(self, instance, value):
        self.bg.pos = instance.pos
        self.bg.size = instance.size

    def update_content_bg(self, instance, value):
        self.content_bg.pos = instance.pos
        self.content_bg.size = instance.size

    # Métodos de navegación
    def ir_a_registrar(self, instance):
        self.manager.current = "registrar"

    def ir_a_reconocer(self, instance):
        self.manager.current = "reconocer"

    def ir_a_ver_usuarios(self, instance):
        self.manager.current = "ver_usuarios"

    def ir_a_cambiar_pin(self, instance):
        self.manager.current = "cambiar_pin"

    def salir_app(self, instance):
        App.get_running_app().stop()

    # FACE_OK: activar y desactivar
    def activar_face_ok(self):
        self.btn_face_ok.disabled = False
        self.btn_face_ok.background_color = (0.2, 0.8, 0.2, 1)
        print("[INFO] Botón FACE_OK activado")

    def desactivar_face_ok(self):
        self.btn_face_ok.disabled = True
        self.btn_face_ok.background_color = (0.8, 0.2, 0.2, 1)
        print("[INFO] Botón FACE_OK desactivado")

    def enviar_face_ok(self, instance):
        print("[INFO] Intentando enviar FACE_OK al vehículo")
        original_color = instance.background_color
        original_disabled = instance.disabled
        instance.disabled = True
        instance.background_color = (0.3, 0.8, 0.3, 1)

        def restore_button(dt=None):
            instance.background_color = original_color
            instance.disabled = original_disabled

        try:
            if send_command("FACE_OK"):
                print("[SUCCESS] Comando FACE_OK enviado correctamente")
            else:
                print("[ERROR] No se pudo enviar el comando")
        except Exception as e:
            print(f"[ERROR] Fallo al enviar comando: {e}")
        finally:
            Clock.schedule_once(restore_button, 2)
