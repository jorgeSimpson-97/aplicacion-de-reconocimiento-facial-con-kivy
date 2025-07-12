from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.app import App
import os
import shutil

class VerUsuariosScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_ui()
    def on_pre_enter(self):
        self.clear_widgets()
        self.build_ui()
    def build_ui(self):
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        faces_folder = "data/faces"
        
        if not os.path.exists(faces_folder):
            layout.add_widget(Label(text="No hay usuarios registrados"))
            self.add_widget(layout)
            return
            
        # Scroll para todos los usuarios
        scroll = ScrollView()
        users_grid = GridLayout(cols=1, spacing=20, size_hint_y=None, padding=10)
        users_grid.bind(minimum_height=users_grid.setter('height'))

        # Añadir cada usuario
        for user in sorted(os.listdir(faces_folder)):
            user_path = os.path.join(faces_folder, user)
            if os.path.isdir(user_path):
                user_box = self.create_user_box(user, user_path)
                users_grid.add_widget(user_box)
        
        scroll.add_widget(users_grid)
        layout.add_widget(scroll)

        # Botón inferior: Volver
        volver_btn = Button(text="Volver al menú", size_hint_y=None, height=50)
        volver_btn.bind(on_press=self.go_back)
        layout.add_widget(volver_btn)

        self.add_widget(layout)

    def create_user_box(self, username, user_path):
        box = BoxLayout(orientation='vertical', spacing=5, size_hint_y=None, height=220)

        # Nombre del usuario centrado
        user_label = Label(text=username, font_size='20sp', size_hint_y=None, height=30)
        box.add_widget(user_label)

        # Fila horizontal de imágenes
        image_row = BoxLayout(orientation='horizontal', spacing=5, size_hint_y=None, height=110)
        for img in sorted(os.listdir(user_path))[:6]:  # Mostrar hasta 6 imágenes
            if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(user_path, img)
                image_widget = Image(source=img_path, size_hint=(None, None), size=(100, 100))
                image_row.add_widget(image_widget)
        box.add_widget(image_row)

        # Botón de borrar usuario
        borrar_btn = Button(text="Borrar usuario", size_hint_y=None, height=40)
        borrar_btn.bind(on_press=lambda instance, user=username: self.borrar_usuario(user))
        box.add_widget(borrar_btn)

        return box

    def go_back(self, instance):
        self.manager.current = "menu"

    def borrar_usuario(self, username):
        user_path = os.path.join("data/faces", username)
        if os.path.exists(user_path):
            shutil.rmtree(user_path)
            print(f"[✓] Usuario '{username}' eliminado")

        # También eliminar su archivo .npy en embeddings
        emb_path = os.path.join("data/embeddings", f"{username}.npy")
        if os.path.exists(emb_path):
            os.remove(emb_path)
            print(f"[✓] Embedding '{username}.npy' eliminado")

        # Recargar embeddings globalmente
        app = App.get_running_app()
        if hasattr(app, 'actualizar_embeddings'):
            app.actualizar_embeddings()

        # Redibujar la pantalla
        self.clear_widgets()
        self.build_ui()
