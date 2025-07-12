import os
import shutil
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from PIL import Image

from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image as KivyImage
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from functools import partial


class GeneradorEmbeddingsTFLite:
    def __init__(self, modelo_tflite_path=None, carpeta_faces="data/faces", carpeta_embeddings="data/embeddings"):
        self.faces_folder = carpeta_faces
        self.embeddings_folder = carpeta_embeddings
        os.makedirs(self.faces_folder, exist_ok=True)
        os.makedirs(self.embeddings_folder, exist_ok=True)

        if modelo_tflite_path is None:
            modelo_tflite_path = os.path.join("models", "mobilefacenet.tflite")
        if not os.path.isfile(modelo_tflite_path):
            raise FileNotFoundError(f"Modelo TFLite no encontrado: {modelo_tflite_path}")

        self.interpreter = tf.lite.Interpreter(model_path=modelo_tflite_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def generar_embedding_promedio(self, ruta_usuario):
        embeddings = []
        for i in range(1, 7):
            img_path = os.path.join(ruta_usuario, f"{i}.jpg")
            if not os.path.isfile(img_path):
                continue
            try:
                with Image.open(img_path).convert("RGB") as img:
                    embedding = self.obtener_embedding(img)
                    if embedding is not None:
                        embeddings.append(embedding)
            except Exception as e:
                print(f"Error en {img_path}: {e}")
        if embeddings:
            promedio = np.mean(embeddings, axis=0)
            promedio /= np.linalg.norm(promedio)
            return promedio
        return None

    def obtener_embedding(self, imagen_pil):
        img_np = self.preprocesar_imagen(imagen_pil)
        self.interpreter.set_tensor(self.input_details[0]['index'], img_np)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        norm = np.linalg.norm(output)
        if norm == 0 or np.any(np.isnan(output)):
            return None
        return output / norm

    def preprocesar_imagen(self, imagen_pil):
        imagen_pil = imagen_pil.resize((112, 112)).convert("RGB")
        img_np = np.array(imagen_pil).astype(np.float32)
        img_np = (img_np - 127.5) / 128.0
        return np.expand_dims(img_np, axis=0)


class RegistrarRostro(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical', spacing=10, padding=20)

        self.img1 = KivyImage(size_hint=(1, 0.6))
        self.label_mensaje = Label(text="", size_hint=(1, 0.1), font_size=18, color=(1, 0, 0, 1))
        self.input_nombre = TextInput(hint_text="Nombre", size_hint=(1, 0.1))
        self.btn_capturar = Button(text="Capturar Rostro", size_hint=(1, 0.1))
        self.btn_volver = Button(text="Volver", size_hint=(1, 0.1))

        self.btn_capturar.bind(on_release=self.iniciar_captura_multiple)
        self.btn_volver.bind(on_release=self.volver_menu)

        self.layout.add_widget(self.img1)
        self.layout.add_widget(self.label_mensaje)
        self.layout.add_widget(self.input_nombre)
        self.layout.add_widget(self.btn_capturar)
        self.layout.add_widget(self.btn_volver)
        self.add_widget(self.layout)

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.2)

        self.blink_count = 0
        self.blink_event = None
        self.capture = None

    def volver_menu(self, instance):
        self.manager.current = "menu"

    def on_enter(self):
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.actualizar, 1.0 / 30.0)

    def on_leave(self):
        if self.capture:
            self.capture.release()
            self.capture = None
        Clock.unschedule(self.actualizar)
        self.label_mensaje.text = ""

    def actualizar(self, dt):
        ret, frame = self.capture.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(frame_rgb)
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    margen = 30
                    x1 = max(0, x - margen)
                    y1 = max(0, y - margen)
                    x2 = min(iw, x + w + margen)
                    y2 = min(ih, y + h + margen)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, 'Rostro', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img1.texture = texture

    def iniciar_captura_multiple(self, *args):
        nombre = self.input_nombre.text.strip()
        if not nombre:
            self.label_mensaje.text = "Debe ingresar un nombre"
            return

        os.makedirs("data/faces", exist_ok=True)
        self.blink_count = 0
        self.blink_event = Clock.schedule_interval(self.parpadear_mensaje, 0.5)

        for i in range(6):
            Clock.schedule_once(partial(self.capturar_foto, nombre, i + 1), i * 1.0)

        Clock.schedule_once(lambda dt: self.generar_y_guardar_embedding(nombre), 7)

    def parpadear_mensaje(self, dt):
        if self.blink_count >= 7:
            self.label_mensaje.text = ""
            Clock.unschedule(self.blink_event)
            return
        self.label_mensaje.text = "Capturando rostro..." if self.blink_count % 2 == 0 else ""
        self.blink_count += 1

    def capturar_foto(self, nombre, numero, dt):
        if self.capture is None:
            return
        ret, frame = self.capture.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(frame_rgb)
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    margen = 30
                    x1 = max(0, x - margen)
                    y1 = max(0, y - margen)
                    x2 = min(iw, x + w + margen)
                    y2 = min(ih, y + h + margen)
                    face_img = frame[y1:y2, x1:x2]

                    carpeta_usuario = os.path.join("data", "faces", nombre)
                    os.makedirs(carpeta_usuario, exist_ok=True)
                    ruta = os.path.join(carpeta_usuario, f"{numero}.jpg")
                    cv2.imwrite(ruta, face_img)

    def generar_y_guardar_embedding(self, nombre):
        generador = GeneradorEmbeddingsTFLite(os.path.join("models", "mobilefacenet.tflite"))
        ruta_usuario = os.path.join(generador.faces_folder, nombre)
        embedding = generador.generar_embedding_promedio(ruta_usuario)

        if embedding is not None:
            ruta_salida = os.path.join(generador.embeddings_folder, f"{nombre}.npy")
            np.save(ruta_salida, embedding)
            self.label_mensaje.text = f"✅ Embedding generado para {nombre}"
            self.input_nombre.text = ''
            Clock.schedule_once(lambda dt: self.volver_menu(None), 2)
        else:
            self.label_mensaje.text = f"❌ Falló generación. Repite la captura."
            self.borrar_carpeta_usuario(nombre)

    def borrar_carpeta_usuario(self, nombre):
        ruta_usuario = os.path.join("data", "faces", nombre)
        if os.path.exists(ruta_usuario):
            shutil.rmtree(ruta_usuario)
