import os
import cv2
import numpy as np
import tensorflow as tf
import asyncio
import threading
from bleak import BleakClient
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import mediapipe as mp
import time

BLE_DEVICE_ADDRESS = "F0:9E:20:D7:F5"
BLE_CHARACTERISTIC_UUID = "abcd1234-5678-90ab-cdef-1234567890ab"

loop = asyncio.new_event_loop()
def run_loop():
    asyncio.set_event_loop(loop)
    loop.run_forever()
threading.Thread(target=run_loop, daemon=True).start()

class TFLiteFaceRecognizer:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def preprocess(self, face_img):
        face_img = cv2.resize(face_img, (112, 112))
        face_img = face_img.astype(np.float32)
        face_img = (face_img - 127.5) / 128.0
        face_img = np.expand_dims(face_img, axis=0)
        return face_img

    def get_embedding(self, face_img):
        input_data = self.preprocess(face_img)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        embedding = self.interpreter.get_tensor(self.output_details[0]['index'])
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.flatten()

class Reconocimiento(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.conocidos_encodings = []
        self.conocidos_nombres = []
        self.cargar_embeddings_conocidos()

        self.layout = BoxLayout(orientation='vertical', spacing=10, padding=20)
        self.img1 = Image(size_hint=(1, 0.6))
        self.label_bienvenida = Label(text="", size_hint=(1, 0.1), font_size=22, color=(0, 1, 0, 1))
        self.label_distancia = Label(text="", size_hint=(1, 0.1), font_size=18)
        self.btn_volver = Button(text="Volver", size_hint=(1, 0.1))
        self.btn_volver.bind(on_release=self.volver_al_menu)

        self.layout.add_widget(self.img1)
        self.layout.add_widget(self.label_bienvenida)
        self.layout.add_widget(self.label_distancia)
        self.layout.add_widget(self.btn_volver)
        self.add_widget(self.layout)

        self.ultimo_nombre_mostrado = ""
        self.capture = None
        self.ble_activado = False
        self.rostro_reconocido_actualmente = False
        self.ultima_activacion_ble = 0

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)

        modelo_path = os.path.join("models", "mobilefacenet.tflite")
        self.tflite_recognizer = TFLiteFaceRecognizer(modelo_path)

    def cargar_embeddings_conocidos(self):
        embeddings_dir = os.path.join("data", "embeddings")
        if not os.path.exists(embeddings_dir):
            print("[!] No existe carpeta de embeddings")
            return
        archivos = [f for f in os.listdir(embeddings_dir) if f.endswith(".npy")]
        if not archivos:
            print("[!] No se encontraron embeddings")
            return

        for archivo in archivos:
            ruta_embedding = os.path.join(embeddings_dir, archivo)
            try:
                embedding = np.load(ruta_embedding)
                embedding = embedding / np.linalg.norm(embedding)
                nombre = os.path.splitext(archivo)[0]
                self.conocidos_encodings.append(embedding)
                self.conocidos_nombres.append(nombre)
                print(f"[i] Cargado embedding: {nombre}")
            except Exception as e:
                print(f"Error cargando embedding {archivo}: {e}")

    def volver_al_menu(self, instance):
        self.manager.current = "menu"

    def on_enter(self):
        if self.capture is None:
            self.capture = cv2.VideoCapture(0)
            Clock.schedule_interval(self.actualizar, 1.0 / 30.0)

    def on_leave(self):
        if self.capture and self.capture.isOpened():
            self.capture.release()
            self.capture = None
        Clock.unschedule(self.actualizar)
        self.label_bienvenida.text = ""
        self.label_distancia.text = ""
        self.ble_activado = False
        self.rostro_reconocido_actualmente = False

    def actualizar(self, dt):
        if self.capture is None or not self.capture.isOpened():
            return

        ret, frame = self.capture.read()
        if not ret:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(frame_rgb)

        nombre_detectado = "Desconocido"
        rostro_detectado = False

        if results.detections:
            alto, ancho, _ = frame.shape
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            x1 = int(bbox.xmin * ancho)
            y1 = int(bbox.ymin * alto)
            w = int(bbox.width * ancho)
            h = int(bbox.height * alto)

            margen = 20
            x1m = max(0, x1 - margen)
            y1m = max(0, y1 - margen)
            x2m = min(ancho, x1 + w + margen)
            y2m = min(alto, y1 + h + margen)

            face_img = frame_rgb[y1m:y2m, x1m:x2m]
            if face_img.shape[0] < 112 or face_img.shape[1] < 112:
                return

            rostro_detectado = True
            embedding = self.tflite_recognizer.get_embedding(face_img)

            if self.conocidos_encodings:
                similitudes = [np.dot(embedding, known) for known in self.conocidos_encodings]
                idx_max = np.argmax(similitudes)
                similitud_max = similitudes[idx_max]
                nombre_detectado = self.conocidos_nombres[idx_max]

                print(f"[DEBUG] Similitudes: {similitudes}")
                print(f"[DEBUG] Máxima similitud: {similitud_max:.4f} - nombre: {nombre_detectado}")

                self.label_distancia.text = f"Similitud: {similitud_max:.4f}"
                self.label_distancia.color = (0, 1, 0, 1)

            cv2.rectangle(frame, (x1m, y1m), (x2m, y2m), (0, 255, 0), 2)
            cv2.putText(frame, nombre_detectado, (x1m, y1m - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if rostro_detectado and nombre_detectado != "Desconocido" and not self.rostro_reconocido_actualmente:
            self.label_bienvenida.text = f"Bienvenido, {nombre_detectado}"
            self.ultimo_nombre_mostrado = nombre_detectado
            self.rostro_reconocido_actualmente = True
            tiempo_actual = time.time()
            if not self.ble_activado and (tiempo_actual - self.ultima_activacion_ble > 10):
                self.ble_activado = True
                self.ultima_activacion_ble = tiempo_actual
                if "menu" in self.manager.screen_names:
                    self.manager.get_screen("menu").activar_face_ok()
                asyncio.run_coroutine_threadsafe(self.activar_rele_ble(), loop)

        elif not rostro_detectado:
            self.label_bienvenida.text = ""
            self.label_distancia.text = ""
            self.ultimo_nombre_mostrado = ""
            self.rostro_reconocido_actualmente = False

        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img1.texture = texture

    async def activar_rele_ble(self):
        print("[DEBUG] Activando rele BLE...")
        try:
            async with BleakClient(BLE_DEVICE_ADDRESS) as client:
                await client.write_gatt_char(BLE_CHARACTERISTIC_UUID, bytearray([1]))
                await asyncio.sleep(5)
                await client.write_gatt_char(BLE_CHARACTERISTIC_UUID, bytearray([0]))
        except Exception as e:
            print(f"Error BLE: {e}")
        finally:
            self.ble_activado = False
