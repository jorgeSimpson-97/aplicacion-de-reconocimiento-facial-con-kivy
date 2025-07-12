import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

FACES_DIR = "data/faces"
EMBEDDINGS_DIR = "data/embeddings"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Inicializamos MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Cargamos el modelo TFLite de embeddings
TFLITE_MODEL_PATH = "mobilefacenet.tflite"
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def run_tflite_embedding(image_rgb):
    """Ejecuta el modelo TFLite para obtener el embedding."""
    input_shape = input_details[0]['shape']
    img = cv2.resize(image_rgb, (input_shape[2], input_shape[1]))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    embedding = interpreter.get_tensor(output_details[0]['index'])[0]
    embedding = embedding / np.linalg.norm(embedding)  # Normalizar embedding
    return embedding

def detectar_rostro_mediapipe(image_rgb):
    results = face_detection.process(image_rgb)
    if not results.detections:
        return None
    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box
    h, w, _ = image_rgb.shape
    x_min = max(0, int(bbox.xmin * w))
    y_min = max(0, int(bbox.ymin * h))
    box_w = int(bbox.width * w)
    box_h = int(bbox.height * h)
    x_max = min(w, x_min + box_w)
    y_max = min(h, y_min + box_h)
    rostro = image_rgb[y_min:y_max, x_min:x_max]

    if rostro.shape[0] < 50 or rostro.shape[1] < 50:
        return None
    return rostro

def procesar_usuario(ruta_usuario):
    embeddings = []
    for archivo in sorted(os.listdir(ruta_usuario)):
        if archivo.lower().endswith(('.jpg', '.jpeg', '.png')):
            ruta_img = os.path.join(ruta_usuario, archivo)
            img_bgr = cv2.imread(ruta_img)
            if img_bgr is None:
                print(f"No se pudo abrir imagen: {ruta_img}")
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            rostro = detectar_rostro_mediapipe(img_rgb)
            if rostro is None:
                print(f"No se detectó rostro válido en {ruta_img}")
                continue
            try:
                embedding = run_tflite_embedding(rostro)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error extrayendo embedding en {ruta_img}: {e}")

    if embeddings:
        promedio = np.mean(embeddings, axis=0)
        promedio = promedio / np.linalg.norm(promedio)
        return promedio
    else:
        print(f"No se generaron embeddings válidos para {os.path.basename(ruta_usuario)}.")
        return None

def procesar_todos_los_usuarios():
    usuarios = [u for u in os.listdir(FACES_DIR) if os.path.isdir(os.path.join(FACES_DIR, u))]
    for usuario in usuarios:
        ruta_usuario = os.path.join(FACES_DIR, usuario)
        print(f"Procesando usuario: {usuario}")
        embedding_usuario = procesar_usuario(ruta_usuario)
        if embedding_usuario is not None:
            ruta_guardado = os.path.join(EMBEDDINGS_DIR, f"{usuario}.npy")
            np.save(ruta_guardado, embedding_usuario)
            print(f"Embedding guardado en {ruta_guardado}")
        else:
            print(f"No se pudo generar embedding para {usuario}")

if __name__ == "__main__":
    procesar_todos_los_usuarios()
