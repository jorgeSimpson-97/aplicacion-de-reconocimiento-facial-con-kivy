import requests

def send_command(command):
    """
    Envía un comando HTTP al ESP32-CAM por WiFi.
    Cambia la IP y el endpoint según tu configuración.
    """
    ESP32CAM_IP = "192.168.4.1"  # Cambia por la IP real del ESP32-CAM
    ENDPOINT = f"http://{ESP32CAM_IP}/comando?cmd={command}"
    try:
        response = requests.get(ENDPOINT, timeout=2)
        return response.status_code == 200
    except Exception as e:
        print(f"[ERROR] No se pudo enviar el comando al ESP32-CAM: {e}")
        return False
