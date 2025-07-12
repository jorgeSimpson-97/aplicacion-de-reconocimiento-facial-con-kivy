
import os

DATA_DIR = "data"
PIN_FILE = os.path.join(DATA_DIR, "pin.txt")

def guardar_pin(pin: str) -> None:
    """
    Guarda el PIN en un archivo de texto.
    Sobrescribe el archivo si ya existe.
    """
    with open(PIN_FILE, "w") as f:
        f.write(pin)

def verificar_pin(pin: str) -> bool:
    """
    Verifica si el PIN ingresado coincide con el guardado en archivo.
    Retorna True si coincide, False si no o si no existe el archivo.
    """
    if not os.path.exists(PIN_FILE):
        return False

    with open(PIN_FILE, "r") as f:
        pin_guardado = f.read().strip()

    return pin == pin_guardado
