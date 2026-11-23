from pathlib import Path
import serial
import struct
import time
import numpy as np
from scipy.io.wavfile import write
import winsound  # Incluido por defecto en Windows para reproducir .wav

# Definición de rutas basadas en la raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent
DIR_TESTS = ROOT_DIR / "tests"
DIR_TESTS.mkdir(parents=True, exist_ok=True)
ARCHIVO_WAV = DIR_TESTS / "prueba_mic.wav"

# --- CONFIGURACIÓN ---
PUERTO = 'COM4'
BAUDRATE = 2000000
DURACION_SEG = 5
HEADER = b'\xAA\x55\xAA\x55'

ser = serial.Serial(PUERTO, BAUDRATE, timeout=1)
ser.reset_input_buffer()

print(f"Grabando {DURACION_SEG} segundos desde el micrófono ESP32-S3... ¡Habla o reproduce el chirp!")

timestamps_bloques, conteo_muestras, todas_las_muestras = [], [], []
buffer = bytearray()
muestras_objetivo = 40000 * DURACION_SEG
tiempo_inicio = time.time()

try:
    while len(todas_las_muestras) < muestras_objetivo:
        if time.time() - tiempo_inicio > (DURACION_SEG + 4.0) and len(todas_las_muestras) == 0:
            raise TimeoutError("No llegan datos del ESP32-S3. Verifica que esté cargado el firmware I2S y cerrado Arduino IDE.")
        
        chunk = ser.read(4096)
        if chunk:
            buffer.extend(chunk)
            
        while True:
            idx = buffer.find(HEADER)
            if idx == -1:
                if len(buffer) > 3:
                    buffer = buffer[-3:]
                break
            if idx > 0:
                buffer = buffer[idx:]
            if len(buffer) < 14:
                break
                
            ts_us, num_muestras = struct.unpack('<qH', buffer[4:14])
            tamano_paquete = 14 + num_muestras * 2
            if len(buffer) < tamano_paquete:
                break
                
            datos_crudos = np.frombuffer(buffer[14:tamano_paquete], dtype=np.uint16)
            timestamps_bloques.append(ts_us)
            conteo_muestras.append(num_muestras)
            todas_las_muestras.extend(datos_crudos)
            buffer = buffer[tamano_paquete:]
finally:
    ser.close()

# Calcular frecuencia real y convertir a audio normalizado de 16 bits
muestras_np = np.array(todas_las_muestras, dtype=np.float64)
timestamps_np = np.array(timestamps_bloques, dtype=np.int64)
conteos_np = np.array(conteo_muestras, dtype=np.int64)

tiempo_total_s = (timestamps_np[-1] - timestamps_np[0]) * 1e-6
fs_real = int(round(np.sum(conteos_np[1:]) / tiempo_total_s))

# Quitar offset DC (~2048) y normalizar volumen para escuchar con claridad
audio_centrado = muestras_np - np.mean(muestras_np)
max_val = np.max(np.abs(audio_centrado))
if max_val > 0:
    audio_norm = (audio_centrado / max_val) * 0.9
else:
    audio_norm = audio_centrado

audio_int16 = (audio_norm * 32767).astype(np.int16)
write(str(ARCHIVO_WAV), fs_real, audio_int16)

print(f"\nGrabación lista ({fs_real} Hz). Guardado en '{ARCHIVO_WAV}'.")
print("Reproduciendo lo que escuchó el micrófono...")

# Reproducir automáticamente en Windows
winsound.PlaySound(str(ARCHIVO_WAV), winsound.SND_FILENAME)
print("Reproducción finalizada.")
