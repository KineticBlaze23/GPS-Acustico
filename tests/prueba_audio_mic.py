from pathlib import Path
import struct
import time

import numpy as np
import serial
import winsound  # Incluido por defecto en Windows para reproducir .wav
from scipy.io.wavfile import write

# Definición de rutas basadas en la raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent
DIR_TESTS = ROOT_DIR / "tests"
DIR_TESTS.mkdir(parents=True, exist_ok=True)
ARCHIVO_WAV = DIR_TESTS / "prueba_mic.wav"

# --- CONFIGURACIÓN ---
PUERTO = 'COM4'
BAUDRATE = 2000000
DURACION_SEG = 5
FS_NOMINAL = 40000       # Debe coincidir con SAMPLE_RATE del firmware
HEADER = b'\xAA\x55\xAA\x55'
FMT_CAB = '<4sIqIqH'     # sync, seq, ts_esp_us, sqw_n, sqw_us, n  (formato del firmware nuevo)
TAM_CAB = struct.calcsize(FMT_CAB)   # 30 bytes
MAX_N = 1024

ser = serial.Serial(PUERTO, BAUDRATE, timeout=1)
ser.reset_input_buffer()
print(f"Grabando {DURACION_SEG} segundos desde el micrófono ESP32-S3... ¡Habla o reproduce el chirp!")

bloques, muestras = [], []
total, buffer = 0, bytearray()
objetivo = FS_NOMINAL * DURACION_SEG
t_ini = time.time()

try:
    while total < objetivo:
        if total == 0 and time.time() - t_ini > DURACION_SEG + 4.0:
            raise TimeoutError("No llegan datos del ESP32-S3. Verifica el firmware, el puerto COM y que Arduino IDE esté cerrado.")
        buffer.extend(ser.read(max(1, ser.in_waiting)))
        while True:
            idx = buffer.find(HEADER)
            if idx == -1:
                del buffer[:-3]
                break
            if idx > 0:
                del buffer[:idx]
            if len(buffer) < TAM_CAB:
                break
            _, seq, ts, sqw_n, sqw_us, n = struct.unpack(FMT_CAB, bytes(buffer[:TAM_CAB]))
            if n == 0 or n > MAX_N:          # falso encabezado dentro de los datos
                del buffer[:1]
                continue
            tam = TAM_CAB + 2 * n
            if len(buffer) < tam:
                break
            muestras.append(np.frombuffer(bytes(buffer[TAM_CAB:tam]), dtype='<i2'))
            bloques.append((seq, ts, sqw_n, sqw_us, n))
            total += n
            del buffer[:tam]
finally:
    ser.close()

if len(bloques) < 2:
    raise RuntimeError("Se recibieron muy pocos bloques.")

b = np.array(bloques, dtype=np.int64)
seq, ts, sqw_n, sqw_us, n = b.T

perdidos = int(np.sum(np.diff(seq) - 1))
if perdidos > 0:
    print(f"ATENCIÓN: se perdieron {perdidos} bloques en la transmisión.")

# Frecuencia de muestreo: regresión contra el DS3231 si hay flancos; si no, contra el reloj del ESP32
x = np.cumsum(n).astype(np.float64)
ok = sqw_n > 0
if ok.sum() >= 20:
    t_ref = sqw_n + (ts - sqw_us) * 1e-6
    a, _ = np.polyfit(x[ok], t_ref[ok], 1)
    fuente = "DS3231"
    fs_real = 1.0 / a
else:
    a, _ = np.polyfit(x, ts * 1e-6, 1)
    fuente = "reloj del ESP32 (sin pulsos SQW del DS3231)"
    fs_real = 1.0 / a
if not (39000.0 <= fs_real <= 41000.0):
    print(f"AVISO: fs_real = {fs_real:.1f} Hz fuera de rango; se usa el nominal {FS_NOMINAL} Hz para el WAV.")
    fs_wav = FS_NOMINAL
else:
    fs_wav = int(round(fs_real))

# Quitar el offset DC del micrófono y normalizar el volumen para escuchar con claridad
audio = np.concatenate(muestras).astype(np.float64)
audio_centrado = audio - np.mean(audio)
max_val = np.max(np.abs(audio_centrado))
print(f"Amplitud pico de la señal: {max_val:.0f} cuentas de 32768 ({max_val / 32768 * 100:.2f} % de la escala)")
if max_val < 300:
    print("AVISO: la señal es muy débil. Acerca la fuente o baja SHIFT_BITS en el firmware (16 -> 14 o 12).")
audio_norm = (audio_centrado / max_val) * 0.9 if max_val > 0 else audio_centrado
write(str(ARCHIVO_WAV), fs_wav, (audio_norm * 32767).astype(np.int16))

print(f"\nGrabación lista (fs = {fs_real:.1f} Hz según {fuente}). Guardado en '{ARCHIVO_WAV}'.")
print("Reproduciendo lo que escuchó el micrófono...")
winsound.PlaySound(str(ARCHIVO_WAV), winsound.SND_FILENAME)
print("Reproducción finalizada.")