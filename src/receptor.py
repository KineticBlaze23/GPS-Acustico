from pathlib import Path
import shutil
import struct
import time

import matplotlib.pyplot as plt
import numpy as np
import serial

# ----------------------------------------------------------------------------
# RUTAS (basadas en la raíz del proyecto)
# ----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DIR_RAW = ROOT_DIR / "data" / "raw"
DIR_RAW.mkdir(parents=True, exist_ok=True)

ETIQUETA = "corrida_01"                              # Nombre de esta grabación (copia de respaldo)
ARCHIVO_SALIDA = DIR_RAW / "grabacion_cruda.npz"     # Es el que leen los scripts de procesamiento
ARCHIVO_COPIA = DIR_RAW / f"{ETIQUETA}.npz"          # Copia con nombre propio para no perder datos

# ----------------------------------------------------------------------------
# PARÁMETROS
# ----------------------------------------------------------------------------
PUERTO = 'COM4'          # Puerto COM del ESP32-S3
BAUDRATE = 2000000
DURACION_SEG = 46        # Segundos de grabación (el protocolo de validación necesita ~54 s)
FS_NOMINAL = 40000       # Debe coincidir con SAMPLE_RATE del firmware
SHIFT_BITS = 16          # Debe coincidir con SHIFT_BITS del firmware
SENSIBILIDAD_FS_POR_PA = 0.0501   # INMP441: -26 dBFS a 1 Pa (típico, ±1 dB). Solo para escalar a Pa.

HEADER = b'\xAA\x55\xAA\x55'
FMT_CAB = '<4sIqIqH'     # sync, seq, ts_esp_us, sqw_n, sqw_us, n
TAM_CAB = struct.calcsize(FMT_CAB)   # 30 bytes
MAX_N = 1024

# ----------------------------------------------------------------------------
# 1. ADQUISICIÓN
# ----------------------------------------------------------------------------
ser = serial.Serial(PUERTO, BAUDRATE, timeout=1)
ser.reset_input_buffer()
print(f"Grabando {DURACION_SEG} s desde {PUERTO}...")

bloques, muestras = [], []
total, buffer = 0, bytearray()
objetivo = FS_NOMINAL * DURACION_SEG
t0 = time.time()

try:
    while total < objetivo:
        if total == 0 and time.time() - t0 > 10.0:
            raise TimeoutError("Sin datos del ESP32-S3. Cierra el Monitor Serie y presiona RST.")
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

if len(bloques) < 25:
    raise RuntimeError("Se recibieron muy pocos bloques. Revisa puerto, firmware y cableado.")

b = np.array(bloques, dtype=np.int64)
seq, ts, sqw_n, sqw_us, n = b.T

# ----------------------------------------------------------------------------
# 2. CONTROL DE CALIDAD: BLOQUES PERDIDOS Y RETRASOS
# ----------------------------------------------------------------------------
perdidos = int(np.sum(np.diff(seq) - 1))
if perdidos > 0:
    print(f"ATENCIÓN: se perdieron {perdidos} bloques en la transmisión. Repite la grabación.")

# El DMA puede descartar datos sin que 'seq' salte: se detecta por intervalos anómalos.
dt_bloque = np.diff(ts) * 1e-6
esperado = n[1:] / FS_NOMINAL
retrasados = int(np.sum(dt_bloque > 1.5 * esperado))
if retrasados > 0:
    print(f"AVISO: {retrasados} bloques llegaron con retraso > 50 % (posible pérdida en el DMA).")

# ----------------------------------------------------------------------------
# 3. FRECUENCIA DE MUESTREO REAL CONTRA EL DS3231
# ----------------------------------------------------------------------------
ok = sqw_n > 0
if ok.sum() < 20:
    raise RuntimeError("No llegan flancos SQW del DS3231. Revisa cableado y pull-up.")

x = np.cumsum(n).astype(np.float64)                 # índice de muestra al final de cada bloque
t_ds = sqw_n + (ts - sqw_us) * 1e-6                 # tiempo DS3231 de cada bloque [s]
xc = x[ok] - x[ok].mean()
a, c = np.polyfit(xc, t_ds[ok], 1)
fs_real = 1.0 / a
res_us = (t_ds[ok] - (a * xc + c)) * 1e6
rms_us = float(np.std(res_us))
ppm = (fs_real / FS_NOMINAL - 1.0) * 1e6

print(f"fs_real = {fs_real:.3f} Hz ({ppm:+.0f} ppm vs nominal)")
print(f"Residuo de la regresión: rms = {rms_us:.0f} µs")
if not (39000.0 <= fs_real <= 41000.0):
    print("ATENCIÓN: fs_real fuera de rango; revisa la grabación (no se corrige en silencio).")

# ----------------------------------------------------------------------------
# 4. GUARDADO DE DATOS CRUDOS + MARCAS DE TIEMPO
# ----------------------------------------------------------------------------
voltaje = np.concatenate(muestras).astype(np.float64) * (2.0 ** (SHIFT_BITS - 16)) / 32768.0  # fracción de escala completa

np.savez(
    ARCHIVO_SALIDA,
    voltaje=voltaje, fs_real=fs_real, perdidos=perdidos,
    seq=seq, ts_esp_us=ts, sqw_n=sqw_n, sqw_us=sqw_us, n_muestras=n,
    fs_ppm=ppm, residuo_rms_us=rms_us,
)
shutil.copyfile(ARCHIVO_SALIDA, ARCHIVO_COPIA)
print(f"Guardado en {ARCHIVO_SALIDA} ({len(voltaje)} muestras)")
print(f"Copia de respaldo: {ARCHIVO_COPIA}")

# ----------------------------------------------------------------------------
# 5. GRÁFICA EN UNIDADES FÍSICAS
# ----------------------------------------------------------------------------
presion_pa = (voltaje - np.mean(voltaje)) / SENSIBILIDAD_FS_POR_PA   # presión sonora aprox. [Pa]
tiempo_s = np.arange(len(voltaje)) / fs_real

plt.figure(figsize=(11, 4.5))
plt.plot(tiempo_s, presion_pa, color='steelblue', linewidth=0.6)
plt.title(f"Señal acústica capturada (INMP441 @ {fs_real:.1f} Hz)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Presión sonora [Pa]")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()