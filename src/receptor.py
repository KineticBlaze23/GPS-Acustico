from pathlib import Path
import serial
import struct
import time
import numpy as np
import matplotlib.pyplot as plt

# Definición de rutas basadas en la raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent
DIR_RAW = ROOT_DIR / "data" / "raw"
DIR_RAW.mkdir(parents=True, exist_ok=True)
ARCHIVO_SALIDA = DIR_RAW / "grabacion_cruda.npz"

PUERTO = 'COM4'         # Ajusta al puerto COM de tu ESP32-S3
BAUDRATE = 2000000
DURACION_SEG = 20       # Segundos de grabación
HEADER = b'\xAA\x55\xAA\x55'

ser = serial.Serial(PUERTO, BAUDRATE, timeout=1)
ser.reset_input_buffer()
print(f"Grabando {DURACION_SEG} segundos desde {PUERTO}...")

timestamps_bloques, conteo_muestras, todas_las_muestras = [], [], []
buffer = bytearray()
muestras_objetivo = 40000 * DURACION_SEG
t0 = time.time()

try:
    while len(todas_las_muestras) < muestras_objetivo:
        if time.time() - t0 > (DURACION_SEG + 4.0) and len(todas_las_muestras) == 0:
            raise TimeoutError("Sin datos desde ESP32-S3. Cierra el Monitor Serie y presiona RST.")
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

muestras_np = np.array(todas_las_muestras, dtype=np.float64)
timestamps_np = np.array(timestamps_bloques, dtype=np.int64)
conteos_np = np.array(conteo_muestras, dtype=np.int64)

tiempo_total_s = (timestamps_np[-1] - timestamps_np[0]) * 1e-6
fs_real = np.sum(conteos_np[1:]) / tiempo_total_s
voltaje = (muestras_np / 4095.0) * 3.3
tiempo_s = np.arange(len(voltaje)) / fs_real

np.savez(ARCHIVO_SALIDA, voltaje=voltaje, fs_real=fs_real)
print(f"Grabación guardada en '{ARCHIVO_SALIDA}' (fs_real = {fs_real:.2f} Hz, {len(voltaje)} muestras).")

# --- GRÁFICA DE LA ONDA CAPTURADA ---
ac = voltaje - np.mean(voltaje)  # Señal centrada en 0 V para ver mejor los pulsos

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

# 1. Señal completa durante toda la grabación
ax1.plot(tiempo_s, ac, color='steelblue', linewidth=0.6)
ax1.set_title(f"Onda Acústica Capturada (INMP441 @ {fs_real:.1f} Hz)")
ax1.set_xlabel("Tiempo [s]")
ax1.set_ylabel("Amplitud AC [V]")
ax1.grid(True, alpha=0.3)

# 2. Zoom de 15 ms alrededor del pulso más fuerte para inspeccionar la forma de onda
idx_pico = np.argmax(np.abs(ac))
ventana = int(0.0075 * fs_real)
ini = max(0, idx_pico - ventana)
fin = min(len(ac), idx_pico + ventana)

ax2.plot(tiempo_s[ini:fin] * 1000.0, ac[ini:fin], 'o-', color='darkorange', markersize=3, linewidth=1.2)
ax2.set_title("Detalle (Zoom 15 ms) de un Pulso Recibido")
ax2.set_xlabel("Tiempo [ms]")
ax2.set_ylabel("Amplitud AC [V]")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
