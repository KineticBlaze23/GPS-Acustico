from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Definición de rutas basadas en la raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent
ARCHIVO_GRABACION = ROOT_DIR / "data" / "raw" / "grabacion_cruda.npz"
ARCHIVO_PLANTILLAS = ROOT_DIR / "data" / "processed" / "plantillas_senales.npz"

C_SONIDO = 343.0    # Velocidad del sonido en interiores [m/s]
PERIODO_WAV = 1.0   # Periodo de repetición del audio [s]

# 1. Cargar grabación y plantilla (usa 'chirp' para máxima inmunidad a ecos)
datos = np.load(ARCHIVO_GRABACION)
voltaje = datos['voltaje']
fs_real = float(datos['fs_real'])
senal = voltaje - np.mean(voltaje)
tiempo_s = np.arange(len(senal)) / fs_real

plantillas = np.load(ARCHIVO_PLANTILLAS)
fs_audio = float(plantillas['fs_audio'])
plantilla_orig = plantillas['pulso']  # Cambia a 'pulso' solo si reproduces senal_pulso.wav

# 2. Interpolación lineal de la plantilla a fs_real (~40 kHz)
t_orig = np.arange(len(plantilla_orig)) / fs_audio
t_nuevo = np.arange(0, t_orig[-1], 1.0 / fs_real)
plantilla = np.interp(t_nuevo, t_orig, plantilla_orig)

# 3. Correlación cruzada directa (producto punto deslizante)
corr = np.abs(np.convolve(senal, plantilla[::-1], mode='valid'))

# 4. Nivel de ruido base y umbral de flanco de subida (primer frente directo)
piso_ruido = np.median(corr)
umbral_flanco = 7.0 * piso_ruido

# Buscar el primer pulso válido entre 0.05 s y 1.05 s
ini_0 = int(0.05 * fs_real)
fin_0 = int(1.05 * fs_real)
seg_0 = corr[ini_0:fin_0]
idx_pico_0 = ini_0 + int(np.argmax(seg_0))

# Retroceder hasta 4 ms antes del pico para hallar el primer cruce del umbral (inicio real del frente)
ini_frente_0 = max(ini_0, idx_pico_0 - int(0.004 * fs_real))
idx_actual = idx_pico_0
for j in range(ini_frente_0, idx_pico_0 + 1):
    if corr[j] >= umbral_flanco:
        idx_actual = j
        break

indices_pulsos = [idx_actual]
paso_muestras = int(round(PERIODO_WAV * fs_real))
ventana_tracking = int(0.006 * fs_real)  # ±6 ms (±2.0 m/s máximo)

while True:
    centro = idx_actual + paso_muestras
    ini = centro - ventana_tracking
    fin = centro + ventana_tracking
    if fin >= len(corr):
        break
    seg = corr[ini:fin]
    if np.max(seg) < umbral_flanco:
        break
    # Recorrer de izquierda a derecha: el PRIMER punto que cruza umbral_flanco es el frente directo
    idx_encontrado = ini + int(np.argmax(seg))
    for j in range(len(seg)):
        if seg[j] >= umbral_flanco:
            idx_encontrado = ini + j
            break
    idx_actual = idx_encontrado
    indices_pulsos.append(idx_actual)

# 5. Interpolación lineal exacta en el cruce del umbral (precisión sub-muestra)
tiempos = []
for idx in indices_pulsos:
    if idx > 0 and corr[idx] != corr[idx - 1]:
        frac = (umbral_flanco - corr[idx - 1]) / (corr[idx] - corr[idx - 1])
        frac = np.clip(frac, 0.0, 1.0)
    else:
        frac = 0.0
    tiempos.append((idx - 1 + frac) / fs_real)
tiempos = np.array(tiempos)

if len(tiempos) > 2 and abs((tiempos[1] - tiempos[0]) - PERIODO_WAV) > 0.015:
    tiempos = tiempos[1:]
    indices_pulsos = indices_pulsos[1:]

# 6. Tiempo de vuelo (ToF) bruto
ciclos = np.arange(len(tiempos))
t_teoricos = tiempos[0] + ciclos * PERIODO_WAV
dist_brutas = C_SONIDO * (tiempos - t_teoricos)

# 7. Mínimos cuadrados lineales sobre los primeros 4 s en reposo
n_cal = min(4, len(tiempos))
if n_cal >= 2:
    x_cal = tiempos[:n_cal] - tiempos[0]
    y_cal = dist_brutas[:n_cal]
    x_prom = np.mean(x_cal)
    y_prom = np.mean(y_cal)
    pendiente_deriva = np.sum((x_cal - x_prom) * (y_cal - y_prom)) / np.sum((x_cal - x_prom)**2)
    distancias_m = dist_brutas - pendiente_deriva * (tiempos - tiempos[0])
else:
    distancias_m = dist_brutas - dist_brutas[0]

# --- SALIDA POR CONSOLA ---
print("\n==================================================")
print("   POSICIONAMIENTO ACÚSTICO 1D (FLANCO DIRECTO)   ")
print("==================================================")
print(f"Frecuencia de muestreo real: {fs_real:.2f} Hz")
print(f"Pulsos válidos detectados:   {len(distancias_m)}\n")

for k in range(len(distancias_m)):
    print(f"Pulso {k+1:2d} (t={tiempos[k]:5.2f}s) -> Posición: {distancias_m[k]:+6.2f} m")

if len(distancias_m) >= 2:
    print(f"\nPosición final estimada: {distancias_m[-1]:+.2f} m")

# --- GRÁFICAS ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9))

ax1.plot(tiempo_s, senal, color='steelblue', linewidth=0.6, label='Señal recibida [V]')
ax1.plot(tiempos, senal[indices_pulsos], 'ro', markersize=5, label='Inicio frente directo')
ax1.set_title(f"1. Onda Acústica Capturada (INMP441 @ {fs_real:.1f} Hz)")
ax1.set_xlabel("Tiempo [s]")
ax1.set_ylabel("Amplitud [V]")
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

idx_zoom = indices_pulsos[0]
ventana_z = int(0.0075 * fs_real)
ini_z = max(0, idx_zoom - ventana_z)
fin_z = min(len(senal), idx_zoom + ventana_z)
ax2.plot(tiempo_s[ini_z:fin_z] * 1000.0, senal[ini_z:fin_z], 'o-', color='darkorange', markersize=3, linewidth=1.2)
ax2.set_title("2. Detalle (Zoom 15 ms) del Primer Pulso")
ax2.set_xlabel("Tiempo [ms]")
ax2.set_ylabel("Amplitud [V]")
ax2.grid(True, alpha=0.3)

ax3.plot(tiempos, distancias_m, 'o-', color='royalblue', linewidth=2, label='Distancia estimada [m]')
ax3.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax3.set_title("3. Posicionamiento 1D (Detección por Flanco + Mínimos Cuadrados)")
ax3.set_xlabel("Tiempo [s]")
ax3.set_ylabel("Distancia [m]")
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
