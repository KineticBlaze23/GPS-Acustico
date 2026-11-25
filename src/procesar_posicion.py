from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import find_peaks

# Definición de rutas basadas en la raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent
DIR_PROCESSED = ROOT_DIR / "data" / "processed"
DIR_RAW = ROOT_DIR / "data" / "raw"

DIR_PROCESSED.mkdir(parents=True, exist_ok=True)

ARCHIVO_GRABACION = DIR_RAW / "grabacion_cruda.npz"
ARCHIVO_PLANTILLAS = DIR_PROCESSED / "plantillas_senales.npz"
ARCHIVO_EVIDENCIA = DIR_PROCESSED / "evidencia_posicion_acustica.png"

# --- PARÁMETROS FÍSICOS Y DE EMISIÓN ---
C_SONIDO = 343.0          # Velocidad del sonido en el aire [m/s] a 20 °C
PERIODO_WAV = 2.0         # Intervalo nominal entre emisiones acústicas [s]

# 1. Cargar grabación capturada por el micrófono
if not ARCHIVO_GRABACION.exists():
    print(f"ERROR: No se encontró el archivo de grabación en {ARCHIVO_GRABACION}")
    exit(1)

datos = np.load(ARCHIVO_GRABACION)
voltaje = datos['voltaje']
fs_guardada = float(datos['fs_real']) if 'fs_real' in datos else 40000.0

# Validación de la frecuencia de muestreo de hardware
if 39000.0 <= fs_guardada <= 41000.0:
    fs_real = fs_guardada
else:
    fs_real = 40000.0

senal = voltaje - np.mean(voltaje)
tiempo_s = np.arange(len(senal)) / fs_real

# 2. Cargar plantilla del pulso de referencia para correlación cruzada
if not ARCHIVO_PLANTILLAS.exists():
    print(f"ERROR: No se encontró el archivo de plantilla en {ARCHIVO_PLANTILLAS}")
    exit(1)

plantillas = np.load(ARCHIVO_PLANTILLAS)
fs_audio = float(plantillas['fs_audio'])
plantilla_orig = plantillas['pulso']
if 'periodo' in plantillas:
    PERIODO_WAV = float(plantillas['periodo'])

# Adaptar plantilla a la frecuencia de muestreo real del micrófono
t_orig = np.arange(len(plantilla_orig)) / fs_audio
t_nuevo = np.arange(0, t_orig[-1], 1.0 / fs_real)
plantilla = np.interp(t_nuevo, t_orig, plantilla_orig)

# 3. Correlación cruzada (Filtro Adaptado para máxima precisión de ToA)
corr = np.abs(np.convolve(senal, plantilla[::-1], mode='valid'))

piso_ruido = np.median(corr)
umbral_pico = max(3.5 * piso_ruido, 0.08 * np.max(corr))
dist_min_muestras = int(0.70 * PERIODO_WAV * fs_real)

indices_picos, _ = find_peaks(corr, height=umbral_pico, distance=dist_min_muestras)

# Descartar último pico si quedó truncado al apagarse la grabación
if len(indices_picos) > 1 and (indices_picos[-1] - indices_picos[-2]) < int(0.75 * PERIODO_WAV * fs_real):
    indices_picos = indices_picos[:-1]

if len(indices_picos) == 0:
    print("ERROR: No se detectaron pulsos acústicos válidos en la grabación.")
    exit(1)

# 4. Estimación de Tiempo de Llegada (ToA) con Interpolación Parabólica Sub-muestra
picos_sub = []
for p in indices_picos:
    if 0 < p < len(corr) - 1:
        y1, y2, y3 = corr[p - 1], corr[p], corr[p + 1]
        delta = (y1 - y3) / (2 * (y1 - 2 * y2 + y3) + 1e-12)
        picos_sub.append(p + np.clip(delta, -0.5, 0.5))
    else:
        picos_sub.append(float(p))
picos_sub = np.array(picos_sub)

tiempos = picos_sub / fs_real

# 5. Cálculo de Posición Acústica Pura (Sin referencias iniciales manuales)
# El primer pulso recibido establece el origen acústico de la medición (d_1 = 0.00 cm)
t_origen = tiempos[0]
ciclos = np.arange(len(tiempos))
tiempos_esperados = t_origen + ciclos * PERIODO_WAV

# Desplazamiento acústico respecto al origen inicial: delta_t * c
delta_t = tiempos - tiempos_esperados
desplazamiento_m = delta_t * C_SONIDO

# Todas las distancias en centímetros y positivas (>= 0.00 cm)
distancias_cm = np.maximum(0.0, desplazamiento_m * 100.0)

# 6. Estadísticas y Error Teórico de Métodos Numéricos
distancia_final_cm = float(distancias_cm[-1])
tiempo_final_s = float(tiempos[-1])

# Error teórico de discretización espacial a fs: e = c / (2 * fs)
error_teorico_cm = float((C_SONIDO / (2.0 * fs_real)) * 100.0)

# --- REPORTE CONSOLA LIMPIO Y CLARO ---
print("\n" + "=" * 55)
print("          MEDICIÓN DE POSICIÓN ACÚSTICA (1D)")
print("=" * 55)
print(f"Total de pulsos acústicos medidos: {len(distancias_cm)}\n")

print(f"{'Pulso':<12} {'Tiempo (s)':<18} {'Distancia (cm)':<18}")
print("-" * 50)

for k in range(len(distancias_cm)):
    print(f"Pulso {k+1:<5}   t = {tiempos[k]:5.2f} s           {distancias_cm[k]:6.2f} cm")

print("-" * 50)
print(f"Distancia final (t = {tiempo_final_s:.2f} s):   {distancia_final_cm:.2f} cm")
print(f"Error teórico de medición:       ±{error_teorico_cm:.2f} cm")
rango_min = max(0.0, distancia_final_cm - error_teorico_cm)
rango_max = distancia_final_cm + error_teorico_cm
print(f"Rango aceptable esperado:        [{rango_min:.2f} cm , {rango_max:.2f} cm]")
print("=" * 55 + "\n")

# --- GRÁFICAS DE EVIDENCIA (20 SEGUNDOS COMPLETOS, EJES CADA 2s Y CADA 1s) ---
duracion_total = int(np.ceil(tiempo_s[-1])) if len(tiempo_s) > 0 else 20
max_x = max(20, duracion_total)
y_max = np.max(np.abs(senal))

# Encontrar el pico de oscilación en la onda real para el punto rojo
ventana = int(0.010 * fs_real)
picos_y = []
for p in indices_picos:
    ini = max(0, p - ventana)
    fin = min(len(senal), p + ventana)
    local_idx = ini + np.argmax(senal[ini:fin])
    picos_y.append(senal[local_idx])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

# Panel 1: Señal Acústica Completa en el Dominio del Tiempo
ax1.plot(tiempo_s, senal, color='#2c3e50', linewidth=0.6, label='Audio capturado [V]')
ax1.plot(tiempos, picos_y, 'ro', markersize=6, label='Pulsos detectados')
ax1.set_ylim(-1.5 * y_max, 1.65 * y_max)

# Evidenciar cada pulso con su etiqueta en el Panel 1
for k, t in enumerate(tiempos):
    ax1.annotate(
        f"P{k+1} ({t:.2f}s)",
        xy=(t, picos_y[k]),
        xytext=(t, 1.35 * y_max),
        textcoords='data',
        ha='center',
        va='center',
        fontsize=8,
        fontweight='bold',
        color='#900C3F',
        arrowprops=dict(arrowstyle='->', color='#C70039', lw=1.0, alpha=0.7),
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#FDEDEC', edgecolor='#C70039', alpha=0.9)
    )

ax1.set_title(f"1. Señal Acústica Completa (0 a {max_x} s @ {fs_real:.1f} Hz)", fontsize=11, fontweight='bold')
ax1.set_ylabel("Amplitud [V]", fontsize=10)
ax1.set_xlim(0, max_x)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1.0))
ax1.grid(True, which='major', color='#7f8c8d', linestyle='-', alpha=0.35)
ax1.grid(True, which='minor', color='#bdc3c7', linestyle=':', alpha=0.25)
ax1.legend(loc='lower right', framealpha=0.95, ncol=2)

# Panel 2: Distancia Acústica Medida (Posición vs Tiempo)
max_d = max(distancias_cm) if len(distancias_cm) > 0 else 100.0

ax2.plot(tiempos, distancias_cm, 'o-', color='#2980b9', linewidth=2.0, markersize=7, label='Distancia acústica [cm]')
ax2.axhline(distancia_final_cm, color='#d35400', linestyle='--', linewidth=1.5, label=f'Distancia final ({distancia_final_cm:.2f} cm)')
ax2.fill_between(
    [0, max_x],
    distancia_final_cm - error_teorico_cm,
    distancia_final_cm + error_teorico_cm,
    color='#e67e22',
    alpha=0.2,
    label=f'Error teórico (±{error_teorico_cm:.2f} cm)'
)
ax2.set_ylim(0, max_d * 1.25 + 10)

# Evidenciar cada medición con su valor numérico en el Panel 2
for k, (t, d) in enumerate(zip(tiempos, distancias_cm)):
    ax2.annotate(
        f"{d:.2f} cm",
        xy=(t, d),
        xytext=(0, 12),
        textcoords="offset points",
        ha='center',
        fontsize=8.5,
        fontweight='bold',
        color='#1a5276',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#ebf5fb', edgecolor='#2980b9', alpha=0.95)
    )

ax2.set_title(f"2. Posición Acústica Medida Parlante - Micrófono (0 a {max_x} s)", fontsize=11, fontweight='bold')
ax2.set_xlabel("Tiempo de grabación [s] ", fontsize=10)
ax2.set_ylabel("Distancia [cm]", fontsize=10)
ax2.set_xlim(0, max_x)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
ax2.xaxis.set_minor_locator(ticker.MultipleLocator(1.0))
ax2.grid(True, which='major', color='#7f8c8d', linestyle='-', alpha=0.35)
ax2.grid(True, which='minor', color='#bdc3c7', linestyle=':', alpha=0.25)
ax2.legend(loc='upper left', framealpha=0.95)

plt.tight_layout()

# Guardar figura como evidencia gráfica para el usuario e informe
plt.savefig(ARCHIVO_EVIDENCIA, dpi=200)
print(f"Evidencia gráfica guardada en: {ARCHIVO_EVIDENCIA}")

plt.show()
