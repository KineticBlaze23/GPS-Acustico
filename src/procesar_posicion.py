from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar
from scipy.signal import butter, filtfilt, fftconvolve, find_peaks, hilbert, resample_poly

# ----------------------------------------------------------------------------
# RUTAS (basadas en la raíz del proyecto)
# ----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DIR_PROCESSED = ROOT_DIR / "data" / "processed"
DIR_RAW = ROOT_DIR / "data" / "raw"
DIR_PROCESSED.mkdir(parents=True, exist_ok=True)

ARCHIVO_GRABACION = DIR_RAW / "grabacion_cruda.npz"
ARCHIVO_PLANTILLAS = DIR_PROCESSED / "plantillas_senales.npz"
ARCHIVO_EVIDENCIA = DIR_PROCESSED / "evidencia_posicion_acustica.png"

# ----------------------------------------------------------------------------
# PARÁMETROS (mismos que validar_posiciones.py para que coincidan al 100 %)
# ----------------------------------------------------------------------------
TEMP_C = 20.0               # Temperatura ambiente [°C]
N_QUIETO = 5                # Pulsos iniciales con el micrófono INMÓVIL (calibración de reloj)
FS_NOMINAL = 40000          # Frecuencia nominal del firmware [Hz]
PERIODO_WAV = 2.0           # Periodo entre emisiones [s] (se lee de la plantilla)
C_SONIDO = 331.3 + 0.606 * TEMP_C

# ----------------------------------------------------------------------------
# 1. CARGAR GRABACIÓN Y FILTRAR RUIDO DE BAJA FRECUENCIA
# ----------------------------------------------------------------------------
if not ARCHIVO_GRABACION.exists():
    raise SystemExit(f"ERROR: No se encontró {ARCHIVO_GRABACION}")

datos = np.load(ARCHIVO_GRABACION)
if 'fs_real' not in datos:
    raise SystemExit("ERROR: la grabación no trae 'fs_real'.")
voltaje = datos['voltaje']
fs_real = float(datos['fs_real'])

senal_cruda = voltaje - np.mean(voltaje)
tiempo_s = np.arange(len(senal_cruda)) / fs_real

# Filtro pasa-banda (1.5 kHz - 6.5 kHz): elimina el ruido lento que tapaba P18-P29
nyq = 0.5 * fs_real
b_filt, a_filt = butter(4, [1500.0 / nyq, 6500.0 / nyq], btype='band')
senal = filtfilt(b_filt, a_filt, senal_cruda)

# ----------------------------------------------------------------------------
# 2. CARGAR PLANTILLA Y REMUESTREAR CON FILTRO ANTI-ALIAS
# ----------------------------------------------------------------------------
if not ARCHIVO_PLANTILLAS.exists():
    raise SystemExit(f"ERROR: No se encontró {ARCHIVO_PLANTILLAS}")

plantillas = np.load(ARCHIVO_PLANTILLAS)
fs_audio = float(plantillas['fs_audio'])
if 'periodo' in plantillas:
    PERIODO_WAV = float(plantillas['periodo'])

r = Fraction(int(round(FS_NOMINAL)), int(round(fs_audio))).limit_denominator(2000)
plantilla = resample_poly(plantillas['pulso'], r.numerator, r.denominator)
plantilla = filtfilt(b_filt, a_filt, plantilla)  # Mismo filtrado a la plantilla
plantilla = plantilla / np.linalg.norm(plantilla)

# ----------------------------------------------------------------------------
# 3. FILTRO ADAPTADO + DETECCIÓN INICIAL EN REPOSO (N_QUIETO)
# ----------------------------------------------------------------------------
corr = np.abs(hilbert(fftconvolve(senal, plantilla[::-1], mode='valid')))
piso_ruido = np.median(corr)
umbral_ruido = 6.0 * piso_ruido
dist_min_muestras = int(0.70 * PERIODO_WAV * fs_real)

# Detectar los primeros pulsos fuertes en reposo para calibrar el paso exacto
indices_init, _ = find_peaks(corr, height=max(umbral_ruido, 0.25 * np.max(corr)),
                             distance=dist_min_muestras)

if len(indices_init) < N_QUIETO:
    raise SystemExit("ERROR: no se detectaron suficientes pulsos iniciales en reposo.")

def estimar_pico_spline(y, idx_max, fs):
    ini = max(0, idx_max - 4)
    fin = min(len(y), idx_max + 5)
    ventana = np.arange(ini, fin)
    t_v = ventana / fs
    cs = CubicSpline(t_v, y[ventana])
    try:
        res = root_scalar(cs.derivative(), bracket=[t_v[1], t_v[-2]], method='brentq')
        return res.root
    except ValueError:
        return idx_max / fs

# Calibrar periodo exacto del emisor (a) y origen (b) con los N_QUIETO primeros pulsos
t_cal = np.array([estimar_pico_spline(corr, p, fs_real) for p in indices_init[:N_QUIETO]])
c_cal = np.round((t_cal - t_cal[0]) / PERIODO_WAV).astype(int)
a, b = np.polyfit(c_cal, t_cal, 1)

# ----------------------------------------------------------------------------
# 4. RASTREO FÍSICO CONTINUO (BLOQUEA ECOS LEJANOS COMO EL DE +226 CM)
# ----------------------------------------------------------------------------
paso_muestras = a * fs_real
ventana_tracking = int(0.0045 * fs_real)  # ±4.5 ms (±154 cm máx entre pulsos, excluye el eco de +6.6 ms)
radio_frente = int(0.003 * fs_real)       # 3 ms hacia atrás para tomar el primer frente directo

indices_picos = list(indices_init[:N_QUIETO])
ciclos = list(c_cal)

ciclo_actual = ciclos[-1]
idx_actual = indices_picos[-1]

while True:
    ciclo_siguiente = ciclo_actual + 1
    centro = int(round(idx_actual + paso_muestras))
    ini = centro - ventana_tracking
    fin = centro + ventana_tracking
    if fin >= len(corr):
        break

    seg = corr[ini:fin]
    max_local = np.max(seg)

    if max_local >= umbral_ruido:
        # Buscar el máximo local dentro de la ventana física y priorizar su borde delantero
        idx_max_loc = ini + int(np.argmax(seg))
        ini_f = max(ini, idx_max_loc - radio_frente)
        sub_picos, _ = find_peaks(corr[ini_f:idx_max_loc + 1], height=max(umbral_ruido, 0.35 * max_local))
        idx_encontrado = ini_f + sub_picos[0] if len(sub_picos) > 0 else idx_max_loc

        indices_picos.append(idx_encontrado)
        ciclos.append(ciclo_siguiente)
        idx_actual = idx_encontrado

    ciclo_actual = ciclo_siguiente

indices_picos = np.array(indices_picos)
ciclos = np.array(ciclos)

# ----------------------------------------------------------------------------
# 5. TIEMPOS SUB-MUESTRA Y DESPLAZAMIENTO RESPECTO AL ORIGEN
# ----------------------------------------------------------------------------
tiempos = np.array([estimar_pico_spline(corr, p, fs_real) for p in indices_picos])
delta_t = tiempos - (a * ciclos + b)
distancias_cm = delta_t * C_SONIDO * 100.0

n_cal = N_QUIETO
error_teorico_cm = float((C_SONIDO / (2.0 * fs_real)) * 100.0)
deriva_ppm = (a / PERIODO_WAV - 1.0) * 1e6
# ----------------------------------------------------------------------------
# 6. REPORTE EN CONSOLA
# ----------------------------------------------------------------------------
print("\n" + "=" * 58)
print("          MEDICIÓN DE POSICIÓN ACÚSTICA (1D)")
print("=" * 58)
print(f"fs_real (DS3231):     {fs_real:.2f} Hz | c = {C_SONIDO:.1f} m/s")
print(f"Deriva del emisor:    {deriva_ppm:+.1f} ppm (calibrado con {n_cal} pulsos)")
print(f"Límite teórico:       ±{error_teorico_cm:.2f} cm\n")

print(f"{'Pulso':<8}{'Ciclo':<8}{'Tiempo (s)':<14}{'Posición (cm)':<14}")
print("-" * 44)
for k in range(len(tiempos)):
    tag = " (cal)" if k < n_cal else ""
    print(f"{k+1:<8}{ciclos[k]:<8}{tiempos[k]:<14.4f}{distancias_cm[k]:<14.2f}{tag}")
print("-" * 44)
print(f"Posición final medida: {distancias_cm[-1]:.2f} cm")
print("=" * 58 + "\n")

# ----------------------------------------------------------------------------
# 7. GRÁFICAS DE EVIDENCIA
# ----------------------------------------------------------------------------
max_x = max(20, int(np.ceil(tiempo_s[-1])))
y_max = float(np.max(np.abs(senal)))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

# Panel 1: Onda filtrada y pulsos detectados
ax1.plot(tiempo_s, senal, color='#2c3e50', linewidth=0.6, label='Audio filtrado (1.5 - 6.5 kHz)')
for k, (p, t) in enumerate(zip(indices_picos, tiempos)):
    ini_v = max(0, p - int(0.01 * fs_real))
    fin_v = min(len(senal), p + int(0.01 * fs_real))
    pico_vis = senal[ini_v + np.argmax(np.abs(senal[ini_v:fin_v]))]
    ax1.plot(t, pico_vis, 'ro', markersize=5)
    ax1.annotate(f"P{k+1}", xy=(t, pico_vis), xytext=(0, 8), textcoords="offset points",
                 ha='center', fontsize=8, fontweight='bold', color='#900C3F')

ax1.set_ylim(-1.3 * y_max, 1.3 * y_max)
ax1.set_title(f"1. Señal acústica filtrada y detecciones (0 a {max_x} s @ {fs_real:.1f} Hz)",
              fontsize=11, fontweight='bold')
ax1.set_ylabel("Amplitud filtrada")
ax1.set_xlim(0, max_x)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', framealpha=0.95)

# Panel 2: Posición vs Tiempo
ax2.axvspan(0, tiempos[n_cal - 1] + 0.5 * PERIODO_WAV, color='#27ae60', alpha=0.12,
            label=f'Zona de calibración ({n_cal} pulsos quietos)')
ax2.axhspan(-error_teorico_cm, error_teorico_cm, color='#e67e22', alpha=0.20,
            label=f'Límite teórico (±{error_teorico_cm:.2f} cm)')
ax2.axhline(0, color='#7f8c8d', linewidth=1.0)
ax2.plot(tiempos, distancias_cm, 'o-', color='#2980b9', linewidth=2.0, markersize=7,
         label='Posición acústica medida [cm]')

for t, d in zip(tiempos, distancias_cm):
    ax2.annotate(f"{d:.1f}", xy=(t, d), xytext=(0, 10), textcoords="offset points",
                 ha='center', fontsize=8.5, fontweight='bold', color='#1a5276',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#ebf5fb', edgecolor='#2980b9', alpha=0.95))

margen = max(10.0, 0.2 * np.max(np.abs(distancias_cm)))
ax2.set_ylim(min(distancias_cm.min(), -error_teorico_cm) - margen,
             max(distancias_cm.max(), error_teorico_cm) + margen)
ax2.set_title("2. Desplazamiento del micrófono respecto al origen (+ = se aleja)",
              fontsize=11, fontweight='bold')
ax2.set_xlabel("Tiempo de grabación [s]")
ax2.set_ylabel("Distancia [cm]")
ax2.set_xlim(0, max_x)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
ax2.xaxis.set_minor_locator(ticker.MultipleLocator(1.0))
ax2.grid(True, which='major', color='#7f8c8d', linestyle='-', alpha=0.35)
ax2.grid(True, which='minor', color='#bdc3c7', linestyle=':', alpha=0.25)
ax2.legend(loc='upper left', framealpha=0.95)

plt.tight_layout()
plt.savefig(ARCHIVO_EVIDENCIA, dpi=200)
print(f"Evidencia gráfica guardada en: {ARCHIVO_EVIDENCIA}")
plt.show()