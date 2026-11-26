from pathlib import Path
import numpy as np
from scipy.io.wavfile import write
from scipy.signal import chirp, fftconvolve, hilbert
from scipy.signal.windows import tukey

# Definición de rutas basadas en la raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent
DIR_SENALES_REF = ROOT_DIR / "data" / "senales_referencia"
DIR_PROCESSED = ROOT_DIR / "data" / "processed"

DIR_SENALES_REF.mkdir(parents=True, exist_ok=True)
DIR_PROCESSED.mkdir(parents=True, exist_ok=True)

RUTA_PULSO = DIR_SENALES_REF / "senal_pulso.wav"
RUTA_PLANTILLAS = DIR_PROCESSED / "plantillas_senales.npz"

# --- PARÁMETROS DEL PULSO ACÚSTICO (CHIRP LINEAL) ---
FS_AUDIO = 44100            # Frecuencia de salida de audio del PC/teléfono (Hz)
F0 = 1500                   # Frecuencia inicial del chirp (Hz)
F1 = 8000                   # Frecuencia final del chirp (Hz)
DURACION_TOTAL = 2.0        # Periodo entre emisiones (s)
DURACION_PULSO = 0.080      # Duración del chirp (s). Más largo = más energía y más robusto con Bluetooth
REPETICIONES = 30           # 30 emisiones = 60 s de audio
AMPLITUD_MAX = 0.82         # Evita disparar el limitador/compresor del parlante
INICIO_PULSO = 0.05         # El chirp empieza a los 0.05 s de cada ciclo
ALFA_TUKEY = 0.2            # Fracción de la ventana usada en suavizado de bordes

# --- RELLENO ANTI-SUSPENSIÓN (parlantes Bluetooth) ---
# Muchos parlantes Bluetooth se duermen o bajan la ganancia con el silencio digital.
# Un tono continuo débil en agudos (casi inaudible para adultos) los mantiene activos; está lejos
# de la banda del chirp (1.5-8 kHz), así que el filtro adaptado lo ignora casi por completo.
# Debe quedar por debajo de 20 kHz (límite de Nyquist del micrófono a 40 kHz).
# Si el parlante aún se duerme, sube la amplitud poco a poco (0.05 -> 0.08 -> 0.12).
# Si se oye, súbele la frecuencia (hasta ~19500 Hz) o bájale la amplitud.
# Pon AMPLITUD_RELLENO = 0 para desactivarlo.
FREC_RELLENO = 18500        # Hz (2 s * 18500 Hz = ciclos enteros: sin saltos de fase al repetir)
AMPLITUD_RELLENO = 0.05     # ~ -26 dBFS

# --- GENERACIÓN DEL CHIRP ---
# Barrido lineal de F0 a F1 con ventana de Tukey: bordes suaves (sin chasquidos)
# y espectro ancho, lo que da un pico de correlación muy estrecho (~1/ancho de banda).
n_pulso = int(round(FS_AUDIO * DURACION_PULSO))
t_p = np.arange(n_pulso) / FS_AUDIO
onda = AMPLITUD_MAX * chirp(t_p, f0=F0, t1=DURACION_PULSO, f1=F1, method='linear') \
       * tukey(n_pulso, alpha=ALFA_TUKEY)

# --- CICLO DE 2 s: SILENCIO + CHIRP + SILENCIO ---
n_ciclo = int(round(FS_AUDIO * DURACION_TOTAL))
t_ciclo = np.arange(n_ciclo) / FS_AUDIO
canal = AMPLITUD_RELLENO * np.sin(2 * np.pi * FREC_RELLENO * t_ciclo)
idx_inicio = int(round(INICIO_PULSO * FS_AUDIO))
canal[idx_inicio:idx_inicio + n_pulso] += onda

# El ciclo contiene un número entero de periodos del relleno (2 s * 18500 Hz = 37000),
# así que al repetirlo no hay saltos de fase.
# Audio estéreo (ambos canales iguales) repetido REPETICIONES veces
estereo_audio = np.column_stack((
    np.tile(canal, REPETICIONES),
    np.tile(canal, REPETICIONES)
))
write(str(RUTA_PULSO), FS_AUDIO, (estereo_audio * 32767).astype(np.int16))

# --- PLANTILLA PARA LA CORRELACIÓN CRUZADA (procesar_posicion.py) ---
np.savez(
    RUTA_PLANTILLAS,
    fs_audio=FS_AUDIO,
    pulso=onda,
    chirp=onda,               # Alias de compatibilidad
    periodo=DURACION_TOTAL,
    f0=F0,
    f1=F1,
    duracion_pulso=DURACION_PULSO
)

# --- VERIFICACIÓN: ANCHO DEL PICO DE AUTOCORRELACIÓN ---
# Cuanto más estrecho, más precisa es la medición del tiempo de llegada.
ac = np.abs(hilbert(fftconvolve(onda, onda[::-1], mode='full')))
ac /= ac.max()
ancho_muestras = np.sum(ac >= 0.5)
ancho_ms = ancho_muestras / FS_AUDIO * 1000.0
ancho_cm = ancho_ms / 1000.0 * 343.0 * 100.0

print("=" * 65)
print("        SEÑAL CHIRP GENERADA CON ÉXITO")
print("=" * 65)
print(f"Archivo de audio:       {RUTA_PULSO}")
print(f"Tipo de señal:          Chirp lineal {F0}-{F1} Hz con ventana de Tukey")
print(f"Duración del pulso:     {DURACION_PULSO * 1000:.1f} ms")
print(f"Relleno anti-suspensión: {FREC_RELLENO} Hz a {AMPLITUD_RELLENO * 100:.1f} % de la escala" if AMPLITUD_RELLENO > 0 else "Relleno anti-suspensión: desactivado")
print(f"Amplitud máxima:        {AMPLITUD_MAX * 100:.1f}% de la escala int16 ({int(AMPLITUD_MAX * 32767)} / 32767)")
print(f"Intervalo entre pulsos: {DURACION_TOTAL:.1f} s")
print(f"Repeticiones:           {REPETICIONES} ({REPETICIONES * DURACION_TOTAL:.0f} s totales)")
print(f"Ancho del pico (-6 dB): {ancho_ms:.3f} ms (~{ancho_cm:.1f} cm)")
print(f"Plantilla guardada:     {RUTA_PLANTILLAS}")
print("=" * 65)