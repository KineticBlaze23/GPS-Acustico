from pathlib import Path
import numpy as np
from scipy.io.wavfile import write

# Definición de rutas basadas en la raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent
DIR_SENALES_REF = ROOT_DIR / "data" / "senales_referencia"
DIR_PROCESSED = ROOT_DIR / "data" / "processed"

DIR_SENALES_REF.mkdir(parents=True, exist_ok=True)
DIR_PROCESSED.mkdir(parents=True, exist_ok=True)

RUTA_PULSO = DIR_SENALES_REF / "senal_pulso.wav"
RUTA_PLANTILLAS = DIR_PROCESSED / "plantillas_senales.npz"

# --- PARÁMETROS DEL PULSO ACÚSTICO DE ALTA POTENCIA ---
FS_AUDIO = 44100            # Frecuencia estándar de salida de audio del PC/teléfono (Hz)
FREC_TONO = 2500            # Frecuencia óptima de resonancia para parlantes portátiles/PC (2.5 kHz)
DURACION_TOTAL = 2.0        # Espaciado de 2 segundos entre cada chasquido (periodo)
DURACION_PULSO = 0.020      # 20 ms de duración (aporta 75x más energía acústica para mover el parlante)
REPETICIONES = 30           # 30 emisiones (60 segundos totales de audio)
AMPLITUD_MAX = 0.82         # 82% de amplitud (~26,868 de 32,767): evita disparar el limitador/compresor automático (DRC/AGC) del parlante

t_ciclo = np.linspace(0, DURACION_TOTAL, int(FS_AUDIO * DURACION_TOTAL), endpoint=False)

# Generación del pulso percusivo de alta energía:
# - Ataque rápido (1.0 ms) para mantener el frente de onda abrupto e inequívoco
# - Cuerpo sostenido a potencia máxima (RMS ~0.67) para excitar con fuerza la membrana del parlante
# - Caída suave (2.0 ms) para evitar chasquidos residuales al terminar
t_p = np.linspace(0, DURACION_PULSO, int(FS_AUDIO * DURACION_PULSO), endpoint=False)
env = np.ones_like(t_p)
n_ataque = int(0.001 * FS_AUDIO)
n_caida = int(0.002 * FS_AUDIO)
env[:n_ataque] = np.sin(0.5 * np.pi * np.linspace(0, 1, n_ataque))**2
env[-n_caida:] = np.cos(0.5 * np.pi * np.linspace(0, 1, n_caida))**2

# Onda acústica escalada al 99.9% de amplitud (volumen máximo sin distorsión)
onda = AMPLITUD_MAX * np.sin(2 * np.pi * FREC_TONO * t_p) * env

# Canal mono con el chasquido ubicado a los 0.05 s de cada ciclo de 2 segundos
canal = np.zeros_like(t_ciclo)
idx_inicio = int(0.05 * FS_AUDIO)
canal[idx_inicio:idx_inicio + len(onda)] = onda

# Audio estéreo unificado al volumen máximo (ambos canales reproducen simultáneamente)
estereo_audio = np.column_stack((
    np.tile(canal, REPETICIONES),
    np.tile(canal, REPETICIONES)
))
write(str(RUTA_PULSO), FS_AUDIO, (estereo_audio * 32767).astype(np.int16))

# Guardamos la plantilla para la correlación cruzada en procesar_posicion.py
np.savez(
    RUTA_PLANTILLAS,
    fs_audio=FS_AUDIO,
    pulso=onda,
    chirp=onda,               # Alias de compatibilidad hacia atrás
    periodo=DURACION_TOTAL,
    frec_tono=FREC_TONO,
    duracion_pulso=DURACION_PULSO
)

print("=" * 65)
print("   SEÑAL PERCUSIVA DE ALTA POTENCIA GENERADA CON ÉXITO")
print("=" * 65)
print(f"Archivo de audio:       {RUTA_PULSO}")
print(f"Tipo de señal:          Chasquido percusivo de alto volumen ('¡CLAC!')")
print(f"Frecuencia acústica:    {FREC_TONO} Hz (frecuencia de alta eficiencia en parlantes)")
print(f"Duración del pulso:     {DURACION_PULSO * 1000:.1f} ms (75x más energía acústica audible)")
print(f"Amplitud máxima:        {AMPLITUD_MAX * 100:.1f}% de la escala int16 ({int(AMPLITUD_MAX * 32767)} / 32767)")
print(f"Intervalo entre pulsos: {DURACION_TOTAL:.1f} s")
print(f"Repeticiones:           {REPETICIONES} ({REPETICIONES * DURACION_TOTAL:.0f} s totales)")
print(f"Plantilla guardada:     {RUTA_PLANTILLAS}")
print("=" * 65)
