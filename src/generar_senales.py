from pathlib import Path
import numpy as np
from scipy.io.wavfile import write
from scipy.signal import chirp

# Definición de rutas basadas en la raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent
DIR_SENALES_REF = ROOT_DIR / "data" / "senales_referencia"
DIR_PROCESSED = ROOT_DIR / "data" / "processed"

DIR_SENALES_REF.mkdir(parents=True, exist_ok=True)
DIR_PROCESSED.mkdir(parents=True, exist_ok=True)

RUTA_PULSO = DIR_SENALES_REF / "senal_pulso.wav"
RUTA_CHIRP = DIR_SENALES_REF / "senal_chirp.wav"
RUTA_PLANTILLAS = DIR_PROCESSED / "plantillas_senales.npz"

FS_AUDIO = 44100        # Frecuencia estándar de salida de audio del PC/teléfono
DURACION_TOTAL = 1.0    # Duración de cada ciclo (1 emisión por segundo)
REPETICIONES = 35       # 35 emisiones (supera el mínimo de 30 emisiones por punto)
RETARDO_CANALES = 0.25  # 250 ms entre el parlante izquierdo (L) y el derecho (R)

t_ciclo = np.linspace(0, DURACION_TOTAL, int(FS_AUDIO * DURACION_TOTAL), endpoint=False)

# --- SEÑAL 1: PULSO CORTO (3 kHz, 5 ms) ---
dur_pulso = 0.005
t_p = np.linspace(0, dur_pulso, int(FS_AUDIO * dur_pulso), endpoint=False)
ventana_p = np.hanning(len(t_p))
pulso = 0.9 * np.sin(2 * np.pi * 3000 * t_p) * ventana_p

canal_L_pulso = np.zeros_like(t_ciclo)
canal_R_pulso = np.zeros_like(t_ciclo)

idx_L = int(0.05 * FS_AUDIO)                     # Izquierdo suena a los 0.05 s
idx_R = int((0.05 + RETARDO_CANALES) * FS_AUDIO) # Derecho suena a los 0.30 s

canal_L_pulso[idx_L:idx_L + len(pulso)] = pulso
canal_R_pulso[idx_R:idx_R + len(pulso)] = pulso

estereo_pulso = np.column_stack((
    np.tile(canal_L_pulso, REPETICIONES),
    np.tile(canal_R_pulso, REPETICIONES)
))
write(str(RUTA_PULSO), FS_AUDIO, (estereo_pulso * 32767).astype(np.int16))

# --- SEÑAL 2: BARRIDO CHIRP (1.5 kHz a 6 kHz, 20 ms) ---
dur_chirp = 0.020
t_c = np.linspace(0, dur_chirp, int(FS_AUDIO * dur_chirp), endpoint=False)
ventana_c = np.hanning(len(t_c))
onda_chirp = 0.9 * chirp(t_c, f0=1500, f1=6000, t1=dur_chirp, method='linear') * ventana_c

canal_L_chirp = np.zeros_like(t_ciclo)
canal_R_chirp = np.zeros_like(t_ciclo)

canal_L_chirp[idx_L:idx_L + len(onda_chirp)] = onda_chirp
canal_R_chirp[idx_R:idx_R + len(onda_chirp)] = onda_chirp

estereo_chirp = np.column_stack((
    np.tile(canal_L_chirp, REPETICIONES),
    np.tile(canal_R_chirp, REPETICIONES)
))
write(str(RUTA_CHIRP), FS_AUDIO, (estereo_chirp * 32767).astype(np.int16))

# Guardamos también la plantilla del chirp para usarla en la detección por correlación
np.savez(RUTA_PLANTILLAS, fs_audio=FS_AUDIO, chirp=onda_chirp, pulso=pulso, retardo_canales=RETARDO_CANALES)
print(f"Archivos generados con éxito:\n  - {RUTA_PULSO}\n  - {RUTA_CHIRP}\n  - {RUTA_PLANTILLAS}")
