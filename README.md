# GPS Acústico - Posicionamiento Acústico 1D

Sistema de estimación de posición unidimensional basado en señales acústicas, desarrollado para la materia de Métodos Numéricos. El sistema utiliza correlación cruzada, detección de flanco de subida, interpolación sub-muestra y compensación por mínimos cuadrados para estimar desplazamientos a partir del tiempo de vuelo (ToF) del sonido.

## Estructura del Proyecto

```text
GPS Acustico/
├── .gitignore
├── README.md
├── codigo_microfono/
│   └── codigo_microfono.ino      # Firmware para ESP32-S3 (lectura I2S y transmisión serial)
├── data/
│   ├── processed/                # Señales procesadas y plantillas de correlación (.npz)
│   ├── raw/                      # Grabaciones crudas en formato .npz y .wav (ignorado en git)
│   └── senales_referencia/       # Audio generado para reproducción (.wav)
├── src/
│   ├── generar_senales.py        # Generación de señales chirp y pulsos de prueba
│   ├── procesar_posicion.py      # Algoritmo de correlación, ToF y estimación de distancia
│   └── receptor.py               # Captura de datos vía puerto serial desde el ESP32-S3
└── tests/
    ├── prueba_audio_mic.py       # Prueba rápida de captura y reproducción de audio
    └── prueba_mic.wav            # Audio grabado de prueba
```

## Requisitos de Hardware

- Placa de desarrollo ESP32-S3.
- Micrófono digital I2S INMP441.
  - Conexiones por defecto (GPIO):
    - SCK (BCLK) -> GPIO 47
    - WS (LRCLK) -> GPIO 10
    - SD (DOUT)  -> GPIO 21
    - L/R        -> GND (Canal Izquierdo)
    - VDD        -> 3.3V
    - GND        -> GND
- Parlantes o altavoces estéreo conectados al emisor de audio.

## Requisitos de Software

- Python 3.8 o superior.
- Librerías necesarias:
  - numpy
  - scipy
  - matplotlib
  - pyserial

Instalación de dependencias:
```bash
pip install numpy scipy matplotlib pyserial
```

## Flujo de Trabajo

1. **Generación de señales de referencia:**
   Ejecutar `src/generar_senales.py` para generar los archivos de audio estéreo (`data/senales_referencia/senal_chirp.wav` y `senal_pulso.wav`) junto con la plantilla de correlación (`data/processed/plantillas_senales.npz`).
   ```bash
   python src/generar_senales.py
   ```

2. **Carga del firmware en ESP32-S3:**
   Abrir `codigo_microfono/codigo_microfono.ino` en Arduino IDE o ESP-IDF, compilar y cargar en la placa. Asegurarse de cerrar el monitor serial de Arduino antes de iniciar los scripts en Python.

3. **Captura de señal:**
   Configurar el puerto COM adecuado en `src/receptor.py` y ejecutar para capturar las muestras del micrófono durante la reproducción del audio:
   ```bash
   python src/receptor.py
   ```
   Los datos se guardan en `data/raw/grabacion_cruda.npz`.

4. **Procesamiento y estimación de posición:**
   Ejecutar `src/procesar_posicion.py` para procesar la grabación cruda, detectar los frentes de onda y graficar la posición estimada:
   ```bash
   python src/procesar_posicion.py
   ```
