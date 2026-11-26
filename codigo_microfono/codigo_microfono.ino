/*
 * Adquisición acústica: INMP441 (I2S) + DS3231 (pulso SQW de 1 Hz) -> PC por serie.
 *
 * Trama enviada por bloque (little-endian, 30 bytes de cabecera + datos):
 *   sync[4]    AA 55 AA 55
 *   seq        uint32  contador de bloque (detecta bloques perdidos al transmitir)
 *   ts_esp_us  int64   esp_timer al terminar i2s_read
 *   sqw_n      uint32  nº de flancos del DS3231 vistos hasta ahora
 *   sqw_us     int64   esp_timer en el último flanco del DS3231
 *   n          uint16  nº de muestras del bloque
 *   datos      n x int16 (muestras con signo)
 * En Python: struct '<4sIqIqH'. La disciplina del reloj se hace en receptor.py.
 *
 * Recomendado: Arduino-ESP32 core 2.0.x (driver I2S "legacy" driver/i2s.h).
 * No uses Serial.print de depuración: corrompería el flujo binario.
 */

#include <Arduino.h>
#include <Wire.h>
#include <driver/i2s.h>
#include <esp_timer.h>

// --- PINES MICRÓFONO INMP441 (I2S) ---
#define I2S_WS   10
#define I2S_SCK  47
#define I2S_SD   21
#define I2S_PORT I2S_NUM_0

// --- PINES DS3231 (TCXO) ---
#define DS_SDA      8
#define DS_SCL      9
#define DS_SQW      11
#define DS3231_ADDR 0x68

// --- PARÁMETROS DE MUESTREO Y PROTOCOLO ---
#define SAMPLE_RATE    40000
#define BUFFER_SAMPLES 512
#define SHIFT_BITS     16     // 32 bits -> int16. Baja a 14 o 12 si la señal sale muy débil.

#pragma pack(push, 1)
struct Cabecera {
  uint8_t  sync[4];
  uint32_t seq;
  int64_t  ts_esp_us;
  uint32_t sqw_n;
  int64_t  sqw_us;
  uint16_t n;
};
#pragma pack(pop)
static_assert(sizeof(Cabecera) == 30, "La cabecera debe medir 30 bytes (receptor.py usa '<4sIqIqH')");

int32_t i2s_raw_buf[BUFFER_SAMPLES];
int16_t tx_buf[BUFFER_SAMPLES];
uint32_t seq = 0;

// --- Sincronización con el pulso de 1 Hz del DS3231 ---
portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;
volatile uint32_t sqw_count = 0;
volatile int64_t  sqw_us = 0;

void IRAM_ATTR isr_ds3231_sqw() {
  int64_t ahora = esp_timer_get_time();
  portENTER_CRITICAL_ISR(&mux);
  sqw_us = ahora;
  sqw_count++;
  portEXIT_CRITICAL_ISR(&mux);
}

void configurar_ds3231_1hz() {
  Wire.begin(DS_SDA, DS_SCL);
  Wire.beginTransmission(DS3231_ADDR);
  Wire.write(0x0E);        // Registro de control
  Wire.write(0b00000000);  // INTCN=0, RS2=0, RS1=0 -> onda cuadrada de 1 Hz en SQW
  Wire.endTransmission();

  pinMode(DS_SQW, INPUT_PULLUP);  // SQW es de drenador abierto: ideal añadir pull-up externo de 4.7 kΩ
  attachInterrupt(digitalPinToInterrupt(DS_SQW), isr_ds3231_sqw, FALLING);
}

void configurar_i2s() {
  i2s_config_t i2s_config = {};
  i2s_config.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
  i2s_config.sample_rate = SAMPLE_RATE;
  i2s_config.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
  i2s_config.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;   // INMP441 con L/R a GND
  i2s_config.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  i2s_config.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  i2s_config.dma_buf_count = 16;
  i2s_config.dma_buf_len = BUFFER_SAMPLES;
  i2s_config.use_apll = false;       // el ESP32-S3 no tiene APLL
  i2s_config.tx_desc_auto_clear = false;
  i2s_config.fixed_mclk = 0;

  i2s_pin_config_t pin_config = {};
  pin_config.mck_io_num = I2S_PIN_NO_CHANGE;
  pin_config.bck_io_num = I2S_SCK;
  pin_config.ws_io_num = I2S_WS;
  pin_config.data_out_num = I2S_PIN_NO_CHANGE;
  pin_config.data_in_num = I2S_SD;

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);
  i2s_zero_dma_buffer(I2S_PORT);
}

void setup() {
  Serial.begin(2000000);
  configurar_ds3231_1hz();
  configurar_i2s();
}

void loop() {
  size_t bytes = 0;
  esp_err_t res = i2s_read(I2S_PORT, i2s_raw_buf, sizeof(i2s_raw_buf), &bytes, portMAX_DELAY);
  if (res != ESP_OK || bytes == 0) return;

  int64_t ts = esp_timer_get_time();
  uint16_t n = bytes / sizeof(int32_t);

  Cabecera h;
  h.sync[0] = 0xAA; h.sync[1] = 0x55; h.sync[2] = 0xAA; h.sync[3] = 0x55;
  h.seq = seq++;
  h.ts_esp_us = ts;
  h.n = n;

  portENTER_CRITICAL(&mux);
  h.sqw_n = sqw_count;
  h.sqw_us = sqw_us;
  portEXIT_CRITICAL(&mux);

  for (uint16_t i = 0; i < n; i++) {
    int32_t v = i2s_raw_buf[i] >> SHIFT_BITS;
    if (v > 32767) v = 32767;
    if (v < -32768) v = -32768;
    tx_buf[i] = (int16_t)v;
  }

  Serial.write((uint8_t*)&h, sizeof(h));
  Serial.write((uint8_t*)tx_buf, n * sizeof(int16_t));
}