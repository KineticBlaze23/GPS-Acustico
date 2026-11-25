#include <Arduino.h>
#include <Wire.h>
#include <driver/i2s.h>

// --- PINES MICRÓFONO INMP441 (I2S) ---
#define I2S_WS   10
#define I2S_SCK  47
#define I2S_SD   21
#define I2S_PORT I2S_NUM_0

// --- PINES RELOJ EXTERNO DS3231 (TCXO) ---
#define DS_SDA      8
#define DS_SCL      9
#define DS_SQW      11
#define DS3231_ADDR 0x68

// --- PARÁMETROS DE MUESTREO Y PROTOCOLO ---
#define SAMPLE_RATE    40000
#define BUFFER_SAMPLES 512

const uint8_t HEADER[4] = {0xAA, 0x55, 0xAA, 0x55};
int32_t  i2s_raw_buf[BUFFER_SAMPLES];
uint16_t tx_buf[BUFFER_SAMPLES];

// Variables de sincronización con el pulso exacto de 1.000000 Hz del DS3231
volatile uint32_t ds3231_segundos = 0;
volatile uint64_t ds3231_ultimo_flanco_us = 0;
volatile uint64_t ds3231_periodo_real_us = 1000000ULL;

void IRAM_ATTR isr_ds3231_sqw() {
  uint64_t ahora = esp_timer_get_time();
  if (ds3231_segundos > 0) {
    ds3231_periodo_real_us = ahora - ds3231_ultimo_flanco_us;
  }
  ds3231_ultimo_flanco_us = ahora;
  ds3231_segundos++;
}

void configurar_ds3231_1hz() {
  Wire.begin(DS_SDA, DS_SCL);
  Wire.beginTransmission(DS3231_ADDR);
  Wire.write(0x0E);       // Registro de control
  Wire.write(0b00000000); // INTCN=0, RS2=0, RS1=0 -> Salida cuadrada 1 Hz en SQW
  Wire.endTransmission();

  pinMode(DS_SQW, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(DS_SQW), isr_ds3231_sqw, FALLING);
}

void configurar_i2s() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = BUFFER_SAMPLES,
    .use_apll = true,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };

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
  size_t bytes_leidos = 0;
  esp_err_t res = i2s_read(I2S_PORT, i2s_raw_buf, sizeof(i2s_raw_buf), &bytes_leidos, portMAX_DELAY);

  if (res == ESP_OK && bytes_leidos > 0) {
    uint16_t num_muestras = bytes_leidos / sizeof(int32_t);
    uint64_t ts_esp_us = esp_timer_get_time();

    // Disciplinar el timestamp interno usando el periodo medido contra el DS3231
    noInterrupts();
    uint32_t seg_ext = ds3231_segundos;
    uint64_t t_ref_us = ds3231_ultimo_flanco_us;
    uint64_t periodo_us = ds3231_periodo_real_us;
    interrupts();

    int64_t ts_corregido_us;
    if (seg_ext > 0 && periodo_us > 900000ULL && periodo_us < 1100000ULL) {
      double fraccion_seg = (double)(ts_esp_us - t_ref_us) / (double)periodo_us;
      ts_corregido_us = (int64_t)(seg_ext * 1000000ULL) + (int64_t)(fraccion_seg * 1000000.0);
    } else {
      ts_corregido_us = (int64_t)ts_esp_us;
    }

    for (uint16_t i = 0; i < num_muestras; i++) {
      int32_t val = (i2s_raw_buf[i] >> 16) + 2048;
      if (val < 0) val = 0;
      if (val > 4095) val = 4095;
      tx_buf[i] = (uint16_t)val;
    }

    // Enviar trama binaria compatible con receptor.py
    Serial.write(HEADER, 4);
    Serial.write((uint8_t*)&ts_corregido_us, sizeof(ts_corregido_us));
    Serial.write((uint8_t*)&num_muestras, sizeof(num_muestras));
    Serial.write((uint8_t*)tx_buf, num_muestras * sizeof(uint16_t));
  }
}