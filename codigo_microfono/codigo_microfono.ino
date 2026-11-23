#include <Arduino.h>
#include <driver/i2s.h>
#include "esp_timer.h"

// Pines GPIO impresos en la placa ESP32-S3
#define I2S_WS            10  // LRCLK (impreso como 10 en la placa)
#define I2S_SCK           47  // BCLK  (impreso como 47 en la placa)
#define I2S_SD            21  // DOUT  (impreso como 21 en la placa)

#define I2S_PORT          I2S_NUM_0
#define SAMPLE_RATE_HZ    40000
#define SAMPLES_PER_FRAME 256

int32_t bufferI2S[SAMPLES_PER_FRAME];

#pragma pack(push, 1)
struct PaqueteAudio {
  uint8_t header[4];
  int64_t timestamp_us;
  uint16_t num_muestras;
  uint16_t muestras[SAMPLES_PER_FRAME];
};
#pragma pack(pop)

PaqueteAudio paquete;

void configurarI2S() {
  const i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE_HZ,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = SAMPLES_PER_FRAME,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  const i2s_pin_config_t pin_config = {
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
  paquete.header[0] = 0xAA;
  paquete.header[1] = 0x55;
  paquete.header[2] = 0xAA;
  paquete.header[3] = 0x55;

  Serial.begin(2000000);
  delay(1000);
  configurarI2S();
}

void loop() {
  size_t bytesLeidos = 0;

  if (i2s_read(I2S_PORT, &bufferI2S, sizeof(bufferI2S), &bytesLeidos, portMAX_DELAY) == ESP_OK) {
    paquete.timestamp_us = esp_timer_get_time();
    uint32_t n = bytesLeidos / sizeof(int32_t);
    if (n > SAMPLES_PER_FRAME) n = SAMPLES_PER_FRAME;
    paquete.num_muestras = (uint16_t)n;

    for (uint32_t i = 0; i < n; i++) {
      // Escala los 24 bits con signo del INMP441 al rango 0..4095 (centro 2048)
      int32_t muestra = bufferI2S[i] >> 14;
      int32_t escalado = muestra + 2048;
      if (escalado < 0) escalado = 0;
      if (escalado > 4095) escalado = 4095;
      paquete.muestras[i] = (uint16_t)escalado;
    }

    size_t bytesAEnviar = 4 + 8 + 2 + (n * sizeof(uint16_t));
    Serial.write((uint8_t *)&paquete, bytesAEnviar);
  }
}