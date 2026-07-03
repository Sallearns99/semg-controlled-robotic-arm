/*
 * emg_stream.ino — Continuous EMG streaming for real-time inference
 *
 * Unlike emg_collection.ino (which buffers 12 000 samples then dumps them),
 * this sketch streams both channels continuously at exactly 2000 Hz.
 *
 * Output format over Serial (921600 baud):
 *   ch1,ch2\n
 * where ch1/ch2 are raw 12-bit ADC counts (0–4095).
 *
 * inference_rt.py reads this stream and maintains its own sliding window.
 *
 * Pins (same as emg_collection.ino):
 *   GPIO4  — Sensor 1 SIG  (extensor, ADC1_CH3)
 *   GPIO5  — Sensor 2 SIG  (flexor,   ADC1_CH4)
 *   GPIO2  — Onboard LED   (blinks at 1 Hz to confirm streaming)
 *
 * WiFi streaming (future):
 *   Uncomment the WiFi section below and fill in SSID/password.
 *   The sketch will open a TCP server on port 5005; inference_rt.py
 *   connects to it instead of Serial.
 */

#include "driver/adc.h"

// ── Pin / ADC config ──────────────────────────────────────────────
// Must match emg_collection.ino wiring: Sensor 1 → GPIO3, Sensor 2 → GPIO8
#define PIN_CH1   3
#define PIN_CH2   8
#define PIN_LED   2

// ── Sampling ──────────────────────────────────────────────────────
#define FS        2000          // Hz
#define PERIOD_US (1000000 / FS)  // 500 µs per sample

// ── Serial ────────────────────────────────────────────────────────
#define BAUD_RATE 921600

// ── Optional WiFi (uncomment to enable) ──────────────────────────
// #define USE_WIFI
// #define WIFI_SSID     "your_ssid"
// #define WIFI_PASSWORD "your_password"
// #define TCP_PORT      5005

#ifdef USE_WIFI
  #include <WiFi.h>
  WiFiServer tcpServer(TCP_PORT);
  WiFiClient tcpClient;
#endif

// ── State ─────────────────────────────────────────────────────────
static unsigned long nextSampleUs = 0;
static uint32_t      sampleCount  = 0;

// ── Setup ─────────────────────────────────────────────────────────
void setup() {
  Serial.begin(BAUD_RATE);
  pinMode(PIN_LED, OUTPUT);

  // ADC config: 12-bit, 11 dB attenuation (0–3.9 V range)
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  analogSetPinAttenuation(PIN_CH1, ADC_11db);
  analogSetPinAttenuation(PIN_CH2, ADC_11db);

  // Warm-up reads to settle the ADC
  for (int i = 0; i < 50; i++) {
    analogRead(PIN_CH1);
    analogRead(PIN_CH2);
    delayMicroseconds(100);
  }

#ifdef USE_WIFI
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) { delay(500); }
  tcpServer.begin();
  // Send IP over Serial so you can see it
  Serial.print("WiFi IP: ");
  Serial.println(WiFi.localIP());
#endif

  nextSampleUs = micros();
}

// ── Loop ──────────────────────────────────────────────────────────
void loop() {
  // Spin-wait for exact 500 µs interval
  while ((long)(micros() - nextSampleUs) < 0) { /* busy wait */ }
  nextSampleUs += PERIOD_US;

  int ch1 = analogRead(PIN_CH1);
  int ch2 = analogRead(PIN_CH2);

  // Build output line in a small buffer to minimise Serial overhead
  char buf[20];
  int  len = snprintf(buf, sizeof(buf), "%d,%d\n", ch1, ch2);

#ifdef USE_WIFI
  if (!tcpClient || !tcpClient.connected()) {
    tcpClient = tcpServer.available();
  }
  if (tcpClient && tcpClient.connected()) {
    tcpClient.write((uint8_t*)buf, len);
  }
#else
  Serial.write(buf, len);
#endif

  sampleCount++;

  // Blink LED every 2000 samples (≈1 Hz)
  if (sampleCount % FS == 0) {
    digitalWrite(PIN_LED, !digitalRead(PIN_LED));
  }
}
