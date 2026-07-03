// ============================================================
//  adc_diagnostic.ino — ADC pin sanity check for ESP32-S3-N16R8
//
//  Reads a set of candidate ADC1 pins and prints raw ADC values
//  plus the converted voltage every 500 ms.
//
//  What to look for:
//    • Unconnected pin  : should read a small noisy value (~100–300),
//                         not a clean 0 or 4095. A stuck 0 on a pin
//                         means it is being driven (e.g. PSRAM bus).
//    • Grounded pin     : should read 0 (or near 0).
//    • 3.3 V applied    : should read ~4095.
//    • Sensor connected : should read 0–4095 tracking muscle activity.
//
//  If a pin always returns 0 regardless of what you connect, it is
//  occupied by internal hardware (flash/PSRAM) — do not use it.
//
//  On the ESP32-S3-N16R8:
//    GPIO33–37 are reserved for the 8 MB octal-SPI PSRAM.
//    Safe ADC1 pins: GPIO1–GPIO10 (all are also touch-capable,
//    but touch does not interfere with ADC).
// ============================================================

#include <Arduino.h>

// Pins to probe — includes the old broken choices and the new safe ones
// so you can see the difference side-by-side.
static constexpr int PROBE_PINS[] = {
    1, 2, 3, 4, 5, 6, 7,   // ADC1_CH0–CH6 (safe on N16R8)
    34, 35,                  // OLD choices — occupied by PSRAM, expect garbage
};
static constexpr int N_PINS = sizeof(PROBE_PINS) / sizeof(PROBE_PINS[0]);

// Multisampling: average N reads to reduce ADC noise
static constexpr int OVERSAMPLE = 16;

// ADC reference for voltage conversion (ESP32-S3 with ADC_11db → 3.9 V range)
static constexpr float ADC_VREF = 3.9f;
static constexpr int   ADC_MAX  = 4095;

static int oversampledRead(int pin) {
    long sum = 0;
    for (int i = 0; i < OVERSAMPLE; i++) sum += analogRead(pin);
    return (int)(sum / OVERSAMPLE);
}

void setup() {
    Serial.begin(115200);
    delay(1000);

    analogReadResolution(12);
    analogSetAttenuation(ADC_11db);  // 0–3.9 V range

    Serial.println("=== ESP32-S3-N16R8 ADC Diagnostic ===");
    Serial.println("Oversampling: " + String(OVERSAMPLE) + "x per reading");
    Serial.println("ADC range: 0–4095 (12-bit), 0–3.9 V (ADC_11db attn)");
    Serial.println();
    Serial.println("EXPECTED BEHAVIOUR:");
    Serial.println("  Floating pin    -> noisy ~100–600 (NOT a clean 0)");
    Serial.println("  GND connected   -> ~0");
    Serial.println("  3.3V connected  -> ~3464  (3.3/3.9 * 4095)");
    Serial.println("  PSRAM pins 34/35 -> stuck at 0 or garbage");
    Serial.println();

    // Print column header
    Serial.print("  ms   |");
    for (int i = 0; i < N_PINS; i++) {
        Serial.printf(" GPIO%-2d (raw/V)  |", PROBE_PINS[i]);
    }
    Serial.println();
}

void loop() {
    Serial.printf("%6lu |", millis());
    for (int i = 0; i < N_PINS; i++) {
        int raw = oversampledRead(PROBE_PINS[i]);
        float v = raw * ADC_VREF / ADC_MAX;
        Serial.printf(" %4d / %.2fV  |", raw, v);
    }
    Serial.println();
    delay(500);
}
