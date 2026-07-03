// ============================================================
//  sEMG Data Collection  —  ESP32-S3-N16R8 + 2× Muscle Sensor v3
//  RP06: sEMG-Controlled Robotic Arm (Salma Farid)
// ============================================================
//
//  Muscle Sensor v3 (Advancer Technologies) — what the sensor
//  actually outputs:
//    Each sensor is a three-lead DIFFERENTIAL sensor. It has 3
//    electrode connections (RED, BLUE, BLACK) but only ONE analog
//    output (SIG pin). Internally it performs:
//      1. Differential amplification of the two active electrodes
//      2. Bandpass filtering (passes EMG frequency content)
//      3. Full-wave rectification  →  signal is ALWAYS ≥ 0
//      4. Smoothing/integration    →  amplitude envelope
//    The SIG pin therefore outputs the rectified+smoothed EMG
//    envelope (0 V to +Vs), NOT raw bipolar EMG.
//
//  ── PIN SELECTION (ESP32-S3-N16R8 specific) ──────────────────────
//  The N16R8 variant has 8 MB of octal-SPI PSRAM. Espressif reserves
//  GPIO33–GPIO37 for the PSRAM bus (SPIIO4–SPIIO7 + SPIDQS).
//  GPIO34 and GPIO35 are NOT free — reading them gives garbage.
//
//  Use ADC1 only (GPIO1–GPIO10). ADC2 (GPIO11–GPIO20) shares hardware
//  with Wi-Fi and is unreliable. All ADC1 pins on the S3 are also
//  touch-capable, but touch does NOT interfere unless touchRead() is
//  called explicitly. GPIO3 and GPIO8 are chosen here: clear of boot
//  pins (0,2,3), USB (19,20), UART0 (43,44), and PSRAM (33–37).
//
//  WIRING — Sensor 1 (forearm EXTENSOR side, GPIO3)
//    RED   cable → electrode on forearm extensor muscle belly center
//    BLUE  cable → electrode on forearm extensor muscle belly end
//    BLACK cable → reference electrode on bony landmark (e.g. elbow)
//    SIG   pin   → voltage divider → GPIO3  (ADC1_CH3)
//    GND   pin   → ESP32-S3 GND
//    +Vs   pin   → +9 V battery positive terminal
//    -Vs   pin   → -9 V battery negative terminal
//    PWR GND pin → junction between the two 9 V batteries
//
//  WIRING — Sensor 2 (forearm FLEXOR side, GPIO8)
//    RED   cable → electrode on forearm flexor muscle belly center
//    BLUE  cable → electrode on forearm flexor muscle belly end
//    BLACK cable → reference electrode on bony landmark (wrist)
//    SIG   pin   → voltage divider → GPIO8  (ADC1_CH4)
//    GND   pin   → ESP32-S3 GND
//    +Vs / -Vs / PWR GND → same dual 9 V supply as Sensor 1
//
//  Voltage divider (each sensor's SIG):
//    Vout = Vin × 22 kΩ / (56 kΩ + 22 kΩ) = 0.282 × Vin
//    Max output: 9 V × 0.282 = 2.54 V  →  safe for 3.3 V ESP32-S3 ✓
//
//  Protocol (mirrors procedure.pptx):
//    5 cycles × 10 gestures × 6 s recording = 50 trials/participant
//    Operator advances slide deck then presses button (or sends any
//    Serial byte) to start each 6-second recording window.
//
//  Serial output (921600 baud) — CSV, one row per sample:
//    participant_id,cycle,gesture_name,gesture_id,trial,sample_idx,ch1,ch2
//    ch1 = Sensor 1 raw ADC value (0–4095), ch2 = Sensor 2
// ============================================================

#include <Arduino.h>

// ── Configuration ────────────────────────────────────────────────
static constexpr int    SAMPLE_RATE_HZ    = 2000;
static constexpr int    SAMPLE_PERIOD_US  = 1000000 / SAMPLE_RATE_HZ; // 500 µs
static constexpr int    SAMPLES_PER_TRIAL = SAMPLE_RATE_HZ * 6;       // 12 000
static constexpr int    NUM_CYCLES        = 5;
static constexpr int    NUM_GESTURES      = 10;

// ADC1 pins — GPIO3 and GPIO8 are safe on the N16R8 board
// (GPIO33–37 are reserved for the octal PSRAM; do not use them)
static constexpr int    CH1_PIN    = 3;    // ADC1_CH2 — Sensor 1 (extensor)
static constexpr int    CH2_PIN    = 8;    // ADC1_CH7 — Sensor 2 (flexor)
static constexpr int    LED_PIN    = 2;    // Onboard LED
static constexpr int    BTN_PIN    = 0;    // Boot button (active LOW)

// ── Gesture manifest (order matches procedure.pptx) ──────────────
// Label IDs are fixed — must match Python pipeline constants.
static const char* GESTURE_NAMES[NUM_GESTURES] = {
    "REST",       // 0
    "EXTENSION",  // 1
    "FLEXION",    // 2
    "ULNAR_DEV",  // 3
    "RADIAL_DEV", // 4
    "GRIP",       // 5
    "ABDUCTION",  // 6
    "ADDUCTION",  // 7
    "SUPINATION", // 8
    "PRONATION"   // 9
};

// ── Sample buffer: 12 000 samples × 2 channels = 48 KB ───────────
// Sampling and Serial streaming are separated: we fill the buffer
// at exact timing, then stream it after the trial is complete so
// Serial I/O never interferes with the 500 µs sampling loop.
static uint16_t sampleBuf[SAMPLES_PER_TRIAL][2];

// ── State ────────────────────────────────────────────────────────
static int participantID = 0;

// ─────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────
static void blinkLED(int n, int halfPeriodMs) {
    for (int i = 0; i < n; i++) {
        digitalWrite(LED_PIN, HIGH); delay(halfPeriodMs);
        digitalWrite(LED_PIN, LOW);  delay(halfPeriodMs);
    }
}

// Block until boot button (active LOW) or any Serial byte arrives.
static void waitForTrigger() {
    while (digitalRead(BTN_PIN) == HIGH && !Serial.available()) delay(5);
    while (Serial.available()) Serial.read(); // flush any queued bytes
}

// 8× oversampling: sample each channel 8 times per output sample and average.
// Effective output rate stays at 2000 Hz; each read group takes ~62.5 µs,
// well within the 500 µs period. Gains ~1.5 ENOB vs single reads.
static constexpr int    OVERSAMPLE    = 8;
static constexpr int    OS_PERIOD_US  = SAMPLE_PERIOD_US / OVERSAMPLE;  // ~62 µs

static void collectTrial() {
    unsigned long nextSampleTime = micros();
    for (int i = 0; i < SAMPLES_PER_TRIAL; i++) {
        while ((long)(micros() - nextSampleTime) < 0) {}  // spin-wait
        uint32_t acc1 = 0, acc2 = 0;
        for (int k = 0; k < OVERSAMPLE; k++) {
            acc1 += analogRead(CH1_PIN);
            acc2 += analogRead(CH2_PIN);
        }
        sampleBuf[i][0] = (uint16_t)(acc1 / OVERSAMPLE);
        sampleBuf[i][1] = (uint16_t)(acc2 / OVERSAMPLE);
        nextSampleTime += SAMPLE_PERIOD_US;
    }
}

// Stream buffered trial as CSV rows over Serial after collection.
// ~360 KB of text; sent post-trial to keep sampling timing clean.
static void streamTrial(int cycle, int gestureID, int trial) {
    for (int i = 0; i < SAMPLES_PER_TRIAL; i++) {
        Serial.printf("%d,%d,%s,%d,%d,%d,%u,%u\n",
            participantID, cycle,
            GESTURE_NAMES[gestureID], gestureID,
            trial, i,
            sampleBuf[i][0], sampleBuf[i][1]);
    }
}

// ─────────────────────────────────────────────────────────────────
// Setup
// ─────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(921600);
    pinMode(LED_PIN, OUTPUT);
    pinMode(BTN_PIN, INPUT_PULLUP);

    // 12-bit ADC, 11 dB attenuation → input range 0–3.9 V
    // Sensor SIG through 56k/22k divider → max 2.54 V  ✓
    analogReadResolution(12);
    analogSetAttenuation(ADC_11db);

    digitalWrite(LED_PIN, LOW);
    delay(500);

    Serial.println(F("=== sEMG Data Collection System ==="));
    Serial.println(F("Board  : ESP32"));
    Serial.println(F("Sensors: 2× Muscle Sensor v3 (Advancer Technologies)"));
    Serial.println(F("Output : rectified+smoothed EMG envelope per sensor"));
    Serial.printf ("Rate   : %d Hz  |  %d samples/trial (6 s)\n",
                   SAMPLE_RATE_HZ, SAMPLES_PER_TRIAL);
    Serial.printf ("Trials : %d cycles × %d gestures = %d total\n",
                   NUM_CYCLES, NUM_GESTURES, NUM_CYCLES * NUM_GESTURES);
    Serial.println(F("---"));
    Serial.println(F("Enter participant ID then press ENTER:"));

    while (!Serial.available()) delay(10);
    participantID = Serial.parseInt();
    while (Serial.available()) Serial.read();

    Serial.printf("Participant ID set to: %d\n", participantID);
    Serial.println(F("---"));
    Serial.println(F("WIRING CHECKLIST (each sensor has 3 leads + 1 SIG output):"));
    Serial.println(F("  Sensor 1 (extensor):  RED+BLUE+BLACK on forearm extensors"));
    Serial.println(F("    SIG → 56k/22k divider → GPIO3  (ADC1_CH3)  |  GND → ESP32-S3 GND"));
    Serial.println(F("  Sensor 2 (flexor):    RED+BLUE+BLACK on forearm flexors"));
    Serial.println(F("    SIG → 56k/22k divider → GPIO8  (ADC1_CH4)  |  GND → ESP32-S3 GND"));
    Serial.println(F("  Power: +9V→+Vs, midpoint→PWR_GND, -9V→-Vs (both sensors share supply)"));
    Serial.println(F("---"));
    Serial.println(F("participant_id,cycle,gesture_name,gesture_id,trial,sample_idx,ch1,ch2"));
    Serial.println(F("Press button or send any key to begin first trial..."));
    waitForTrigger();
    blinkLED(3, 150);
}

// ─────────────────────────────────────────────────────────────────
// Main loop
// ─────────────────────────────────────────────────────────────────
void loop() {
    for (int cycle = 1; cycle <= NUM_CYCLES; cycle++) {
        Serial.printf("### CYCLE %d / %d  (P%d) ###\n",
                      cycle, NUM_CYCLES, participantID);
        blinkLED(cycle, 100);

        for (int g = 0; g < NUM_GESTURES; g++) {
            // ── Prompt operator to cue participant ────────────────
            Serial.printf("READY: %s  [cycle %d, gesture %d/%d]\n",
                          GESTURE_NAMES[g], cycle, g + 1, NUM_GESTURES);
            Serial.println(F("  Advance slide deck to matching gesture, then press button/key."));
            blinkLED(2, 80);
            waitForTrigger();

            // ── LED ON for entire 6-second recording window ───────
            digitalWrite(LED_PIN, HIGH);
            collectTrial();
            digitalWrite(LED_PIN, LOW);

            // ── Stream buffered data after trial ──────────────────
            Serial.printf("DATA_START P%d C%d G%s T%d\n",
                          participantID, cycle, GESTURE_NAMES[g], cycle);
            streamTrial(cycle, g, cycle);
            Serial.println(F("DATA_END"));

            Serial.printf("OK: %s  C%d complete.\n", GESTURE_NAMES[g], cycle);
        }

        Serial.printf("### CYCLE %d COMPLETE ###\n\n", cycle);
        blinkLED(5, 100);
        delay(1000);
    }

    Serial.printf("=== ALL %d CYCLES COMPLETE  P%d ===\n",
                  NUM_CYCLES, participantID);
    Serial.println(F("Data collection finished. You may disconnect."));
    digitalWrite(LED_PIN, HIGH); // solid LED = all done
    while (true) delay(1000);
}
