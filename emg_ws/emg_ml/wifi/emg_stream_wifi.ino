/*
 * emg_stream_wifi.ino — Continuous EMG streaming over WiFi/TCP (and optionally USB serial)
 *
 * Streams both EMG channels at exactly 2000 Hz as ASCII lines:  ch1,ch2\n
 * (raw 12-bit ADC counts 0-4095) — identical wire format to emg_stream.ino, so the
 * existing pyserial/readline() consumer works unchanged via wifi/tcp_receiver.py.
 *
 * Board: ESP32-S3-N16R8.  ESP32 = WiFi station + TCP SERVER on port 5005; the Pi
 * (or PC) connects as a TCP client. Pins GPIO3/GPIO8 are ADC1 -> safe with WiFi on.
 * Prints its DHCP IP over USB Serial on boot and advertises mDNS "esp32.local".
 */

#include <WiFi.h>
#include <ESPmDNS.h>

// ── EDIT THESE ────────────────────────────────────────────────────────────────
#define WIFI_SSID     "your_ssid"     // 2.4 GHz hotspot name
#define WIFI_PASSWORD "your_password" // hotspot password
// NOTE: ESP32-S3 is 2.4 GHz only. Use a 2.4 GHz SSID (not a 5 GHz-only one).

#define MDNS_HOSTNAME "esp32"                 // -> reachable as esp32.local
#define TCP_PORT      5005

// Also echo the stream over USB Serial in addition to WiFi? (de-risks bring-up).
#define ALSO_SERIAL   1

// ── Pin / ADC config ──────────────────────────────────────────────────────────
#define PIN_CH1   3       // ADC1 — safe with WiFi
#define PIN_CH2   8       // ADC1 — safe with WiFi
#define PIN_LED   2

// ── Sampling ──────────────────────────────────────────────────────────────────
#define FS        2000               // Hz
#define PERIOD_US (1000000 / FS)      // 500 µs per sample

// ── Serial (USB) ──────────────────────────────────────────────────────────────
#define BAUD_RATE 921600

// ── TCP batching ──────────────────────────────────────────────────────────────
#define FLUSH_MS  10                  // flush the batch every 10 ms (~20 lines)
#define TX_BUF_SZ 4096

// ── State ─────────────────────────────────────────────────────────────────────
WiFiServer tcpServer(TCP_PORT);
WiFiClient tcpClient;

static unsigned long nextSampleUs = 0;
static unsigned long lastFlushMs  = 0;
static uint32_t      sampleCount  = 0;

static char     txBuf[TX_BUF_SZ];
static size_t   txLen = 0;

static bool     wifiUp = false;

// ── WiFi bring-up (non-fatal: if it fails we still stream over Serial) ─────────
static void connectWiFi() {
  Serial.print("Connecting to WiFi \"");
  Serial.print(WIFI_SSID);
  Serial.print("\" ...");
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);               // disable modem sleep -> lower latency
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 15000) {
    delay(250);
    Serial.print('.');
  }

  if (WiFi.status() == WL_CONNECTED) {
    wifiUp = true;
    Serial.println(" connected.");
    Serial.print("WiFi IP: ");
    Serial.println(WiFi.localIP());
    Serial.print("RSSI = ");
    Serial.println(WiFi.RSSI());
    Serial.print("Hostname = ");
    Serial.println(MDNS_HOSTNAME);

    tcpServer.begin();
    tcpServer.setNoDelay(true);

    if (MDNS.begin(MDNS_HOSTNAME)) {
      MDNS.addService("emg", "tcp", TCP_PORT);
      Serial.print("mDNS responder started: ");
      Serial.print(MDNS_HOSTNAME);
      Serial.println(".local");
    } else {
      Serial.println("mDNS start FAILED (use the raw IP above instead).");
    }
    Serial.print("TCP server listening on port ");
    Serial.println(TCP_PORT);
  } else {
    wifiUp = false;
    Serial.println(" FAILED.");
    Serial.println("Continuing in Serial-only mode. Check SSID/password (2.4 GHz).");
  }
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(BAUD_RATE);
  pinMode(PIN_LED, OUTPUT);

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  analogSetPinAttenuation(PIN_CH1, ADC_11db);
  analogSetPinAttenuation(PIN_CH2, ADC_11db);

  for (int i = 0; i < 50; i++) {      // warm-up reads to settle the ADC
    analogRead(PIN_CH1);
    analogRead(PIN_CH2);
    delayMicroseconds(100);
  }

  connectWiFi();

  nextSampleUs = micros();
  lastFlushMs  = millis();
}

// Accept / refresh the single TCP client. Non-blocking.
static void serviceClient() {
  if (!tcpClient || !tcpClient.connected()) {
    WiFiClient incoming = tcpServer.available();
    if (incoming) {
      tcpClient = incoming;
      tcpClient.setNoDelay(true);
      Serial.println("TCP client connected.");
      txLen = 0;                       // drop any stale partial batch on reconnect
    }
  }
}

// Flush the accumulated batch to the TCP client WITHOUT blocking the sampler.
// If the socket can't take the whole batch right now, drop it and keep sampling.
static void flushBatch() {
  if (txLen == 0) return;

  // Write the whole batch to the TCP client. Do NOT gate on availableForWrite():
  // on several ESP32 cores it returns 0 even when the socket is healthy, which
  // silently drops every batch -> "connected but no data". write() only blocks
  // briefly while the client (the Pi) is reading, which it always is.
  if (wifiUp && tcpClient && tcpClient.connected()) {
    tcpClient.write((const uint8_t*)txBuf, txLen);
  }

#if ALSO_SERIAL
  Serial.write((const uint8_t*)txBuf, txLen);   // mirror batch to USB serial
#endif

  txLen = 0;
}

// ── Loop ──────────────────────────────────────────────────────────────────────
void loop() {
  // 1) Sample on the exact 500 µs schedule (network must never delay this).
  while ((long)(micros() - nextSampleUs) < 0) { /* busy wait */ }
  nextSampleUs += PERIOD_US;

  int ch1 = analogRead(PIN_CH1);
  int ch2 = analogRead(PIN_CH2);

  // 2) Append "ch1,ch2\n" to the batch buffer (skip if it would overflow).
  if (txLen + 16 < TX_BUF_SZ) {
    txLen += snprintf(txBuf + txLen, TX_BUF_SZ - txLen, "%d,%d\n", ch1, ch2);
  } else {
    txLen = 0;   // safety: should never fill if flushing works; reset if it does
  }

  sampleCount++;

  // 3) Periodically: service the client and flush. Cheap; keeps cadence.
  serviceClient();
  unsigned long nowMs = millis();
  if (nowMs - lastFlushMs >= FLUSH_MS) {
    lastFlushMs = nowMs;
    flushBatch();
  }

  // 4) LED heartbeat: ~2 Hz blink while waiting for a client, ~1 Hz once connected.
  if (wifiUp && (!tcpClient || !tcpClient.connected())) {
    if (sampleCount % (FS / 4) == 0) digitalWrite(PIN_LED, !digitalRead(PIN_LED));
  } else {
    if (sampleCount % FS == 0) digitalWrite(PIN_LED, !digitalRead(PIN_LED));
  }
}
