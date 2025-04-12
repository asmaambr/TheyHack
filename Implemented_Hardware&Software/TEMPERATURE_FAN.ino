#include <WiFi.h>                    // WiFi library for ESP32
#include <WiFiClientSecure.h>       // For secure MQTT (TLS)
#include <PubSubClient.h>           // MQTT client library
#include "DHT.h"                    // DHT sensor library
#include <Wire.h>                   // I2C communication library
#include <LiquidCrystal_I2C.h>      // LCD I2C display library

// ----- LCD SETUP -----
LiquidCrystal_I2C lcd(0x27, 16, 2);  // LCD I2C address, 16 columns and 2 rows

// ----- WiFi Credentials -----
const char* ssid = "Rahil brk";           // Your WiFi SSID
const char* password = "rahil545083";     // Your WiFi password

// ----- MQTT Configuration -----
const char* mqtt_server = "6a26a25b84f745cd9ab64971ced603b0.s1.eu.hivemq.cloud";
const int mqtt_port = 8883;               // MQTT over TLS port
const char* mqtt_user = "Heck22";         // MQTT username
const char* mqtt_password = "Theyhack2025"; // MQTT password

// ----- MQTT Topics -----
const char* temp_topic = "device/temperature"; // Topic to publish temperature
const char* hum_topic = "device/humidity";     // Topic to publish humidity
const char* fan_topic = "device/fan";          // Topic to publish fan status

// ----- DHT Sensor Setup -----
#define DHTPIN 42             // GPIO pin connected to DHT11 data pin
#define DHTTYPE DHT11         // Type of DHT sensor
DHT dht(DHTPIN, DHTTYPE);     // Create DHT object

// ----- Relay Setup (Fan Control) -----
#define RELAY_FAN_PIN 37      // GPIO pin connected to relay module
                              // Relay controls the fan

// ----- MQTT Client Setup -----
WiFiClientSecure espClient;
PubSubClient client(espClient);

// ----- WiFi Connection -----
void setup_wifi() {
  Serial.print("Connecting to WiFi");
  WiFi.begin(ssid, password);      // Begin WiFi connection

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println(" Connected!");   // Print once connected
}

// ----- MQTT Reconnect -----
void reconnect() {
  while (!client.connected()) {
    Serial.print("Connecting to MQTT...");
    if (client.connect("ESP32S3Client", mqtt_user, mqtt_password)) {
      Serial.println(" Connected to MQTT.");
    } else {
      Serial.print(" failed, rc=");
      Serial.print(client.state());
      Serial.println(" retrying in 5 seconds...");
      delay(5000); // Wait 5 seconds before retrying
    }
  }
}

// ----- Setup Function -----
void setup() {
  Serial.begin(9600);                    // Start serial communication for debug

  pinMode(RELAY_FAN_PIN, OUTPUT);        // Set fan relay pin as output
  digitalWrite(RELAY_FAN_PIN, LOW);     // Turn fan ON initially for 5s
  delay(5000);
  digitalWrite(RELAY_FAN_PIN, HIGH);      // Then turn fan OFF

  dht.begin();                           // Initialize DHT sensor
  setup_wifi();                          // Connect to WiFi

  espClient.setInsecure();               // Disable certificate checking for testing
  client.setServer(mqtt_server, mqtt_port); // Set MQTT broker

  Wire.begin(21, 20);                    // Set I2C SDA and SCL pins for LCD

  lcd.init();                            // Initialize LCD
  lcd.backlight();                       // Turn on LCD backlight
  lcd.setCursor(0, 0);
  lcd.print("System Starting");          // Show startup message
  delay(2000);
  lcd.clear();                           // Clear display
}

// ----- Main Loop -----
void loop() {
  if (!client.connected()) {
    reconnect();                         // Ensure MQTT is connected
  }
  client.loop();                         // Handle MQTT background tasks

  float temperature = dht.readTemperature(); // Read temperature
  float humidity = dht.readHumidity();       // Read humidity

  // ----- If sensor readings are valid -----
  if (!isnan(temperature) && !isnan(humidity)) {
    char temp_str[6];
    dtostrf(temperature, 4, 1, temp_str);  // Format temp as string

    char hum_str[6];
    dtostrf(humidity, 4, 1, hum_str);      // Format humidity as string

    client.publish(temp_topic, temp_str);  // Publish temperature to MQTT
    client.publish(hum_topic, hum_str);    // Publish humidity to MQTT

    // ----- Fan Control Logic -----
    if (temperature >= 30.0) {
      digitalWrite(RELAY_FAN_PIN, HIGH);   // Turn ON fan
      client.publish(fan_topic, "ON");
      Serial.println("Fan ON (>= 30°C)");
    } else {
      digitalWrite(RELAY_FAN_PIN, LOW);    // Turn OFF fan
      client.publish(fan_topic, "OFF");
      Serial.println("Fan OFF (< 30°C)");
    }

    // ----- Update LCD Display -----
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("Temp: ");
    lcd.print(temp_str);
    lcd.print((char)223); // Degree symbol
    lcd.print("C");

    lcd.setCursor(0, 1);
    lcd.print("Humidity: ");
    lcd.print(hum_str);
    lcd.print("%");

    // ----- Print to Serial Monitor -----
    Serial.print("Temperature: ");
    Serial.print(temp_str);
    Serial.print("°C, Humidity: ");
    Serial.print(hum_str);
    Serial.println("%");

  } else {
    // ----- Error Handling -----
    Serial.println("DHT sensor error.");
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("Sensor Error");
  }

  delay(1000); // Wait 1 second before next loop
}