# Intelligent Power & Security System for Server Rooms and Call Centers

##Overview
This project is a smart, AI-powered system designed to monitor and protect server rooms and call centers from environmental hazards and unauthorized access. It intelligently controls fan speed, detects anomalies, and presents real-time data on a web dashboard — combining IoT, AI, and security features.

---

## System Architecture

### 1. Sensor & Security Layer
-  Temperature Sensor (DHT11)
-  Humidity Sensor
-  Dust Sensor *(future feature)*
-  Water Leakage Sensor
-  Smoke Sensor (AI-analyzed)
-  Motion Sensor (PIR)
-  Camera with DeepFace for facial recognition
-  RFID Reader (authorized access)

### 2. Microcontroller Layer
-  ESP32 + Arduino Uno
- Reads sensor/security inputs
- Controls:
  - Fan (with AI-based speed control)
  - LCD Display
- Publishes data to MQTT Broker

### 3. Communication Layer
-  MQTT Protocol
  - Secure, lightweight messaging
  - Topics: temperature, humidity, fan speed, alerts, access logs, etc.

### 4. AI + Control Logic Layer
-  AI Model (Reinforcement Learning)  
  Predicts optimal fan speed based on environmental data
-  Fuzzy Logic Controller  
  Smooth, adaptive fan control
-  Anomaly Detection Model  
  Triggers alerts for overheating, intrusion, or water leakage

### 5. User Interface Layer
-  Web Dashboard
  - Real-time sensor data
  - Access logs
  - Alerts and system control

---

## Key Features
-  AI-controlled fan speed (not just ON/OFF)
-  RFID & Camera (DeepFace) for access control
-  Real-time anomaly detection
-  Web-based dashboard via MQTT
-  Modular and scalable IoT architecture

---

##  Technologies Used
- Hardware: ESP32, Arduino Uno, DHT11, PIR, RFID-RC522, camera, smoke sensor
- Software: Python, Arduino IDE, MQTT, Flask (or Node.js), HTML/CSS/JS
- AI/ML: Reinforcement Learning, Fuzzy Logic, DeepFace for facial recognition
- Tools: VS Code, Google Colab, PlatformIO, Mosquitto Broker

---

##  Unimplemented (Future) Features
Due to time and component limitations:
-  Dust Sensor integration (e.g., DSM501A)
-  Smart Vacuum Pump for water removal
-  Voltage Monitoring (e.g., ZMPT101B + ACS712)
-  Mobile App for remote alerts
-  Power backup and UPS integration
-  Custom hardware enclosure


##  Future Improvements
- Integrate machine learning for smarter decision-making
- Add voice or SMS alert system
- Use edge AI boards (e.g., Jetson Nano) for better face recognition
- Full IoT platform deployment with mobile and cloud

---

##  About Us
We are a passionate team of developers interested in AI, IoT, and embedded systems. Our goal is to build smart, efficient, and secure systems tailored for high-risk environments like server rooms and call centers.

---
