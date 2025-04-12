Intelligent Power & Security System for Server Rooms and Call Centers
Overview
This project presents a smart, AI-powered system designed to monitor and safeguard server rooms and call centers from environmental risks and unauthorized access. It offers intelligent fan control, anomaly detection, and real-time monitoring through a web-based dashboard. The system seamlessly integrates IoT, AI, and security functionalities.

System Architecture
1. Sensor and Security Layer
Temperature Sensor (DHT11)

Humidity Sensor

Dust Sensor (planned for future implementation)

Water Leakage Sensor

Smoke Sensor (analyzed via AI)

Motion Sensor (PIR)

Camera with facial recognition (DeepFace)

RFID Reader for authorized access

2. Microcontroller Layer
ESP32 and Arduino Uno

Interfaces with all sensors and security modules

Controls:

Fan with AI-based speed adjustment

LCD display for on-site data visualization

Publishes data to an MQTT broker

3. Communication Layer
MQTT Protocol

Lightweight and secure messaging

Topics include temperature, humidity, fan speed, alerts, and access logs

4. AI and Control Logic Layer
AI Model (Reinforcement Learning)
Predicts optimal fan speed based on environmental conditions

Fuzzy Logic Controller
Provides smooth and adaptive fan control

Anomaly Detection Module
Alerts for events such as overheating, intrusion, or water leakage

5. User Interface Layer
Web-based Dashboard

Real-time sensor data visualization

Access log review

System alerts and control panel

Key Features
Intelligent fan speed regulation using AI models

Integrated access control with RFID and facial recognition

Real-time anomaly detection for critical risks

Modular, scalable IoT architecture

Remote monitoring via a web dashboard connected through MQTT

Technologies Used
Hardware: ESP32, Arduino Uno, DHT11, PIR, RFID-RC522, camera module, smoke sensor

Software: Python, Arduino IDE, MQTT, Flask or Node.js, HTML/CSS/JavaScript

AI/ML: Reinforcement Learning, Fuzzy Logic, DeepFace for facial recognition

Development Tools: Visual Studio Code, Google Colab, PlatformIO, Mosquitto MQTT Broker

Planned Features (Future Work)
Due to time and hardware constraints, the following features are planned for future development:

Dust sensor integration (e.g., DSM501A)

Smart vacuum pump for automatic water drainage

Voltage monitoring (e.g., ZMPT101B and ACS712)

Mobile application for remote notifications

Power backup system and UPS integration

Custom-designed hardware enclosure for protection and portability

Future Enhancements
Improved decision-making using advanced machine learning algorithms

Voice or SMS-based alert system for critical notifications

Edge AI integration (e.g., Jetson Nano) for enhanced facial recognition

Full deployment on an IoT platform with mobile and cloud support

About Us
We are a dedicated team of developers with a passion for artificial intelligence, IoT, and embedded systems. Our mission is to create intelligent, efficient, and secure solutions tailored to high-risk environments such as server rooms and call centers.

