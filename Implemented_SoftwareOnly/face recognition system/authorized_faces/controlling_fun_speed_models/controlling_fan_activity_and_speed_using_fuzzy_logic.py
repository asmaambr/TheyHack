import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from collections import deque
import matplotlib.pyplot as plt

# 1. Define fuzzy variables
temperature = ctrl.Antecedent(np.arange(15, 46, 1), 'temperature')
humidity = ctrl.Antecedent(np.arange(10, 101, 1), 'humidity')
fan_speed = ctrl.Consequent(np.arange(0, 101, 1), 'fan_speed')

# 2. Define membership functions
temperature['low'] = fuzz.trimf(temperature.universe, [15, 15, 25])
temperature['medium'] = fuzz.trimf(temperature.universe, [20, 30, 35])
temperature['high'] = fuzz.trimf(temperature.universe, [30, 45, 45])

humidity['low'] = fuzz.trimf(humidity.universe, [10, 10, 40])
humidity['medium'] = fuzz.trimf(humidity.universe, [30, 55, 70])
humidity['high'] = fuzz.trimf(humidity.universe, [60, 100, 100])

fan_speed['low'] = fuzz.trimf(fan_speed.universe, [0, 0, 40])
fan_speed['medium'] = fuzz.trimf(fan_speed.universe, [30, 50, 70])
fan_speed['high'] = fuzz.trimf(fan_speed.universe, [60, 100, 100])

# 3. Define fuzzy rules
rules = [
    ctrl.Rule(temperature['low'] & humidity['low'], fan_speed['low']),
    ctrl.Rule(temperature['low'] & humidity['medium'], fan_speed['low']),
    ctrl.Rule(temperature['low'] & humidity['high'], fan_speed['medium']),
    
    ctrl.Rule(temperature['medium'] & humidity['low'], fan_speed['low']),
    ctrl.Rule(temperature['medium'] & humidity['medium'], fan_speed['medium']),
    ctrl.Rule(temperature['medium'] & humidity['high'], fan_speed['high']),
    
    ctrl.Rule(temperature['high'] & humidity['low'], fan_speed['medium']),
    ctrl.Rule(temperature['high'] & humidity['medium'], fan_speed['high']),
    ctrl.Rule(temperature['high'] & humidity['high'], fan_speed['high']),
]

# 4. Control system
fan_ctrl = ctrl.ControlSystem(rules)
fan_simulation = ctrl.ControlSystemSimulation(fan_ctrl)

# 5. Anomaly detection using rolling history
class SensorHistory:
    def __init__(self, maxlen=10):
        self.temp_history = deque(maxlen=maxlen)
        self.hum_history = deque(maxlen=maxlen)

    def update(self, temp, hum):
        self.temp_history.append(temp)
        self.hum_history.append(hum)

    def detect_anomaly(self):
        if len(self.temp_history) < 5:
            return False
        temp_mean = np.mean(self.temp_history)
        temp_std = np.std(self.temp_history)
        hum_mean = np.mean(self.hum_history)
        hum_std = np.std(self.hum_history)
        latest_temp = self.temp_history[-1]
        latest_hum = self.hum_history[-1]
        return (
            abs(latest_temp - temp_mean) > 2 * temp_std or
            abs(latest_hum - hum_mean) > 2 * hum_std
        )

# 6. Fan control system
def fuzzy_fan_controller(temp, hum):
    fan_simulation.input['temperature'] = temp
    fan_simulation.input['humidity'] = hum
    fan_simulation.compute()
    return fan_simulation.output['fan_speed']

# 7. Main simulation
def main():
    history = SensorHistory(maxlen=10)
    temperatures = [25, 26, 26.5, 27, 27.5, 35, 40, 38, 39, 20]
    humidities = [40, 42, 43, 44, 45, 70, 75, 72, 71, 30]
    outputs = []

    for temp, hum in zip(temperatures, humidities):
        history.update(temp, hum)
        anomaly = history.detect_anomaly()
        if anomaly:
            print(f"⚠️  Anomaly Detected! Temp: {temp}, Hum: {hum}")
        speed = fuzzy_fan_controller(temp, hum)
        outputs.append(speed)
        print(f"Temp: {temp}°C | Humidity: {hum}% -> Fan Speed: {speed:.2f}%")

    # Plotting results
    plt.plot(outputs, marker='o', label='Fan Speed (%)')
    plt.title("Fuzzy Fan Speed Control")
    plt.xlabel("Reading Index")
    plt.ylabel("Fan Speed (%)")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
