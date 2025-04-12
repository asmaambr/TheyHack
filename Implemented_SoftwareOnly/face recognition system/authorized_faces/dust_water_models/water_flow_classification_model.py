import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Generate Data
np.random.seed(42)
water_leak = np.random.randint(0, 2, 1000)
water_alert = [1 if w == 1 else 0 for w in water_leak]
data = pd.DataFrame({'water_leak': water_leak, 'water_alert': water_alert})

# Train/Test Split
X = data[['water_leak']]
y = data['water_alert']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("📊 Water Leak Model Report:\n")
print(classification_report(y_test, y_pred))

# Test New Data
new_data = pd.DataFrame({'water_leak': [0]})
prediction = model.predict(new_data)[0]
print(f"\n🔍 New Input => Water Leak: 0")
print("🚨 Water Leak Detected" if prediction == 1 else "✅ No Water Leak")

# Save Model
joblib.dump(model, 'water_model.pkl')
print("\n✅ Water model saved as 'water_model.pkl'")
