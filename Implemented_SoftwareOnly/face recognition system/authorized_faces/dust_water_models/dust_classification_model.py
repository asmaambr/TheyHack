import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Generate Data
np.random.seed(42)
dust = np.random.randint(0, 101, 1000)
dust_alert = [1 if d > 70 else 0 for d in dust]
data = pd.DataFrame({'dust': dust, 'dust_alert': dust_alert})

# Train/Test Split
X = data[['dust']]
y = data['dust_alert']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("📊 Dust Model Report:\n")
print(classification_report(y_test, y_pred))

# Test New Data
new_data = pd.DataFrame({'dust': [85]})
prediction = model.predict(new_data)[0]
print(f"\n🔍 New Input => Dust: 85")
print("🚨 Dust Alert" if prediction == 1 else "✅ Dust Level Safe")

# Save Model
joblib.dump(model, 'dust_model.pkl')
print("\n✅ Dust model saved as 'dust_model.pkl'")
