import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, average_precision_score, silhouette_score
from sklearn.model_selection import ParameterGrid, train_test_split, KFold, cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
import seaborn as sns
from tqdm import tqdm

# Load and preprocess the data
print("Loading and preprocessing data...")
data = pd.read_csv("sensor_data_rows.csv")

# Drop missing values
data = data.dropna(subset=["temperature", "humidity", "recorded_at"])

# Convert timestamp to datetime
data["recorded_at"] = pd.to_datetime(data["recorded_at"], format='mixed')

# Sort by time
data = data.sort_values("recorded_at").reset_index(drop=True)

# Calculate temperature difference
data["temp_diff"] = data["temperature"].diff().abs()
data["time_diff"] = data["recorded_at"].diff().dt.total_seconds() / 60  # time diff in minutes

# Fill first row's NaNs (diffs) with 0
data[["temp_diff", "time_diff"]] = data[["temp_diff", "time_diff"]].fillna(0)

# Mark anomaly if temp change > 3°C in 5 min
data["anomaly_label"] = ((data["temp_diff"] > 3) & (data["time_diff"] <= 10)).astype(int)

# Features for clustering
features = data[["temperature", "humidity", "temp_diff"]].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# === HYPERPARAMETER TUNING FOR DBSCAN ===
print("Performing hyperparameter tuning for DBSCAN...")

# Define parameter grid
param_grid = {
    'eps': [0.3, 0.5, 0.8, 1.0, 1.2, 1.5],
    'min_samples': [3, 5, 10, 15, 20]
}

# Initialize variables to store best parameters
best_score = -1
best_params = {}
best_anomaly_count = 0
results = []

# Random search (sample from parameter grid)
import random
param_combinations = list(ParameterGrid(param_grid))
random.shuffle(param_combinations)
sampled_combinations = param_combinations[:10]  # Take 10 random combinations

for params in tqdm(sampled_combinations):
    # Apply DBSCAN with current parameters
    dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
    cluster_labels = dbscan.fit_predict(X_scaled)
    
    # Skip if all points are labeled as noise (-1)
    if np.all(cluster_labels == -1):
        continue
    
    # Calculate silhouette score if there's more than one cluster
    try:
        # Filter out noise points for silhouette score
        mask = cluster_labels != -1
        if np.unique(cluster_labels[mask]).size > 1:
            score = silhouette_score(X_scaled[mask], cluster_labels[mask])
        else:
            score = -1
    except:
        score = -1
    
    # Mark anomalies
    anomaly_labels = (cluster_labels == -1).astype(int)
    anomaly_count = np.sum(anomaly_labels)
    
    # Store results
    result = {
        'eps': params['eps'],
        'min_samples': params['min_samples'],
        'silhouette': score,
        'anomaly_count': anomaly_count
    }
    results.append(result)
    
    # Update best parameters if current score is better
    if score > best_score and anomaly_count > 0:
        best_score = score
        best_params = params
        best_anomaly_count = anomaly_count

# Create DataFrame from results
results_df = pd.DataFrame(results)
print("DBSCAN hyperparameter tuning results:")
print(results_df.sort_values('silhouette', ascending=False).head())

if best_params:
    print(f"Best parameters: eps={best_params['eps']}, min_samples={best_params['min_samples']}")
    print(f"Best silhouette score: {best_score:.3f}")
    print(f"Number of anomalies with best parameters: {best_anomaly_count}")
else:
    # Use default parameters if no good parameters found
    best_params = {'eps': 0.8, 'min_samples': 5}
    print("Using default parameters as no good parameters found")

# Apply DBSCAN with best parameters
dbscan = DBSCAN(eps=best_params['eps'], min_samples=best_params['min_samples'])
cluster_labels = dbscan.fit_predict(X_scaled)

# Mark DBSCAN noise as anomaly (-1 label)
data["dbscan_anomaly"] = (cluster_labels == -1).astype(int)
print("Number of anomalies (DBSCAN):", data["dbscan_anomaly"].sum())

# === VISUALIZE ANOMALIES ===
print("Plotting anomalies...")

# Convert time for better plotting
data['time_index'] = range(len(data))

# Plot 1: Time series with anomalies
plt.figure(figsize=(15, 10))

# Plot temperature over time
plt.subplot(3, 1, 1)
plt.plot(data['time_index'], data['temperature'], 'b-', label='Temperature')
plt.scatter(data[data['dbscan_anomaly'] == 1]['time_index'], 
            data[data['dbscan_anomaly'] == 1]['temperature'], 
            color='red', s=50, label='Anomalies')
plt.title('Temperature Over Time with Anomalies')
plt.ylabel('Temperature (°C)')
plt.legend()

# Plot humidity over time
plt.subplot(3, 1, 2)
plt.plot(data['time_index'], data['humidity'], 'g-', label='Humidity')
plt.scatter(data[data['dbscan_anomaly'] == 1]['time_index'], 
            data[data['dbscan_anomaly'] == 1]['humidity'], 
            color='red', s=50, label='Anomalies')
plt.title('Humidity Over Time with Anomalies')
plt.ylabel('Humidity (%)')
plt.legend()

# Plot temperature difference
plt.subplot(3, 1, 3)
plt.plot(data['time_index'], data['temp_diff'], 'm-', label='Temp Difference')
plt.scatter(data[data['dbscan_anomaly'] == 1]['time_index'], 
            data[data['dbscan_anomaly'] == 1]['temp_diff'], 
            color='red', s=50, label='Anomalies')
plt.title('Temperature Difference with Anomalies')
plt.xlabel('Time Index')
plt.ylabel('Temp Diff (°C)')
plt.legend()

plt.tight_layout()
plt.savefig('time_series_anomalies.png')
plt.show()

# Combine labels
data["final_label"] = data["dbscan_anomaly"]  # Use DBSCAN as final label

# Prepare data for modeling
X = data[["temperature", "humidity", "temp_diff"]].values.astype(np.float32)
y = data["final_label"].values

# Apply SMOTE for balancing the dataset
smote = SMOTE(sampling_strategy=0.3, k_neighbors=10, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check the class distribution after SMOTE
print("Class distribution after SMOTE:", pd.Series(y_resampled).value_counts())
print(f"X_resampled shape: {X_resampled.shape}")

# === IMPLEMENT CROSS-VALIDATION ===
print("\n=== Implementing Cross-Validation ===")

# Number of folds for cross-validation
n_folds = 5

# Initialize StratifiedKFold for balanced folds
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Lists to store performance metrics for each fold
cnn_f1_scores = []
lstm_f1_scores = []
cnn_pr_scores = []
lstm_pr_scores = []
cnn_histories = []
lstm_histories = []

# Arrays to store all predictions for aggregate metrics
all_y_true = []
all_y_pred_cnn = []
all_y_pred_lstm = []

print(f"Starting {n_folds}-fold cross-validation...")

# Loop through each fold
for fold, (train_index, test_index) in enumerate(skf.split(X_resampled, y_resampled)):
    print(f"\nFold {fold+1}/{n_folds}")
    
    # Split data
    X_train, X_test = X_resampled[train_index], X_resampled[test_index]
    y_train, y_test = y_resampled[train_index], y_resampled[test_index]
    
    # Reshape data for models
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Save test set data for aggregate metrics
    all_y_true.extend(y_test)
    
    # === CNN MODEL ===
    print("Training CNN model...")
    cnn_model = Sequential()
    cnn_model.add(Input(shape=(3, 1)))  # 3 features, 1 channel
    cnn_model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
    cnn_model.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(32, activation='relu'))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(Dense(1, activation='sigmoid'))
    
    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train CNN model
    cnn_history = cnn_model.fit(
        X_train_reshaped, 
        y_train, 
        epochs=5, 
        batch_size=32, 
        validation_data=(X_test_reshaped, y_test),
        verbose=0
    )
    cnn_histories.append(cnn_history)
    
    # === LSTM MODEL ===
    print("Training LSTM model...")
    lstm_model = Sequential()
    lstm_model.add(Input(shape=(3, 1)))
    lstm_model.add(LSTM(32, return_sequences=False))
    lstm_model.add(Dense(16, activation='relu'))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(1, activation='sigmoid'))
    
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train LSTM model
    lstm_history = lstm_model.fit(
        X_train_reshaped, 
        y_train, 
        epochs=5, 
        batch_size=32, 
        validation_data=(X_test_reshaped, y_test),
        verbose=0
    )
    lstm_histories.append(lstm_history)
    
    # Evaluate the models
    y_pred_cnn = (cnn_model.predict(X_test_reshaped, verbose=0) > 0.5).astype(int)
    y_pred_lstm = (lstm_model.predict(X_test_reshaped, verbose=0) > 0.5).astype(int)
    
    # Save predictions for aggregate metrics
    all_y_pred_cnn.extend(y_pred_cnn)
    all_y_pred_lstm.extend(y_pred_lstm)
    
    # Calculate metrics for this fold
    f1_cnn = f1_score(y_test, y_pred_cnn, pos_label=1)
    pr_score_cnn = average_precision_score(y_test, y_pred_cnn)
    f1_lstm = f1_score(y_test, y_pred_lstm, pos_label=1)
    pr_score_lstm = average_precision_score(y_test, y_pred_lstm)
    
    # Store scores
    cnn_f1_scores.append(f1_cnn)
    cnn_pr_scores.append(pr_score_cnn)
    lstm_f1_scores.append(f1_lstm)
    lstm_pr_scores.append(pr_score_lstm)
    
    # Print fold results
    print(f"CNN - F1: {f1_cnn:.3f}, PR: {pr_score_cnn:.3f} | LSTM - F1: {f1_lstm:.3f}, PR: {pr_score_lstm:.3f}")

# Convert lists to numpy arrays for calculations
all_y_true = np.array(all_y_true)
all_y_pred_cnn = np.array(all_y_pred_cnn)
all_y_pred_lstm = np.array(all_y_pred_lstm)

# === OVERALL CROSS-VALIDATION RESULTS ===
print("\n=== Cross-Validation Results ===")

# Calculate average metrics across folds
avg_f1_cnn = np.mean(cnn_f1_scores)
avg_pr_cnn = np.mean(cnn_pr_scores)
avg_f1_lstm = np.mean(lstm_f1_scores)
avg_pr_lstm = np.mean(lstm_pr_scores)

# Calculate std deviation for confidence intervals
std_f1_cnn = np.std(cnn_f1_scores)
std_pr_cnn = np.std(cnn_pr_scores)
std_f1_lstm = np.std(lstm_f1_scores)
std_pr_lstm = np.std(lstm_pr_scores)

# Print average results with standard deviations
print(f"\nCNN Model - Average Metrics:")
print(f"F1 Score: {avg_f1_cnn:.3f} (±{std_f1_cnn:.3f})")
print(f"PR Score: {avg_pr_cnn:.3f} (±{std_pr_cnn:.3f})")

print(f"\nLSTM Model - Average Metrics:")
print(f"F1 Score: {avg_f1_lstm:.3f} (±{std_f1_lstm:.3f})")
print(f"PR Score: {avg_pr_lstm:.3f} (±{std_pr_lstm:.3f})")

# Calculate aggregate metrics on all predictions
print("\n=== Aggregate Classification Reports ===")

# CNN Aggregate Results
print("\nCNN Classification Report (Aggregate):")
print(classification_report(all_y_true, all_y_pred_cnn, target_names=["Normal", "Anomaly"]))

# LSTM Aggregate Results
print("\nLSTM Classification Report (Aggregate):")
print(classification_report(all_y_true, all_y_pred_lstm, target_names=["Normal", "Anomaly"]))

# Aggregate F1 and PR scores
f1_cnn_agg = f1_score(all_y_true, all_y_pred_cnn, pos_label=1)
pr_score_cnn_agg = average_precision_score(all_y_true, all_y_pred_cnn)
f1_lstm_agg = f1_score(all_y_true, all_y_pred_lstm, pos_label=1)
pr_score_lstm_agg = average_precision_score(all_y_true, all_y_pred_lstm)

print(f"\nAggregate F1 Score (Anomalies) CNN: {f1_cnn_agg:.3f}")
print(f"Aggregate PR Score (Anomalies) CNN: {pr_score_cnn_agg:.3f}")
print(f"\nAggregate F1 Score (Anomalies) LSTM: {f1_lstm_agg:.3f}")
print(f"Aggregate PR Score (Anomalies) LSTM: {pr_score_lstm_agg:.3f}")

# Plot confusion matrices
from sklearn.metrics import confusion_matrix

plt.figure(figsize=(12, 5))

# CNN Confusion Matrix
plt.subplot(1, 2, 1)
cm_cnn = confusion_matrix(all_y_true, all_y_pred_cnn)
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Anomaly'], 
            yticklabels=['Normal', 'Anomaly'])
plt.title('CNN Confusion Matrix (All Folds)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# LSTM Confusion Matrix
plt.subplot(1, 2, 2)
cm_lstm = confusion_matrix(all_y_true, all_y_pred_lstm)
sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Anomaly'], 
            yticklabels=['Normal', 'Anomaly'])
plt.title('LSTM Confusion Matrix (All Folds)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('cross_val_confusion_matrices.png')
plt.show()

# Plot cross-validation metrics boxplots
plt.figure(figsize=(12, 5))

# F1 Score comparison
plt.subplot(1, 2, 1)
boxplot_data_f1 = [cnn_f1_scores, lstm_f1_scores]
plt.boxplot(boxplot_data_f1, labels=['CNN', 'LSTM'])
plt.title('F1 Score Comparison (Cross-Validation)')
plt.ylabel('F1 Score')
plt.grid(True, linestyle='--', alpha=0.7)

# PR Score comparison
plt.subplot(1, 2, 2)
boxplot_data_pr = [cnn_pr_scores, lstm_pr_scores]
plt.boxplot(boxplot_data_pr, labels=['CNN', 'LSTM'])
plt.title('PR Score Comparison (Cross-Validation)')
plt.ylabel('PR Score')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('cross_val_metrics_comparison.png')
plt.show()

# Plot average learning curves
plt.figure(figsize=(12, 10))

# Compute average histories
avg_cnn_train_acc = np.mean([hist.history['accuracy'] for hist in cnn_histories], axis=0)
avg_cnn_val_acc = np.mean([hist.history['val_accuracy'] for hist in cnn_histories], axis=0)
avg_cnn_train_loss = np.mean([hist.history['loss'] for hist in cnn_histories], axis=0)
avg_cnn_val_loss = np.mean([hist.history['val_loss'] for hist in cnn_histories], axis=0)

avg_lstm_train_acc = np.mean([hist.history['accuracy'] for hist in lstm_histories], axis=0)
avg_lstm_val_acc = np.mean([hist.history['val_accuracy'] for hist in lstm_histories], axis=0)
avg_lstm_train_loss = np.mean([hist.history['loss'] for hist in lstm_histories], axis=0)
avg_lstm_val_loss = np.mean([hist.history['val_loss'] for hist in lstm_histories], axis=0)

# Compute standard deviations for error bands
std_cnn_train_acc = np.std([hist.history['accuracy'] for hist in cnn_histories], axis=0)
std_cnn_val_acc = np.std([hist.history['val_accuracy'] for hist in cnn_histories], axis=0)
std_cnn_train_loss = np.std([hist.history['loss'] for hist in cnn_histories], axis=0)
std_cnn_val_loss = np.std([hist.history['val_loss'] for hist in cnn_histories], axis=0)

std_lstm_train_acc = np.std([hist.history['accuracy'] for hist in lstm_histories], axis=0)
std_lstm_val_acc = np.std([hist.history['val_accuracy'] for hist in lstm_histories], axis=0)
std_lstm_train_loss = np.std([hist.history['loss'] for hist in lstm_histories], axis=0)
std_lstm_val_loss = np.std([hist.history['val_loss'] for hist in lstm_histories], axis=0)

epochs = range(1, len(avg_cnn_train_acc) + 1)

# Accuracy plots with error bands
plt.subplot(2, 2, 1)
plt.plot(epochs, avg_cnn_train_acc, 'b-', label='Train')
plt.plot(epochs, avg_cnn_val_acc, 'r-', label='Validation')
plt.fill_between(epochs, avg_cnn_train_acc - std_cnn_train_acc, avg_cnn_train_acc + std_cnn_train_acc, alpha=0.1, color='b')
plt.fill_between(epochs, avg_cnn_val_acc - std_cnn_val_acc, avg_cnn_val_acc + std_cnn_val_acc, alpha=0.1, color='r')
plt.title('CNN Model Accuracy (Avg ± Std)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(epochs, avg_lstm_train_acc, 'b-', label='Train')
plt.plot(epochs, avg_lstm_val_acc, 'r-', label='Validation')
plt.fill_between(epochs, avg_lstm_train_acc - std_lstm_train_acc, avg_lstm_train_acc + std_lstm_train_acc, alpha=0.1, color='b')
plt.fill_between(epochs, avg_lstm_val_acc - std_lstm_val_acc, avg_lstm_val_acc + std_lstm_val_acc, alpha=0.1, color='r')
plt.title('LSTM Model Accuracy (Avg ± Std)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Loss plots with error bands
plt.subplot(2, 2, 3)
plt.plot(epochs, avg_cnn_train_loss, 'b-', label='Train')
plt.plot(epochs, avg_cnn_val_loss, 'r-', label='Validation')
plt.fill_between(epochs, avg_cnn_train_loss - std_cnn_train_loss, avg_cnn_train_loss + std_cnn_train_loss, alpha=0.1, color='b')
plt.fill_between(epochs, avg_cnn_val_loss - std_cnn_val_loss, avg_cnn_val_loss + std_cnn_val_loss, alpha=0.1, color='r')
plt.title('CNN Model Loss (Avg ± Std)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(epochs, avg_lstm_train_loss, 'b-', label='Train')
plt.plot(epochs, avg_lstm_val_loss, 'r-', label='Validation')
plt.fill_between(epochs, avg_lstm_train_loss - std_lstm_train_loss, avg_lstm_train_loss + std_lstm_train_loss, alpha=0.1, color='b')
plt.fill_between(epochs, avg_lstm_val_loss - std_lstm_val_loss, avg_lstm_val_loss + std_lstm_val_loss, alpha=0.1, color='r')
plt.title('LSTM Model Loss (Avg ± Std)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig('cross_val_learning_curves.png')
plt.show()

# Save cross-validation results to CSV
cv_results = pd.DataFrame({
    'Fold': list(range(1, n_folds + 1)) + ['Average', 'Std Dev', 'Aggregate'],
    'CNN_F1': cnn_f1_scores + [avg_f1_cnn, std_f1_cnn, f1_cnn_agg],
    'CNN_PR': cnn_pr_scores + [avg_pr_cnn, std_pr_cnn, pr_score_cnn_agg],
    'LSTM_F1': lstm_f1_scores + [avg_f1_lstm, std_f1_lstm, f1_lstm_agg],
    'LSTM_PR': lstm_pr_scores + [avg_pr_lstm, std_pr_lstm, pr_score_lstm_agg]
})

cv_results.to_csv('cross_validation_results.csv', index=False)
print("Cross-validation results saved to CSV file.")

# Determine which model performed better based on aggregate F1 scores
better_model = "CNN" if f1_cnn_agg > f1_lstm_agg else "LSTM"
print(f"\n🏆 Based on cross-validation results, the {better_model} model performs better for anomaly detection.")
print("Results saved to CSV files and plots saved as PNG images.")