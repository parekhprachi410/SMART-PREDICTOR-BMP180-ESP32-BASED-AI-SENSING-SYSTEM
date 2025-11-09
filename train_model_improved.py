import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import joblib

# Load data
df = pd.read_csv('bmp180_data.csv')
data = df[['pressure', 'temperature', 'altitude']].values

print(f"Original data shape: {data.shape}")
print(f"Data statistics:")
print(f"Pressure: min={data[:,0].min():.2f}, max={data[:,0].max():.2f}, mean={data[:,0].mean():.2f}, std={data[:,0].std():.4f}")
print(f"Temperature: min={data[:,1].min():.2f}, max={data[:,1].max():.2f}, mean={data[:,1].mean():.2f}, std={data[:,1].std():.4f}")
print(f"Altitude: min={data[:,2].min():.2f}, max={data[:,2].max():.2f}, mean={data[:,2].mean():.2f}, std={data[:,2].std():.4f}")

# Robust outlier removal for sensor data
def remove_outliers_sensor_data(data, n_std=3):
    """
    Remove outliers while preserving sensor data patterns
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    print(f"\nStandard deviations: Pressure={std[0]:.4f}, Temperature={std[1]:.4f}, Altitude={std[2]:.4f}")
    
    # Create mask for points within n_std standard deviations
    z_scores = np.abs((data - mean) / std)
    mask = np.all(z_scores < n_std, axis=1)
    
    clean_data = data[mask]
    removed_count = len(data) - len(clean_data)
    print(f"Removed {removed_count} outliers ({removed_count/len(data)*100:.1f}%)")
    
    # Show removed outliers statistics
    if removed_count > 0:
        outliers = data[~mask]
        print(f"Outlier range - Pressure: {outliers[:,0].min():.2f} to {outliers[:,0].max():.2f}")
        print(f"Outlier range - Temperature: {outliers[:,1].min():.2f} to {outliers[:,1].max():.2f}")
        print(f"Outlier range - Altitude: {outliers[:,2].min():.2f} to {outliers[:,2].max():.2f}")
    
    return clean_data

# Remove outliers
data_clean = remove_outliers_sensor_data(data, n_std=3)
print(f"Data after outlier removal: {data_clean.shape}")

# Normalize
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_clean)

# Print scaler parameters for ESP32
print("\n" + "="*50)
print("SCALER PARAMETERS FOR ESP32")
print("="*50)
print(f"Data min: [{scaler.data_min_[0]:.2f}, {scaler.data_min_[1]:.2f}, {scaler.data_min_[2]:.2f}]")
print(f"Data max: [{scaler.data_max_[0]:.2f}, {scaler.data_max_[1]:.2f}, {scaler.data_max_[2]:.2f}]")

# Create sequences with optimal length
seq_length = 15  # Good balance for your data variation
X, y = [], []
for i in range(len(data_scaled) - seq_length):
    X.append(data_scaled[i:i+seq_length].flatten())
    y.append(data_scaled[i+seq_length])
X, y = np.array(X), np.array(y)

print(f"\nSequences created - X: {X.shape}, y: {y.shape}")

# Train/val/test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

print(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Optimized model for sensor data prediction
model = Sequential([
    Dense(64, activation='relu', input_shape=(seq_length * 3,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.1),
    Dense(16, activation='relu'),
    Dense(3)  # Output: pressure, temperature, altitude
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

model.summary()

# Train with early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

print("\nStarting training...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

# Evaluate
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
predictions = model.predict(X_test, verbose=0)

# Inverse transform to original scale
predictions_inv = scaler.inverse_transform(predictions)
y_test_inv = scaler.inverse_transform(y_test)

# Calculate accuracy metrics
errors = np.abs(predictions_inv - y_test_inv)
percentage_errors = np.abs((predictions_inv - y_test_inv) / y_test_inv) * 100

# Print detailed results
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)
print(f"Test Loss (MSE): {test_loss:.6f}")
print(f"Test MAE: {test_mae:.6f}")

print("\nMean Absolute Errors (in original units):")
print(f"Pressure: {np.mean(errors[:, 0]):.4f} hPa")
print(f"Temperature: {np.mean(errors[:, 1]):.4f} °C")
print(f"Altitude: {np.mean(errors[:, 2]):.4f} m")

print("\nRoot Mean Square Errors (in original units):")
print(f"Pressure: {np.sqrt(np.mean(errors[:, 0]**2)):.4f} hPa")
print(f"Temperature: {np.sqrt(np.mean(errors[:, 1]**2)):.4f} °C")
print(f"Altitude: {np.sqrt(np.mean(errors[:, 2]**2)):.4f} m")

print("\nMean Percentage Errors:")
print(f"Pressure: {np.mean(percentage_errors[:, 0]):.2f}%")
print(f"Temperature: {np.mean(percentage_errors[:, 1]):.2f}%")
print(f"Altitude: {np.mean(percentage_errors[:, 2]):.2f}%")

# Calculate accuracy within different thresholds
accuracy_1pct = np.mean(percentage_errors < 1) * 100
accuracy_2pct = np.mean(percentage_errors < 2) * 100
accuracy_5pct = np.mean(percentage_errors < 5) * 100

print(f"\nAccuracy within 1%: {accuracy_1pct:.2f}%")
print(f"Accuracy within 2%: {accuracy_2pct:.2f}%")
print(f"Accuracy within 5%: {accuracy_5pct:.2f}%")

# Show some example predictions
print(f"\nExample predictions (first 5 test samples):")
print("Actual vs Predicted:")
for i in range(min(5, len(predictions_inv))):
    print(f"Sample {i+1}:")
    print(f"  Pressure: {y_test_inv[i,0]:.2f} -> {predictions_inv[i,0]:.2f} hPa (error: {errors[i,0]:.2f})")
    print(f"  Temp: {y_test_inv[i,1]:.2f} -> {predictions_inv[i,1]:.2f} °C (error: {errors[i,1]:.2f})")
    print(f"  Altitude: {y_test_inv[i,2]:.2f} -> {predictions_inv[i,2]:.2f} m (error: {errors[i,2]:.2f})")

# Save model and scaler parameters
model.save('bmp180_model.h5')

# Save scaler parameters for ESP32
scaler_params = {
    'data_min': scaler.data_min_.tolist(),
    'data_max': scaler.data_max_.tolist(),
}

np.save('scaler_params.npy', scaler_params)

# Save as text file for easy copying to Arduino
with open('scaler_params.txt', 'w') as f:
    f.write("// Copy these values to your Arduino code:\n\n")
    f.write(f"// Pressure range: {scaler.data_min_[0]:.2f} to {scaler.data_max_[0]:.2f} hPa\n")
    f.write(f"// Temperature range: {scaler.data_min_[1]:.2f} to {scaler.data_max_[1]:.2f} °C\n")
    f.write(f"// Altitude range: {scaler.data_min_[2]:.2f} to {scaler.data_max_[2]:.2f} m\n\n")
    f.write(f"const float data_min[3] = {{{scaler.data_min_[0]:.2f}, {scaler.data_min_[1]:.2f}, {scaler.data_min_[2]:.2f}}};\n")
    f.write(f"const float data_max[3] = {{{scaler.data_max_[0]:.2f}, {scaler.data_max_[1]:.2f}, {scaler.data_max_[2]:.2f}}};\n")

print(f"\nScaler parameters saved to scaler_params.txt")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save TFLite model
with open('bmp180_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Get model size info
tflite_size = len(tflite_model) / 1024
print(f"TFLite model size: {tflite_size:.2f} KB")

print("\n✅ Model saved as bmp180_model.tflite!")
print("✅ Training completed successfully!")

# Instructions for next steps
print("\n" + "="*60)
print("NEXT STEPS FOR ESP32 DEPLOYMENT:")
print("1. Convert the .tflite file to C array using:")
print("   xxd -i bmp180_model.tflite > model.h")
print("2. Update the data_min and data_max arrays in Arduino code")
print("3. Copy model.h to your Arduino project folder")
print("="*60)