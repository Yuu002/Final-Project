import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# ----------------------------
# 1) โหลด dataset
# ----------------------------
data = pd.read_csv("bands_extracted.csv")

# Features และ Target
X = data[["Band1", "Band2", "Band3", "Band4"]].values
y = data[["Blue (B1)", "Green (B2)", "Red (B3)", "NIR (B4)"]].values

# ----------------------------
# 2) Normalize
# ----------------------------
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# ----------------------------
# 3) Train/Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# ----------------------------
# 4) สร้างโมเดล MLP พร้อม Dropout + L2
# ----------------------------
model = keras.Sequential([
    keras.layers.Dense(32, activation="relu",
                       input_shape=(X_train.shape[1],),
                       kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation="relu",
                       kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation="relu",
                       kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dense(4)  # output: Blue_gt, Green_gt, Red_gt, NIR_gt
])

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

# ----------------------------
# 5) Early Stopping
# ----------------------------
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

# ----------------------------
# 6) Train
# ----------------------------
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# ----------------------------
# 7) Evaluate
# ----------------------------
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MAE: {mae:.4f}")

# ----------------------------
# 8) Visualization
# ----------------------------
# Loss curve
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train vs Validation Loss')
plt.legend()
plt.show()

# Predict
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)

# Scatter plot per band
bands = ["Blue", "Green", "Red", "NIR"]
for i in range(4):
    plt.figure(figsize=(5,5))
    plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.7)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"{bands[i]}: True vs Predicted")
    plt.plot([y_true[:, i].min(), y_true[:, i].max()],
             [y_true[:, i].min(), y_true[:, i].max()],
             "r--", lw=2)  # เส้น y=x
    plt.show()
