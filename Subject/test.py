import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow import keras

# ================================
# 1. Load Data (local project path)
# ================================
# 현재 파일 기준 상대경로
# ../Data/Option_Data.csv 로 불러오기
data_path = "./Data/Option_Data.csv"

df = pd.read_csv(data_path)

y = df[['Option Price with Noise','Option Price']]   # target
X = df[['Spot price','Strike Price','Risk Free Rate','Volatility','Maturity','Dividend']]

# ================================
# 2. Train / Val / Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=100)

# ================================
# 3. Scaling
# ================================
scaler = StandardScaler()
scaler.fit(X_train)

X_scaled_train = scaler.transform(X_train)
X_scaled_vals = scaler.transform(X_val)
X_scaled_test = scaler.transform(X_test)

y_train = np.asarray(y_train)
y_val = np.asarray(y_val)
y_test = np.asarray(y_test)

# ================================
# 4. Build NN Model (20-20-20 sigmoid)
# ================================
model = keras.models.Sequential([
    Dense(20, activation="sigmoid", input_shape=(6,)),
    Dense(20, activation="sigmoid"),
    Dense(20, activation="sigmoid"),
    Dense(1)
])

model.summary()

# ================================
# 5. Compile (MAE + Adam)
# ================================
model.compile(loss="mae", optimizer="adam")

# ================================
# 6. Train (10,000 epochs)
# ================================
history = model.fit(
    X_scaled_train,
    y_train[:, 0],       # predict noisy option price
    epochs=10000,
    batch_size=128,
    verbose=1,
    validation_data=(X_scaled_vals, y_val[:, 0])
)

# ================================
# 7. Plot MAE Curve
# ================================
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Training MAE')
plt.plot(history.history['val_loss'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.gca().set_ylim(0.1,0.21)
plt.title('Training vs Validation MAE (Sigmoid Activation, 10,000 epochs)')
plt.legend()
plt.grid(True)
plt.show()
