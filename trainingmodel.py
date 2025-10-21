import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# -------------------------------
# 1. Load dataset
# -------------------------------
data = pd.read_csv("fer2013.csv")

# Images
pixels = data['pixels'].tolist()
X = np.array([np.fromstring(p, sep=' ') for p in pixels], dtype='float32')
X = X.reshape(-1, 48, 48, 1) / 255.0  # normalize

# Labels
y = to_categorical(data['emotion'], num_classes=7)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 2. Build CNN Model
# -------------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D((2,2)),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')   # 7 emotions
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------------
# 3. Train Model
# -------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=25,
    batch_size=64
)

# -------------------------------
# 4. Save Model
# -------------------------------
model.save("emotion_model.h5")
print("✅ Model trained and saved as emotion_model.h5")
