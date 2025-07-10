import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Dataset path
base_path = "c://Users//acer//Downloads//archive//Rice_Image_Dataset"

# Prepare dataframe
image_paths = list(Path(base_path).rglob("*.jpg"))
labels = [x.parent.name for x in image_paths]

df = pd.DataFrame({
    "filepath": [str(path) for path in image_paths],
    "label": labels
})

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# Data generators
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_dataframe(
    train_df,
    x_col='filepath',
    y_col='label',
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

val_data = val_gen.flow_from_dataframe(
    test_df,
    x_col='filepath',
    y_col='label',
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True),
    ModelCheckpoint("model_cnnh5.keras", monitor="val_accuracy", save_best_only=True),
    ReduceLROnPlateau(patience=2, factor=0.2, min_lr=1e-4)
]

# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    callbacks=callbacks
)

# Evaluate and save test data
X_test = []
y_test = []
for i in range(len(val_data)):
    x, y = val_data[i]
    X_test.extend(x)
    y_test.extend(y)
    if i >= len(val_data) - 1:
        break

X_test = np.array(X_test)
y_test = np.array(y_test)

np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.show()