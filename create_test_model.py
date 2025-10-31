import tensorflow as tf
from tensorflow.keras import layers

# Create a simple model
model = tf.keras.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(124, 124, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(6, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Save the model
model.save('Ewaste1.keras')