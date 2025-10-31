import tensorflow as tf 
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  # Plotting graphs and images
from sklearn.metrics import confusion_matrix, classification_report  
import gradio as gr  
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image 

from keras import layers, models, optimizers, callbacks  
from keras.models import Sequential, load_model  
from keras.applications import EfficientNetV2B0, EfficientNetV2B2
from keras.applications.efficientnet_v2 import preprocess_input
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image

import matplotlib
# matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt


# Define the dataset directory and parameters
dataset_dir = 'C:\\Users\\LENOVO\\Desktop\\E-Waste'  # Update this path to your dataset location
image_size = (124, 124)
batch_size = 32
seed = 42

# Load the dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=seed,
    shuffle = True,
    image_size=image_size,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    shuffle = True,
    image_size=image_size,
    batch_size=batch_size
)
val_class = val_ds.class_names

# Get the total number of batches in the validation dataset
val_batches = tf.data.experimental.cardinality(val_ds)  

# Split the validation dataset into two equal parts:
# First half becomes the test dataset
test_ds = val_ds.take(val_batches // 2)  

# Second half remains as the validation dataset
val_dat = val_ds.skip(val_batches // 2)  

# Optimize test dataset by caching and prefetching to improve performance
test_ds_eval = test_ds.cache().prefetch(tf.data.AUTOTUNE)  

print("Class Names:", train_ds.class_names)
print("Validation Classes:", val_class)
print("Number of Classes:", len(train_ds.class_names))

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(min(12, len(images))):
        ax = plt.subplot(4, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(train_ds.class_names[labels[i]])
        plt.axis("off")
plt.show()

# Function to count the distribution of classes in the dataset
def count_distribution(dataset, class_names):
    total = 0
    counts = {name: 0 for name in class_names}
    
    for _, labels in dataset:
        for label in labels.numpy():
            class_name = class_names[label]
            counts[class_name] += 1
            total += 1

    for k in counts:
        counts[k] = round((counts[k] / total) * 100, 2)  # Convert to percentage
    return counts

# Function to plot class distribution
def simple_bar_plot(dist, title):
    plt.figure(figsize=(10, 6))
    plt.bar(dist.keys(), dist.values(), color='cornflowerblue')
    plt.title(title)
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()

class_names = train_ds.class_names

# Get class distributions
train_dist = count_distribution(train_ds, class_names)
val_dist = count_distribution(val_ds, class_names)
test_dist = count_distribution(test_ds, class_names)
overall_dist = {}
for k in class_names:
    overall_dist[k] = round((train_dist[k] + val_dist[k]) / 2, 2)

print("\nClass Distributions:")
print("Training:", train_dist)
print("Validation:", val_dist)
print("Test:", test_dist)
print("Overall:", overall_dist)

# Show visualizations
simple_bar_plot(train_dist, "Training Set Class Distribution (%)")
simple_bar_plot(val_dist, "Validation Set Class Distribution (%)")
simple_bar_plot(test_dist, "Test Set Class Distribution (%)")
simple_bar_plot(overall_dist, "Overall Class Distribution (%)")

# Count class occurrences and prepare label list
class_counts = {i: 0 for i in range(len(class_names))}
all_labels = []

for images, labels in train_ds:
    for label in labels.numpy():
        class_counts[label] += 1
        all_labels.append(label)

# Compute class weights
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(len(class_names)),
    y=all_labels
)

# Create dictionary mapping class index to weight
class_weights = {i: w for i, w in enumerate(class_weights_array)}

print("\nClass Statistics:")
print("Class Counts:", class_counts)
print("Class Weights:", class_weights)

##2 Processing
#  Define data augmentation pipeline
data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

##3 Model Selection
#  Load the pretrained EfficientNetV2B2 model
base_model = EfficientNetV2B2(
    include_top=False,
    input_shape=(124, 124, 3),
    include_preprocessing=True,
    weights='imagenet'
)

#  Freeze early layers
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

###4 Model Architecture
#  Build the final model with improved architecture
model = Sequential([
    layers.Input(shape=(124, 124, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')  # num classes from dataset
])

# Compile the model with improved settings
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4, weight_decay=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Set training parameters (short smoke test)
epochs = 2  # run only 1-2 epochs for quick verification
batch_size = 32

# Train the model
print("\nStarting model training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    class_weight=class_weights,
    callbacks=[early_stopping]
)

# Print model summaries
print("\nModel Architecture:")
model.summary()
print("\nBase Model Architecture:")
base_model.summary()

### 4 Performance Visualization
# Extract metrics
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

# Plot training metrics
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training vs Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training vs Validation Loss')
plt.tight_layout()
plt.show()

##5 Evaluation
# Evaluate on test set
print("\nEvaluating model on test set...")
loss, accuracy = model.evaluate(test_ds_eval)
print(f'Test accuracy: {accuracy:.4f}, Test loss: {loss:.4f}')

# Get predictions
y_true = np.concatenate([y.numpy() for x, y in test_ds_eval], axis=0)
y_pred_probs = model.predict(test_ds_eval)
y_pred = np.argmax(y_pred_probs, axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names,
            yticklabels=class_names,
            cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

##6 Save and Test Model
# Save the model
print("\nSaving model...")
model.save('Ewaste1.keras')

# Visual test on sample images
print("\nVisual testing on sample images...")
for images, labels in test_ds_eval.take(1):
    predictions = model.predict(images)
    pred_labels = tf.argmax(predictions, axis=1)
    
    plt.figure(figsize=(15, 8))
    for i in range(min(8, len(images))):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(f"True: {class_names[labels[i]]}\nPred: {class_names[pred_labels[i]]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

##7 Gradio Interface


# Load model
model = tf.keras.models.load_model("model.h5")

# Your class labels
class_names = ['Battery', 'Cable', 'Chip', 'Mobile', 'Plastic']

# Prediction function
def predict(img):
    img = img.resize((224, 224))  # adjust based on your model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    confidence = float(np.max(predictions))
    predicted_class = class_names[int(np.argmax(predictions))]

    # Format output
    result = f"""
    🔋 **Prediction:** {predicted_class}  
    📊 **Confidence:** {confidence * 100:.2f}%  
    🖼️ **Uploaded Image:**  
    """

    return result, img

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an E-Waste Image"),
    outputs=[
        gr.Markdown(label="Result"),
        gr.Image(label="Uploaded Image")
    ],
    title="♻️ E-Waste Classification System",
    description="Upload an image of e-waste and the model will predict its category with confidence."
)

iface.launch()