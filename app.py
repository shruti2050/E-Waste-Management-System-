from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import traceback
from keras.applications.efficientnet_v2 import preprocess_input

# Initialize Flask app
app = Flask(__name__)

try:
    # Load the trained model (expected file produced by training)
    MODEL_PATH = "Ewaste1.keras"
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    # If loading fails, keep `model` as None and show clear message later
    model = None
    print(f"Error loading model: {str(e)}")
    traceback.print_exc()

# Class names (same order as your dataset)
class_names = ['Battery', 'Charger', 'Circuit Board', 'Keyboard', 'Mobile', 'Monitor']  # 🧩 Change to your classes

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((124, 124), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, None
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded.")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected.")

        if file:
            # If the model didn't load, return an informative error immediately
            if model is None:
                return render_template(
                    "index.html",
                    error=("Model not loaded. Please train the model first so that 'Ewaste1.keras'"
                           " exists in the project root, or place a trained model file there.")
                )
            try:
                # Save the uploaded file
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)

                # Preprocess the image
                img_array, error = preprocess_image(filepath)
                if error:
                    return render_template("index.html", error=error)

                # Make prediction with error handling
                try:
                    prediction = model.predict(img_array, verbose=0)
                    
                    # Get top 3 predictions
                    top_indices = np.argsort(prediction[0])[-3:][::-1]
                    predictions = []
                    
                    for idx in top_indices:
                        confidence = float(prediction[0][idx])
                        if confidence > 0.1:  # 10% confidence threshold
                            predictions.append({
                                'label': class_names[idx],
                                'confidence': f"{confidence*100:.2f}%"
                            })
                    
                    if not predictions:
                        return render_template(
                            "index.html",
                            error="Unable to make a confident prediction. Please try a clearer image."
                        )
                    
                    # Return the best prediction and alternatives
                    return render_template(
                        "index.html",
                        uploaded_image=file.filename,
                        label=predictions[0]['label'],
                        confidence=predictions[0]['confidence'],
                        alternatives=predictions[1:] if len(predictions) > 1 else None
                    )
                
                except Exception as e:
                    return render_template(
                        "index.html",
                        error=f"Error during prediction: {str(e)}"
                    )
                    
            except Exception as e:
                return render_template(
                    "index.html",
                    error=f"Error processing upload: {str(e)}"
                )

            # (all successful returns are handled above)

    return render_template("index.html")
    
if __name__ == "__main__":
    app.run(debug=True)
