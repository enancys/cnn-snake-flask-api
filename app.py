from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# ✅ Load model CNN kamu
model = tf.keras.models.load_model("model_venom_classifier_v2.keras")

# Fungsi untuk memproses gambar yang diupload
def prepare_image(image_file, target_size=(224, 224)):
    img = Image.open(io.BytesIO(image_file)).convert("RGB")
    img = img.resize(target_size)
    img_array = np.asarray(img) / 255.0  # normalisasi
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image'].read()
    img_array = prepare_image(image_file)

    # 🔍 Prediksi
    prediction = model.predict(img_array)[0][0]

    # Interpretasi output binary classification
    label = "Non-Venomous" if prediction >= 0.5 else "Venomous"
    confidence = float(prediction if prediction >= 0.5 else 1 - prediction)

    return jsonify({
        "label": label,
        "confidence": f"{confidence * 100:.2f}%"
    })

if __name__ == '__main__':
    app.run(debug=True)
