import os
import numpy as np
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import mysql.connector
from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load model
try:
    new_model = load_model('a_g1.h5')
    print("Model loaded successfully.")
except Exception as e:
    new_model = None
    print(f"Error loading model: {e}")

# Fungsi untuk memuat dan memproses gambar
def load_image(filepath):
    test_img = image.load_img(filepath, target_size=(198, 198))
    test_img = image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)
    test_img /= 255.0
    return test_img

# Fungsi prediksi menggunakan model
def model_predict(img_path):
    global new_model
    if new_model is None:
        raise ValueError("Model is not loaded.")

    age_pred, gender_pred = new_model.predict(load_image(img_path))
    gender = 'male' if gender_pred[0][0] > gender_pred[0][1] else 'female'
    return age_pred[0][0], gender

# Endpoint Home
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Age and Gender Prediction API!"})

# Endpoint untuk prediksi usia dan gender
@app.route('/predict', methods=['POST'])
def predict():
    if new_model is None:
        return jsonify({"error": "Model not loaded. Please check the model file."}), 500

    # Validasi apakah file gambar diunggah
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    # Simpan file sementara
    file_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(file_path)

    # Prediksi usia dan gender
    try:
        age, gender = model_predict(file_path)

        # Encode gambar ke Base64
        with open(file_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')

        os.remove(file_path)  # Hapus file setelah prediksi
        return jsonify({
            "age": round(age, 2),
            "gender": gender,
            "image_base64": encoded_image
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)