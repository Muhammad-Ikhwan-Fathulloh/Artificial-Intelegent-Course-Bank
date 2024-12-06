import mysql.connector
from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

# Konfigurasi Database
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'learning_bangkit_ml'
}

# Konfigurasi JWT
app.config['JWT_SECRET_KEY'] = 'bangkit101'
jwt = JWTManager(app)

# Fungsi untuk Membuat Koneksi ke Database
def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

# Endpoint Home
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Microservice with MySQL and JWT!"})

# Registrasi User
@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    # Hash password sebelum disimpan
    hashed_password = generate_password_hash(password)

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({"message": "User registered successfully"}), 201

# Login dan Mendapatkan JWT
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()

    if not user or not check_password_hash(user['password'], password):
        return jsonify({"error": "Invalid username or password"}), 401

    # Generate JWT
    access_token = create_access_token(identity={"username": user["username"]})
    return jsonify({"access_token": access_token}), 200

# Endpoint untuk Mendapatkan Semua Item
@app.route('/items', methods=['GET'])
@jwt_required()
def get_items():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM items")
    items = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(items), 200

# Endpoint untuk Menambahkan Item Baru
@app.route('/items', methods=['POST'])
@jwt_required()
def add_item():
    data = request.json
    name = data.get('name')
    description = data.get('description')

    if not name or not description:
        return jsonify({"error": "Name and description are required"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO items (name, description) VALUES (%s, %s)", (name, description))
    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({"message": "Item added successfully"}), 201

# Endpoint untuk Mengupdate Item
@app.route('/items/<int:item_id>', methods=['PUT'])
@jwt_required()
def update_item(item_id):
    data = request.json
    name = data.get('name')
    description = data.get('description')

    if not name or not description:
        return jsonify({"error": "Name and description are required"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE items SET name = %s, description = %s WHERE id = %s", (name, description, item_id))
    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({"message": "Item updated successfully"}), 200

# Endpoint untuk Menghapus Item
@app.route('/items/<int:item_id>', methods=['DELETE'])
@jwt_required()
def delete_item(item_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM items WHERE id = %s", (item_id,))
    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({"message": "Item deleted successfully"}), 200

# Endpoint untuk Mendapatkan Detail Item
@app.route('/items/<int:item_id>', methods=['GET'])
@jwt_required()
def get_item(item_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM items WHERE id = %s", (item_id,))
    item = cursor.fetchone()
    cursor.close()
    conn.close()

    if not item:
        return jsonify({"error": "Item not found"}), 404

    return jsonify(item), 200

if __name__ == '__main__':
    app.run(debug=True)