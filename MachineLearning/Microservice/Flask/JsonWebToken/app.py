from flask import Flask, jsonify, request
from flask_jwt_extended import JWTManager, create_access_token, jwt_required

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'supersecretkey'

jwt = JWTManager(app)

# Data user untuk contoh
users = [{"username": "user1", "password": "pass1"}]

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    user = next((user for user in users if user["username"] == data["username"] and user["password"] == data["password"]), None)
    if user:
        access_token = create_access_token(identity=user["username"])
        return jsonify({"access_token": access_token})
    return jsonify({"error": "Invalid credentials"}), 401

@app.route('/protected', methods=['GET'])
@jwt_required()
def protected():
    return jsonify({"message": "This is a protected route!"})

if __name__ == '__main__':
    app.run(debug=True)