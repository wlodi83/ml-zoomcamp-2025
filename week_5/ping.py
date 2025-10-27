from flask import Flask, jsonify

app = Flask('ping_service')

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"message": "pong"}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)