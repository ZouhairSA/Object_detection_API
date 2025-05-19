from flask import Flask, request, jsonify
import torch
from ultralytics import YOLO
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
model = YOLO('yolo11n.pt') 

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    image = Image.open(file.stream)
    results = model(image)
    detections = results[0].boxes.data.tolist()
    return jsonify({'detections': detections})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))  # Utilise le port défini par Render.com, ou 5000 par défaut
    app.run(host='0.0.0.0', port=port, debug=False)