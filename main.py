from flask import Flask, request, Response, abort
from .detection_service import DetectionService
import tensorflow as tf
import cv2
import numpy as np

MODEL_PATH = './model.keras'
image_size = 512

app = Flask(__name__)
service = DetectionService(MODEL_PATH)

@app.route("/detect-service", methods=['GET'])
def main():
    image = request.files["images"]
    image_name = image.filename
    img_raw = tf.image.decode_image(
        open(image_name, 'rb').read(), channels=3)
    
    img = tf.expand_dims(img_raw, 0)
    img = service.prepare_image(img, image_size)

    img_ex = np.expand_dims(img, axis=0)
    IMAGES = np.vstack([img_ex])

    images = np.array(IMAGES)

    result = service.predict(images)

    _, img_encoded = cv2.imencode('.png', result[0])
    response = img_encoded.tostring()

    try:
        return Response(response=response, status=200, mimetype='image/png')
    except FileNotFoundError:
        abort(500)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
