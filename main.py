from flask import Flask, request, Response
from flask_cors import CORS
from detection_service import DetectionService
import cv2
from PIL import Image
import numpy as np
import io

MODEL_PATH = './model.keras'
image_size = 512

app = Flask(__name__)
CORS(app)
service = DetectionService(MODEL_PATH)


@app.route("/detect-service", methods=['POST'])
def main():
    image = cv2.imdecode(np.frombuffer(request.data, np.uint8), 1)
    img_resized = service.resize_image(image, image_size)

    images = service.prepare_image(img_resized)
    result = service.predict(images)

    r = result[0].reshape((512, 512))
    result_img = service.create_transparent_mask(r)

    orig = Image.fromarray(img_resized)
    combined = service.combine_images(orig, result_img)
    combined_bytes = io.BytesIO()

    combined.save(combined_bytes, format='png')
    combined_bytes.seek(0)
    return Response(response=combined_bytes.getvalue(), mimetype='image/png')


if __name__ == '__main__':
    print('Started')
    app.run(host='0.0.0.0', debug=True)
