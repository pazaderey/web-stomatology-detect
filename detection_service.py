import keras
import cv2


class DetectionService:
    def __init__(self, model_path: str):
        self.model = keras.models.load_model(model_path, compile=False)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def predict(self, images):
        result = self.model.predict(images)
        print(result[0])
        
    def prepare_image(self, img, image_size):
        img = cv2.resize(img, (image_size, image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
