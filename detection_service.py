import keras
import cv2
from cv2.typing import MatLike
import numpy as np
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap


class DetectionService:
    def __init__(self, model_path: str):
        self.model = keras.models.load_model(model_path)
        self.model.compile(
            optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    @staticmethod
    def create_detection_colormap() -> LinearSegmentedColormap:
        colors = [(0, 0, 0, 0), (1, 1, 0, 1)]
        n_bins = 100
        return LinearSegmentedColormap.from_list("CustomColormap_TransparentBlack_OpaqueYellow", colors, N=n_bins)

    def predict(self, images: np.ndarray) -> np.ndarray:
        result = self.model.predict(images)
        return result

    def resize_image(self, img: MatLike, image_size: int) -> MatLike:
        img = cv2.resize(img, (image_size, image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def prepare_image(self, image: MatLike) -> np.ndarray:
        img_ex = np.expand_dims(image, axis=0)
        images_stack = np.vstack([img_ex])
        return np.array(images_stack)

    def create_transparent_mask(self, r: np.ndarray) -> Image.Image:
        cmap_with_transparency = DetectionService.create_detection_colormap()

        result_img = Image.fromarray(np.uint8(cmap_with_transparency(r) * 255))
        result_img.convert('RGBA')
        result_img.putalpha(127)

        return result_img

    def combine_images(self, orig: Image.Image, predict: Image.Image) -> Image.Image:
        combined_img = orig.copy()
        combined_img.paste(predict, (0, 0), predict)
        return combined_img
