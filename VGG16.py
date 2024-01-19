import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np
class VGG16Classifier:
    def __init__(self):
        # Load the full VGG16 model
        self.base_model = VGG16(weights='imagenet')
        # Create a new model that outputs the layer before the last layer
        self.model = Model(inputs=self.base_model.inputs, outputs=self.base_model.layers[-2].output)

    def preprocess_image(self, img_path):
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array

    def get_features_before_last_layer(self, img_path):
        # Preprocess the image
        img_array = self.preprocess_image(img_path)
        # Get the output of the second-to-last layer
        features = self.model.predict(img_array)
        return features
    def predict(self, img_path):
        # Preprocess the image
        img_array = self.preprocess_image(img_path)
        # Get model predictions
        predictions = self.base_model.predict(img_array)
        return predictions