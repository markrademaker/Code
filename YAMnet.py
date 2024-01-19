import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras.models import Model
class YAMnetClassifier:
    def __init__(self):
        # Load the YAMnet model from TensorFlow Hub
        self.original_model = hub.load('https://tfhub.dev/google/yamnet/1')
        self.class_names = self.model.class_names
        self.model = Model(inputs=self.original_model.inputs, 
                           outputs=self.original_model.layers[-2].output)

    def predict(self, audio_data):
        # Use the custom model to get the output of the second-to-last layer
        return self.model.predict(audio_data)

    def predict(self, audio_data):
        # Ensure the audio is at the correct sample rate (16kHz)
        # Note: You might need to preprocess your audio data to match this requirement
        scores, embeddings, spectrogram = self.original_model(audio_data)
        # Use softmax to convert logits to probabilities
        probabilities = tf.nn.softmax(scores, axis=-1)
        # Get the top predicted class
        top_class = np.argmax(probabilities, axis=-1)
        return [self.class_names[i] for i in top_class]
    
    def process(self, audio_data):
        # Ensure the audio is at the correct sample rate (16kHz)
        # Note: You might need to preprocess your audio data to match this requirement
        scores, embeddings, spectrogram = self.model(audio_data)
        # Use softmax to convert logits to probabilities
        probabilities = tf.nn.softmax(scores, axis=-1)
        # Get the top predicted class
        top_class = np.argmax(probabilities, axis=-1)
        return [self.class_names[i] for i in top_class]   
    def get_embedding(self, audio_data):
        # Ensure the audio is at the correct sample rate (16kHz)
        # Note: You might need to preprocess your audio data to match this requirement
        scores, embeddings, spectrogram = self.model(audio_data)
        return embeddings 