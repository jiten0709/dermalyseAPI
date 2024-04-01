from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image as image
from tensorflow.keras.preprocessing import image as keras_image
from io import BytesIO
import base64


app = Flask(__name__)

# Load your model
model = tf.keras.models.load_model('xception.keras')

class_labels = {'Acne': 0, 'Actinic keratosis': 1, 'Basal Cell Carcinoma': 2, 'Benign keratosis': 3, 'Dermatofibroma': 4, 'Eczema': 5, 'Fungal Infections ': 6, 'Melanocytic Nevi': 7, 'Melanoma': 8, 'Normal Skin': 9, 'Rosacea': 10, 'Squamous cell carcinoma': 11, 'Urticaria Hives': 12, 'Vascular Tumors': 13, 'Warts Molluscum and other Viral Infections': 14}

@app.route('/', methods=['GET'])
def hello():
    return jsonify({'message': 'hello'})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image data'}), 400

    # Decode the Base64 string back into bytes
    image_bytes = base64.b64decode(data['image'])
    # Convert the bytes to a PIL Image object
    img = image.open(BytesIO(image_bytes))
    # Resize the image to the target size
    img = img.resize((299, 299))
    # Convert the image to a NumPy array
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.

    # Assuming model is already loaded
    predictions = model.predict(img_array)
    instance_predictions = predictions[0]
    flattened_predictions = instance_predictions.flatten()
    sorted_indices = np.argsort(flattened_predictions)[::-1]

    top3_indices = sorted_indices[:3]
    top3_predictions = {}

    for index in top3_indices:
        class_name = list(class_labels.keys())[index]
        # Convert float32 to float
        probability = float(flattened_predictions[index])
        top3_predictions[class_name] = probability

    return jsonify(top3_predictions)


if __name__ == '__main__':
    app.run(port=5000)
