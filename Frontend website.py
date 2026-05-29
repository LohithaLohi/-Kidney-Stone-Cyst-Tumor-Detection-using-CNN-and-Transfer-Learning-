from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = load_model("kidney_cnn_model.h5")

# Class names
classes = ['Cyst', 'Normal', 'Stone', 'Tumor']

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction
@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['file']

    filepath = os.path.join("static", file.filename)

    file.save(filepath)

    img = image.load_img(filepath, target_size=(200,200))

    img = image.img_to_array(img)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    result = classes[np.argmax(prediction)]

    return render_template(
        'index.html',
        prediction=result,
        img_path=filepath
    )

# Run app
if __name__ == "__main__":
    app.run(debug=True)