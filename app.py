from flask import Flask, render_template, request
from keras.models import load_model # type: ignore
import numpy as np
from PIL import Image
app = Flask(__name__)
model = load_model('model\saved_image_model.h5')
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if 'file' not in request.files:
        return render_template('index.html', prediction_text='No file part')
    if file.filename == '':
        return render_template('index.html', prediction_text='First Select The Image')
    img = Image.open(file.stream).convert("L")
    img = img.resize((224, 224))  # Resize image to match model input shape
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    if predicted_class == 0:
        predicted_class = "Pneumonia not Detect"
    else:
        predicted_class = "Pneumonia Detect"
    return render_template('index.html', prediction_text=f'{predicted_class}')
if __name__ == '__main__':
    app.run(debug=True)