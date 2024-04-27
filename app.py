from flask import Flask, render_template, request
from keras.models import load_model # type: ignore
import numpy as np
import cv2
from PIL import Image
app = Flask(__name__)
#model = load_model('model\saved_image_model.h5')
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction_text='No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction_text='No selected file')
    model_path = 'model/saved_image_model.h5' # save the model path 
    loaded_model = load_model(model_path)
    class_labels = ["NORMAL", "PNEUMONIA"] # Define class labels
    imagefile= request.files['file.filename'] # access input image path and save 
    image_path = "./static/images/" + imagefile.filename
    image_path = imagefile.save(image_path)
    def preprocess_image(image_path):  # Function to preprocess the input image
        img = Image.open(image_path.stream)
        img = cv2.imread(img)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img/255.0
        img = np.reshape(img, (1, 224, 224, 1))
        return img
    def predict_image(model, image_path): # Function to make predictions
        preprocessed_img = preprocess_image(image_path)
        prediction = model.predict(preprocessed_img)
        return (prediction)
    def get_predicted_class(prediction): # Function to get predicted class and probability
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_index]
        probability = prediction[0][predicted_class_index]
        predicted_class_probability = prediction[0][predicted_class_index]
        negative_class_index = 1 - predicted_class_index
        negative_class_probability = prediction[0][negative_class_index]
        print("Positive Probability (Predicted Class):", predicted_class_probability) #Display probabilities
        print("Negative Probability (Opposite Class):", negative_class_probability) 
        return (predicted_class, probability)
    prediction = predict_image(loaded_model, image_path)# Example usage
    predicted_class, probability = get_predicted_class(prediction)
    print("Predicted Class:", predicted_class)
    print("Probability:", probability)
    return render_template('index.html', prediction_text=f'{predicted_class}')

if __name__ == '_main_':
    app.run(debug=True)