from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


app = Flask(__name__)
file_path = 'plant_main.h5'
model = load_model(file_path)
model_fruit = load_model("fruits_classification.h5")
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home_page')
def home_page():
    return render_template('index.html')

@app.route('/plant_disease_view')
def plant_disease_view():
    return render_template('plant_disease.html')

@app.route('/fruits_classification_view')
def fruits_classification_view():
    return render_template('fruits_classification.html')
    

@app.route('/predict', methods=['POST'])
def predict():
     # Check if the request contains a file
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    # Load and preprocess the image
    # img_path = '/content/new-plant-diseases-dataset/test/test/AppleCedarRust1.JPG'  # Replace with the path to your image file
    
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = image.load_img(os.path.join(app.config['UPLOAD_FOLDER'], filename), target_size=(150, 150))  # Resize image to match model input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        # img_array /= 255.0  # Normalize pixel values (assuming your model was trained with normalized data)
        class_name = ['Apple___Apple_scab',  'Apple___Black_rot',  'Apple___Cedar_apple_rust',  'Apple___healthy',  
                    'Blueberry___healthy',  'Cherry_(including_sour)___Powdery_mildew',  'Cherry_(including_sour)___healthy',  
                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',  'Corn_(maize)___Common_rust_',  
                    'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot','Grape___Esca_(Black_Measles)',
                    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy','Orange___Haunglongbing_(Citrus_greening)',
                    'Peach___Bacterial_spot','Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy',
                    'Potato___Early_blight','Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy',
                    'Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy','Tomato___Bacterial_spot',
                    'Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot',
                    'Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                    'Tomato___Tomato_mosaic_virus','Tomato___healthy']

        # Perform prediction
        predictions = model.predict(img_array)

        # Interpret predictions
        predicted_class = np.argmax(predictions)
        confidence = predictions[0][predicted_class]

        print("Predicted class:", predicted_class)
        print("Confidence:", confidence)
        plant_disease = class_name[predicted_class]
    return render_template('plant_disease.html', plant_disease=plant_disease)

@app.route('/predict_fruit', methods=['POST'])
def predict_fruit():
     # Check if the request contains a file
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    # Load and preprocess the image
    # img_path = '/content/new-plant-diseases-dataset/test/test/AppleCedarRust1.JPG'  # Replace with the path to your image file
    
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = image.load_img(os.path.join(app.config['UPLOAD_FOLDER'], filename), target_size=(100, 100))  # Resize image to match model input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize pixel values (assuming your model was trained with normalized data)
        fruits_class_dict = {0: 'apple', 1: 'apple_braeburn', 2: 'apple_crimson_snow', 3: 'apple_golden', 4: 'apple_golden', 5: 
                             'apple_golden', 6: 'apple_granny_smith', 7: 'apple_hit', 
                             8: 'apple_pink_lady', 9: 'apple_red', 10: 'apple_red', 11: 'apple_red', 
                             12: 'apple_red_delicios', 13: 'apple_red_yellow', 14: 'apple_rotten', 15: 'cabbage_white', 
                             16: 'carrot', 17: 'cucumber_1', 18: 'cucumber_3', 19: 'eggplant_violet', 20: 'pear_1', 
                             21: 'pear_3', 22: 'zucchini', 23: 'zucchini_dark'}

        # Perform prediction
        predictions = model_fruit.predict(img_array)

        # Interpret predictions
        predicted_class = np.argmax(predictions)
        confidence = predictions[0][predicted_class]

        print("Predicted class:", predicted_class)
        print("Confidence:", confidence)
        predict_fruit_class = fruits_class_dict[predicted_class]
    return render_template('fruits_classification.html', predict_fruit_class=predict_fruit_class)

   



if __name__ == '__main__':
    app.run(debug=True)