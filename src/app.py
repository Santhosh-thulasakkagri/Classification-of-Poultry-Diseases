import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Set up Flask
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model
model = load_model('poultry_diseases_detection.keras')

# Replace with your actual class labels (in the order your model was trained)
labels = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella'] 

def get_model_predictions(model, image_path):
    img = load_img(image_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  
    predictions = model.predict(x, verbose=0)
    return labels[np.argmax(predictions)]

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            prediction = get_model_predictions(model, file_path)
            return render_template('result.html', user_image=file_path, prediction_text=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
