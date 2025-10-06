# fixed_medical_api.py - Clean version without indentation errors
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the fixed ResNet50 model
model = tf.keras.models.load_model('models/resnet50_pneumonia_fixed.h5')
print("✅ Loaded ResNet50 model successfully!")

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess X-ray image for prediction"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def home():
    return '''
    <html>
        <head>
            <title>Pneumonia Detection AI</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .upload-box { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
                .result { background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 5px; }
                .pneumonia { background: #ffe6e6; border-left: 4px solid #dc3545; }
                .normal { background: #e6ffe6; border-left: 4px solid #28a745; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏥 Pneumonia Detection AI</h1>
                <p>Upload a chest X-ray image to check for pneumonia</p>
                
                <form method="post" action="/predict" enctype="multipart/form-data">
                    <div class="upload-box">
                        <input type="file" name="file" accept=".png,.jpg,.jpeg" required>
                        <br><br>
                        <input type="submit" value="Analyze X-Ray" style="padding: 10px 20px;">
                    </div>
                </form>
                
                <div class="disclaimer">
                    <h3>⚠️ Medical Disclaimer</h3>
                    <p>This AI tool is for assistance only. Always consult healthcare professionals for medical diagnosis.</p>
                </div>
            </div>
        </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess and predict
            processed_img = preprocess_image(filepath)
            prediction = model.predict(processed_img)[0][0]
            
            # Interpret results
            if prediction > 0.5:
                result = "PNEUMONIA"
                confidence = float(prediction)
                recommendation = "⚠️ Potential pneumonia detected. Please consult a healthcare professional immediately."
                result_class = "pneumonia"
            else:
                result = "NORMAL" 
                confidence = float(1 - prediction)
                recommendation = "✅ No signs of pneumonia detected. Continue regular health checkups."
                result_class = "normal"
            
            # Clean up
            os.remove(filepath)
            
            return f'''
            <html>
                <head>
                    <title>Results - Pneumonia Detection</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; }}
                        .container {{ max-width: 800px; margin: 0 auto; }}
                        .result {{ padding: 20px; margin: 20px 0; border-radius: 5px; }}
                        .pneumonia {{ background: #ffe6e6; border-left: 4px solid #dc3545; }}
                        .normal {{ background: #e6ffe6; border-left: 4px solid #28a745; }}
                        .back-btn {{ padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>🔍 Analysis Results</h1>
                        <div class="result {result_class}">
                            <h2>Prediction: {result}</h2>
                            <p><strong>Confidence:</strong> {confidence:.1%}</p>
                            <p><strong>Recommendation:</strong> {recommendation}</p>
                        </div>
                        <button class="back-btn" onclick="window.history.back()">Analyze Another X-Ray</button>
                        
                        <div class="disclaimer" style="margin-top: 30px;">
                            <h3>⚠️ Medical Disclaimer</h3>
                            <p>This AI tool is for assistance only. Always consult healthcare professionals for medical diagnosis.</p>
                        </div>
                    </div>
                </body>
            </html>
            '''
        
        return jsonify({'error': 'Invalid file type'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'Pneumonia Detection API'})

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    print("🚀 Pneumonia Detection API starting...")
    print("📍 Access at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)