from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'paddy_disease_model.pkl'

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load model at startup
model = None
encoder = None

def load_model():
    global model, encoder
    try:
        if os.path.exists(MODEL_PATH):
            model, encoder = joblib.load(MODEL_PATH)
            print("✅ Model loaded successfully!")
            return True
        else:
            print("⚠️ Model file not found. Please train the model first.")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_hog_features(image):
    """Extract HOG features from an image"""
    try:
        # Resize image to 128x128
        img = cv2.resize(image, (128, 128))
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Extract HOG features
        hog_features = hog(
            gray, 
            orientations=9, 
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), 
            block_norm='L2-Hys'
        )
        return hog_features
    except Exception as e:
        raise Exception(f"Feature extraction failed: {str(e)}")

def predict_disease(image_path):
    """Predict disease from image path"""
    if model is None or encoder is None:
        raise Exception("Model not loaded. Please train the model first.")
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("Failed to read image file")
    
    # Extract features
    features = extract_hog_features(img)
    
    # Predict
    prediction = model.predict([features])[0]
    probabilities = model.predict_proba([features])[0]
    
    # Get label
    disease_label = encoder.inverse_transform([prediction])[0]
    
    # Get confidence
    confidence = float(max(probabilities) * 100)
    
    # Get all class probabilities
    all_classes = {}
    for idx, class_name in enumerate(encoder.classes_):
        all_classes[class_name] = float(probabilities[idx] * 100)
    
    return {
        'disease': disease_label,
        'confidence': confidence,
        'all_predictions': all_classes
    }

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not_loaded"
    return jsonify({
        'status': 'ok',
        'model_status': model_status
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Check if model is loaded
        if model is None or encoder is None:
            return jsonify({
                'error': 'Model not loaded. Please train the model first using paddy_disease_classification.py'
            }), 503
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only PNG, JPG, and JPEG are allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict
        result = predict_disease(filepath)
        
        # Read image and convert to base64 for display
        with open(filepath, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'prediction': result,
            'image': f"data:image/jpeg;base64,{img_data}"
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get available disease classes"""
    try:
        if encoder is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        classes = encoder.classes_.tolist()
        return jsonify({
            'classes': classes
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Paddy Disease Classification Server...")
    
    # Load model
    model_loaded = load_model()
    
    if not model_loaded:
        print("\n⚠️  WARNING: Model not found!")
        print("Please train the model first by running:")
        print("python paddy_disease_classification.py\n")
    
    # Run server
    app.run(debug=True, host='0.0.0.0', port=5000)