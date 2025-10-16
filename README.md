# Paddy Disease Classification Website

## Overview
This project is a fullstack web application for classifying paddy (rice) leaf diseases using image analysis and machine learning. Users can upload images of rice leaves, and the system will predict the disease type using a trained machine learning model.

## Features
- Upload rice leaf images (PNG, JPG, JPEG, max 16MB)
- Automated disease prediction using a trained SVM model
- Beautiful, modern web interface
- Displays confidence and probabilities for all classes
- REST API endpoints for prediction and health check

## Disease Classes
- Bacterial Leaf Blight
- Brown Spot
- Leaf Smut
- Healthy

## How It Works
1. **Model Training**: The script `paddy-disease-classification.py` loads images from the `dataset/` folder, extracts HOG features, trains an SVM classifier, and saves the model as `paddy_disease_model.pkl`.
2. **Web Application**: The Flask app (`app.py`) loads the trained model and serves the website. Users upload images, which are processed and classified by the model.
3. **Frontend**: The `templates/index.html` file provides a drag-and-drop interface for uploading images and viewing results.

## File Structure
```
app.py                      # Flask web server
paddy-disease-classification.py  # Model training script
requirements.txt            # Python dependencies
README.md                   # Project documentation
/templates/index.html       # Web UI
/dataset/                   # Training images (organized by class)
/uploads/                   # Temporary uploaded images
```

## Setup Instructions
1. **Install Python (recommended: 3.10+)**
2. **Create a virtual environment**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```
4. **Prepare the dataset**
   - Place images in `dataset/` subfolders named after each class (e.g., `bacterial_leaf_blight`, `brown_spot`, `leaf_smut`, `healthy`).
5. **Train the model**
   ```powershell
   python paddy-disease-classification.py
   ```
   - This will create `paddy_disease_model.pkl`.
6. **Run the web server**
   ```powershell
   python app.py
   ```
7. **Open the website**
   - Go to [http://localhost:5000](http://localhost:5000) in your browser.

## API Endpoints
- `GET /api/health` — Check server and model status
- `POST /api/predict` — Upload an image and get prediction
- `GET /api/classes` — List available disease classes

## Requirements
See `requirements.txt` for all dependencies:
- Flask
- flask-cors
- opencv-python
- numpy
- scikit-learn
- scikit-image
- joblib
- Werkzeug

## Credits
- Developed using Python, Flask, scikit-learn, OpenCV, and modern web technologies.

## License
This project is for educational and research purposes.
