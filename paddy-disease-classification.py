# paddy_disease_classification.py
import os
import time
import argparse
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from skimage.feature import hog

# ==============================
# 1️⃣ LOAD DATASET
# ==============================
# Folder structure example:
# dataset/
# ├── bacterial_leaf_blight/
# ├── brown_spot/
# └── leaf_smut/

DATA_DIR = "dataset"

def load_dataset(data_dir, max_images=None):
    """Load images, labels and paths from dataset folder.

    Skips files that cannot be read and prints a warning with the path.
    Returns: (images_array, labels_array, paths_list)
    """
    images = []
    labels = []
    paths = []
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    # First collect all candidate image paths with labels
    all_items = []  # list of (img_path, label_name)
    for label_name in os.listdir(data_dir):
        folder = os.path.join(data_dir, label_name)
        if not os.path.isdir(folder):
            continue
        for file in os.listdir(folder):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(folder, file)
                all_items.append((img_path, label_name))

    if len(all_items) == 0:
        return np.array(images), np.array(labels), paths

    # If limiting, pick a random sample across all classes to avoid class-only selection
    if max_images is not None and max_images < len(all_items):
        import random
        random.seed(42)
        sampled = random.sample(all_items, max_images)
    else:
        sampled = all_items

    # Load sampled images
    for idx, (img_path, label_name) in enumerate(sampled, start=1):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to read image, skipping: {img_path}")
            continue
        try:
            img = cv2.resize(img, (128, 128))
        except Exception as e:
            print(f"Warning: Failed to resize image {img_path}: {e}. Skipping.")
            continue
        images.append(img)
        labels.append(label_name)
        paths.append(img_path)
        if idx % 200 == 0:
            print(f"Loaded {idx} images...")

    if max_images is not None:
        print(f"Reached max_images={max_images}, stopping load.")

    return np.array(images), np.array(labels), paths

# Top-level execution removed. Use the main() function below to run the pipeline.

# ==============================
# 2️⃣ FEATURE EXTRACTION
# ==============================
# Using HOG (Histogram of Oriented Gradients) for feature extraction
def extract_features(images, paths=None):
    """Extract HOG features for a list/array of images.

    This function computes a safe HOG feature length using a blank image so
    that if HOG fails on a particular image we can append a fallback vector
    of the same length (zeros) and continue processing without crashing.
    """
    # Precompute a safe feature length with a blank image of the target size
    sample = np.zeros((128, 128), dtype=np.uint8)
    sample_feat = hog(sample, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), block_norm='L2-Hys')
    feat_len = len(sample_feat)

    features = []
    for idx, img in enumerate(images):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), block_norm='L2-Hys')
            features.append(hog_features)
        except Exception as e:
            src = paths[idx] if paths and idx < len(paths) else f"index {idx}"
            print(f"Warning: HOG failed for {src}: {e}. Using zero-vector fallback.")
            features.append(np.zeros(feat_len))
    return np.array(features)

# ==============================
# 7️⃣ PREDICT NEW IMAGE
# ==============================
def predict_image(image_path):
    model, encoder = joblib.load("paddy_disease_model.pkl")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found or cannot be read: {image_path}")
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys')
    pred = model.predict([hog_features])[0]
    label = encoder.inverse_transform([pred])[0]
    return label
def main(max_images=None):
    try:
        print("Loading dataset...")
        X, y, paths = load_dataset(DATA_DIR, max_images=max_images)
        print(f"Loaded {len(X)} images.")

        if len(X) == 0:
            print("No images found in the dataset. Please add images under the dataset folder and retry.")
            return

        # ==============================
        # 2️⃣ FEATURE EXTRACTION
        # ==============================
        print("Extracting features...")
        X_features = extract_features(X, paths)

        # ==============================
        # 3️⃣ ENCODE LABELS
        # ==============================
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)

        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_info = dict(zip(unique, counts))
        print("Class distribution:")
        for k, v in class_info.items():
            print(f"  {k}: {v}")
        if len(unique) < 2:
            print("Need at least 2 classes to train. Exiting.")
            return

        # ==============================
        # 4️⃣ SPLIT DATA & TRAIN MODEL
        # ==============================
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y_encoded, test_size=0.2, random_state=42
        )

        model = SVC(kernel='linear', probability=True)
        print("Training model...")
        model.fit(X_train, y_train)

        # ==============================
        # 5️⃣ EVALUATE MODEL
        # ==============================
        y_pred = model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=encoder.classes_))

        # ==============================
        # 6️⃣ SAVE MODEL
        # ==============================
        joblib.dump((model, encoder), "paddy_disease_model.pkl")
        print("Model saved as paddy_disease_model.pkl")
    except Exception as e:
        print(f"Error during execution: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train paddy disease classifier")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images to load (for testing)")
    parser.add_argument("--predict", type=str, default=None, help="Path to an image to run prediction on and print the label")
    # Use parse_known_args so accidental extra words (for example when running from some shells)
    # don't cause the script to exit with an 'unrecognized arguments' error.
    args, unknown = parser.parse_known_args()
    if args.predict:
        # If user supplies --predict, just run prediction and exit
        try:
            label = predict_image(args.predict)
            print(label)
        except Exception as e:
            print(f"Prediction failed: {e}")
    else:
        main(max_images=args.max_images)

# The pipeline is executed inside main(); remove the duplicate top-level pipeline code.

# Example use:
# print(predict_image("test_leaf.jpg"))

# Example use:
# print(predict_image("test_leaf.jpg"))
