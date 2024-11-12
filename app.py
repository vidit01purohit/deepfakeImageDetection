import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import pickle
from huggingface_hub import snapshot_download
from sklearn.preprocessing import LabelEncoder
import tqdm

# Set the page configuration
st.set_page_config(page_title="Deepfake Detection", layout="centered")

# Constants
IMAGE_SIZE = 224
BATCH_SIZE = 32
MODEL_CACHE_DIR = 'models_cache'  # Directory to cache models

# Function to load models with caching and error handling
def load_models():
    # Ensure the cache directory exists
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    try:
        # Load ResNet50 model
        resnet50_model_path = os.path.join(MODEL_CACHE_DIR, "resnet50")
        if not os.path.exists(resnet50_model_path):
            st.write("\nDownloading ResNet50 model\n")
            local_dir_resnet50 = snapshot_download(repo_id="vidit01purohit/resnet50_model_for_deepfakeImage")
            resnet50_model_path = os.path.join(local_dir_resnet50, "resnet50_model_for_deepfakeImage")
        else:
            st.write("\nLoading cached ResNet50 model\n")
        resnet50_model = tf.saved_model.load(resnet50_model_path)

        # Load VGG16 model
        vgg16_model_path = os.path.join(MODEL_CACHE_DIR, "vgg16")
        if not os.path.exists(vgg16_model_path):
            st.write("\nDownloading VGG16 model\n")
            local_dir_vgg16 = snapshot_download(repo_id="vidit01purohit/vgg16_model_for_deepfakeImage")
            vgg16_model_path = os.path.join(local_dir_vgg16, "vgg16_model_for_deepfakeImage")
        else:
            st.write("\nLoading cached VGG16 model\n")
        vgg16_model = tf.saved_model.load(vgg16_model_path)

        # Load InceptionV3 model
        inceptionv3_model_path = os.path.join(MODEL_CACHE_DIR, "inceptionv3")
        if not os.path.exists(inceptionv3_model_path):
            st.write("\nDownloading InceptionV3 model\n")
            local_dir_inceptionv3 = snapshot_download(repo_id="vidit01purohit/inceptionv3_model_for_deepfakeImage")
            inceptionv3_model_path = os.path.join(local_dir_inceptionv3, "inceptionv3_model_for_deepfakeImage")
        else:
            st.write("\nLoading cached InceptionV3 model\n")
        inceptionv3_model = tf.saved_model.load(inceptionv3_model_path)

        # Load Label Encoder (Cache locally as well)
        try:
            with open('le.pkl', 'rb') as file:
                le = pickle.load(file)
            st.write("\nLoading Label Encoder Successful\n")
        except FileNotFoundError:
            le = None
            st.write("\nError loading Label Encoder\n")
        
        return resnet50_model, vgg16_model, inceptionv3_model, le
    except Exception as e:
        st.error(f"Error while loading models: {e}")
        return None, None, None, None

resnet50_model, vgg16_model, inceptionv3_model, le = load_models()

# Function to classify image using majority voting across three models
def classify_image_voting(image, resnet50_model, vgg16_model, inceptionv3_model, le):
    try:
        # Preprocess image
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0
        img = np.expand_dims(img, axis=0).astype(np.float32)
        img_batch = np.repeat(img, BATCH_SIZE, axis=0)

        # Get predictions from each model
        resnet50_prediction = resnet50_model.signatures['serving_default'](inputs=tf.constant(img_batch))
        vgg16_prediction = vgg16_model.signatures['serving_default'](inputs=tf.constant(img_batch))
        inceptionv3_prediction = inceptionv3_model.signatures['serving_default'](inputs=tf.constant(img_batch))

        # Extract the prediction for the first image in the batch
        resnet50_prob = resnet50_prediction['output_0'][0][1].numpy()
        vgg16_prob = vgg16_prediction['output_0'][0][1].numpy()
        inceptionv3_prob = inceptionv3_prediction['output_0'][0][1].numpy()

        # Combine predictions using majority voting
        average_prob = (resnet50_prob + vgg16_prob + inceptionv3_prob) / 3
        predicted_class = 1 if average_prob > 0.5 else 0

        return predicted_class, average_prob
    except Exception as e:
        st.error(f"Error during image classification: {e}")
        return None, None

# Streamlit UI
st.title("Deepfake Image Detection")
st.markdown("Upload a `.jpg` image to check if it's a **deepfake** or **real**.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    try:
        # Read and preprocess the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Classify the image
        with st.spinner("Analyzing image..."):
            if resnet50_model is not None and vgg16_model is not None and inceptionv3_model is not None:
                predicted_class, average_prob = classify_image_voting(image, resnet50_model, vgg16_model, inceptionv3_model, le)
            else:
                st.error("Models are not loaded correctly. Please try again.")
                predicted_class, average_prob = None, None

        # Display results
        if predicted_class is not None:
            result = "Deepfake" if predicted_class == 1 else "Real"
            st.write(f"**Prediction:** {result}")
            st.write(f"**Confidence Level:** {average_prob * 100:.2f}%")
    except Exception as e:
        st.error(f"Error processing the image: {e}")
