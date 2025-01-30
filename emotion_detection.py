import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model_path = "best_model.h5"
model = tf.keras.models.load_model(model_path)

# Class labels
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Function to preprocess an image
def preprocess_image(image):
    image = image.resize((150, 150))  # Resize to match model input size
    image = img_to_array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("Facial Expression Recognition using CNN")

st.write("Upload an image and the model will predict the facial expression.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.write("Classifying...")
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    
    st.write(f"Prediction: {class_labels[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}")
    
    # Plot confidence scores
    plt.figure(figsize=(6, 4))
    sns.barplot(x=class_labels, y=predictions[0])
    plt.title("Prediction Confidence Scores")
    plt.xlabel("Expression")
    plt.ylabel("Confidence")
    st.pyplot(plt)
