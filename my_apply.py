import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
from torch import nn
import numpy as np
import cv2

# Emotion categories
class_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Image processing setup
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Change grayscale image to 3 channels
    transforms.Resize((48, 48)),  # Resize to FER-2013 dimensions
    transforms.ToTensor(),  # Convert the image to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
])

# Create the Emotion Detection Model
class EmotionDetectionModel(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionDetectionModel, self).__init__()
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

# Load the trained model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionDetectionModel(num_classes=7).to(device)
    model.load_state_dict(torch.load("mobilenet_emotion_model.pth", map_location=device))
    model.eval()
    return model, device

model, device = load_model()

# Function to find faces with OpenCV
@st.cache_data
def detect_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    image_cv = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0

# Streamlit user interface
st.title("🎭 Emotion Detection App")
st.write("Upload an image, and the model will predict the emotion!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # Convert the uploaded image to RGB
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Check for face detection
    if detect_face(image):
        # Prepare the image for prediction
        image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
        
        # Make a prediction
        with torch.no_grad():
            output = model(image_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            emotion = class_labels[predicted_class]
        
        st.write(f"Predicted Emotion:{emotion.capitalize()}")
    else:
        st.write(" No face detected. Please upload a valid face image.")