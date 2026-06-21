import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121
from torchvision import transforms
from PIL import Image


@st.cache_resource
def load_model(model_path="./Models/best_densenet121_binary.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize empty DenseNet121
    model = densenet121()
    num_ftrs = model.classifier.in_features 
    
    # Rebuild the custom classifier
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=num_ftrs, out_features=2, bias=True) 
    )
    
    # Load the weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        st.error(f"⚠️ weights file '{model_path}' not found. Please ensure it's in the same folder as this script.")
        return None, device
        
    model = model.to(device)
    model.eval() # Set to evaluation mode
    return model, device

def predict_image(image, model, device, threshold):
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Prepare the image
    image = image.convert("RGB")
    input_tensor = val_transforms(image).unsqueeze(0).to(device) 
    
    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        prob_positive = probabilities[0, 1].item()
        
    predicted_class = 1 if prob_positive >= threshold else 0
    return predicted_class, prob_positive


st.set_page_config(page_title="Knee Osteoarthritis Detector", page_icon="🦴", layout="centered")

st.title("🦴 Knee Osteoarthritis Detector")
st.write("Upload a knee X-ray to predict the likelihood of definite Osteoarthritis (KL Grade ≥ 2).")

# Load model in the background
model, device = load_model("./Models/best_densenet121_binary.pth")

if model is not None:
    # Sidebar for advanced controls
    st.sidebar.header("Advanced Settings")
    threshold = st.sidebar.slider("Diagnostic Threshold", min_value=0.01, max_value=0.99, value=0.55, step=0.01, 
                                  help="Lower the threshold to increase sensitivity (catch more cases). Raise it to increase specificity.")
    
    st.sidebar.info(f"Currently running on: **{device.type.upper()}**")

    # File Uploader
    uploaded_file = st.file_uploader("Choose a Knee X-ray image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # 1. Display the image
        image = Image.open(uploaded_file)
        
        # Center the image nicely
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption='Uploaded X-ray', use_container_width=True)

        # 2. Add a predict button
        if st.button("Run AI Diagnosis", type="primary", use_container_width=True):
            with st.spinner("Analyzing bone textures..."):
                # Run the prediction
                predicted_class, prob_positive = predict_image(image, model, device, threshold)
                
            # 3. Display the Results beautifully
            st.markdown("---")
            st.subheader("Diagnosis Results")
            
            # Progress bar for probability
            st.progress(prob_positive)
            st.write(f"**Confidence (Probability of Disease):** {prob_positive * 100:.2f}%")
            
            # Outcome Message
            if predicted_class == 1:
                st.error("🚨 **Diagnosis: Positive / Definite Osteoarthritis** (KL Grade 2, 3, or 4 detected)")
                st.write("*Recommendation: Please consult an orthopedic specialist for clinical evaluation.*")
            else:
                st.success("✅ **Diagnosis: Negative / Doubtful** (Healthy or KL Grade 1 detected)")
                st.write("*Recommendation: No severe osteoarthritis features detected based on current threshold.*")