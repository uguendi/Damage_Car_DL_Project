import streamlit as st
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os

# --- Configuration ---
st.set_page_config(page_title="Car Damage Severity Detection")

# --- Model Loading (Cached) ---
@st.cache(allow_output_mutation=True)
def load_models():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # YOLO
    yolo_model = YOLO("best.pt")
    
    # ResNet
    class_names = ['high', 'low', 'medium'] 
    resnet_model = models.resnet18(pretrained=False)
    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_ftrs, len(class_names))
    resnet_model.load_state_dict(torch.load("severity_model_resnet18.pth", map_location=device))
    resnet_model = resnet_model.to(device)
    resnet_model.eval()
    
    return yolo_model, resnet_model, device

yolo_model, resnet_model, device = load_models()

# Transforms
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- UI Layout ---
st.title("Car Damage Severity Detection")
st.markdown("Upload an image of a damaged car to analyze the severity.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Save temp file for YOLO (optional, or pass PIL/numpy directly)
    # YOLO ultralytics handles PIL images well.
    
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Analyze Damage"):
        with st.spinner("Analyzing..."):
            # Convert to OpenCV (BGR)
            original_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # YOLO Inference
            results = yolo_model(original_image, verbose=False)
            
            damage_count = 0
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    damage_count += 1
                    
                    # 1. Bounding Box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Crop
                    h, w = original_image.shape[:2]
                    cx1, cy1 = max(0, x1), max(0, y1)
                    cx2, cy2 = min(w, x2), min(h, y2)
                    crop = original_image[cy1:cy2, cx1:cx2]
                    
                    if crop.size == 0:
                        continue

                    # 2. Severity Classification (ResNet)
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pil_crop = Image.fromarray(crop_rgb)
                    input_tensor = preprocess(pil_crop).unsqueeze(0).to(device)

                    with torch.no_grad():
                        outputs = resnet_model(input_tensor)
                        _, preds = torch.max(outputs, 1)
                        severity_idx = preds.item()
                        class_names = ['high', 'low', 'medium']
                        severity_label = class_names[severity_idx]
                        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][severity_idx].item()
                    
                    # YOLO Class Name
                    yolo_cls_id = int(box.cls[0])
                    part_name = yolo_model.names[yolo_cls_id]

                    # 3. Visualization
                    color = (0, 255, 0) # Green default
                    if severity_label == 'high':
                        color = (0, 0, 255) # Red (BGR)
                    elif severity_label == 'medium':
                        color = (0, 165, 255) # Orange
                    elif severity_label == 'low':
                        color = (0, 255, 255) # Yellow
                    
                    # Draw Box
                    cv2.rectangle(original_image, (x1, y1), (x2, y2), color, 3)
                    
                    # Label Text
                    label_text = f"{part_name} - {severity_label}"
                    
                    # Draw Label Background
                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(original_image, (x1, y1 - 25), (x1 + tw + 10, y1), color, -1)
                    
                    # Draw Text
                    cv2.putText(original_image, label_text, (x1 + 5, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Show Result
            final_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            st.success(f"Analysis Complete! Found {damage_count} damages.")
            st.image(final_image, caption="Analyzed Image with Severity Levels", use_column_width=True)
            
            st.markdown("""
            **Legend:**
            - 🔴 **Red Box:** High Severity
            - 🟠 **Orange Box:** Medium Severity
            - 🟡 **Yellow Box:** Low Severity
            """)
