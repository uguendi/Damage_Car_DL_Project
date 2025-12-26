import gradio as gr
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --- 1. Model Setup ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load YOLO
print("Loading YOLO model...")
yolo_model = YOLO("best.pt")

# Load ResNet
print("Loading ResNet model...")
class_names = ['high', 'low', 'medium'] # Must match training
resnet_model = models.resnet18(pretrained=False)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, len(class_names))
resnet_model.load_state_dict(torch.load("severity_model_resnet18.pth", map_location=device))
resnet_model = resnet_model.to(device)
resnet_model.eval()

# Transforms
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 2. Inference Function ---
def process_image(input_img):
    """
    Takes a PIL Image or numpy array from Gradio, 
    detects damages with YOLO, classifiers with ResNet, 
    and returns annotated image.
    """
    if input_img is None:
        return None

    # Convert to OpenCV format (BGR)
    # Gradio passes RGB numpy array by default
    original_image = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    
    # YOLO Inference
    results = yolo_model(original_image, verbose=False)
    
    detections = [] # To store list of (severity, label) for summary text

    for r in results:
        boxes = r.boxes
        for box in boxes:
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
                severity_label = class_names[severity_idx]
                confidence = torch.nn.functional.softmax(outputs, dim=1)[0][severity_idx].item()
            
            # YOLO Class Name
            yolo_cls_id = int(box.cls[0])
            part_name = yolo_model.names[yolo_cls_id]

            # 3. Visualization
            # Color Coding
            color = (0, 255, 0) # Green default
            if severity_label == 'high':
                color = (0, 0, 255) # Red (BGR)
            elif severity_label == 'medium':
                color = (0, 165, 255) # Orange
            elif severity_label == 'low':
                color = (0, 255, 255) # Yellow
            
            # Draw Box
            cv2.rectangle(original_image, (x1, y1), (x2, y2), color, 3)
            
            # Label Text: "Scratch - High"
            label_text = f"{part_name.upper()} - {severity_label.upper()}"
            
            detections.append(label_text)

            # Draw Label Background
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(original_image, (x1, y1 - 25), (x1 + tw + 10, y1), color, -1)
            
            # Draw Text
            cv2.putText(original_image, label_text, (x1 + 5, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Convert back to RGB for Gradio display
    final_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Summary String
    summary = "Detected Damages:\n" + "\n".join(detections) if detections else "No damages detected."

    return final_image, summary

# --- 3. UI Setup ---
demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(label="Upload Car Image"),
    outputs=[
        gr.Image(label="Analyzed Image"),
        gr.Textbox(label="Detection Summary")
    ],
    title="Car Damage Severity Detection System",
    description="Upload an image of a damaged car. The system will detect the damage type (using YOLOv8) and classify its severity (High/Medium/Low using ResNet18).",
    theme="default"
)

if __name__ == "__main__":
    demo.launch(share=False)
