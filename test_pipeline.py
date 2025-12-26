import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np

def run_pipeline(
    source_dir="In_Person",
    yolo_path="best.pt",
    resnet_path="severity_model_resnet18.pth",
    output_dir="In_Person_Results",
    class_names=['high', 'low', 'medium'] # Must match training order! Alphabetical usually by ImageFolder
):
    # Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load YOLO
    print("Loading YOLO model...")
    yolo_model = YOLO(yolo_path)

    # 2. Load ResNet
    print("Loading ResNet model...")
    resnet_model = models.resnet18(pretrained=False) # No need to download weights, we load ours
    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_ftrs, len(class_names))
    resnet_model.load_state_dict(torch.load(resnet_path, map_location=device))
    resnet_model = resnet_model.to(device)
    resnet_model.eval()

    # Transforms for ResNet (Must match validation transforms)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Process Images
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [
        p for p in source_path.rglob("*") 
        if p.suffix.lower() in image_extensions
    ]

    print(f"Found {len(image_files)} images in {source_dir}. Starting pipeline...")

    for img_path in tqdm(image_files, desc="Processing"):
        try:
            # Read Image
            original_image = cv2.imread(str(img_path))
            if original_image is None:
                print(f"Failed to load {img_path}")
                continue
            
            # YOLO Inference
            results = yolo_model(original_image, verbose=False)

            for r in results:
                boxes = r.boxes
                
                for box in boxes:
                    # Get Box Coords
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Crop for Severity Check
                    # Ensure within bounds
                    h, w = original_image.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    crop = original_image[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                        
                    # Prepare Crop for ResNet
                    # Convert BGR (OpenCV) to RGB (PIL/Torch)
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(crop_rgb)
                    
                    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
                    
                    # ResNet Inference
                    with torch.no_grad():
                        outputs = resnet_model(input_tensor)
                        _, preds = torch.max(outputs, 1)
                        severity_idx = preds.item()
                        severity_label = class_names[severity_idx]
                        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][severity_idx].item()

                    # Draw on Image
                    # Color based on severity?
                    color = (0, 255, 0) # Green default
                    if severity_label == 'high':
                        color = (0, 0, 255) # Red
                    elif severity_label == 'medium':
                        color = (0, 165, 255) # Orange (BGR)
                    elif severity_label == 'low':
                        color = (0, 255, 255) # Yellow
                        
                    # Draw Box
                    cv2.rectangle(original_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw Label
                    label_text = f"{severity_label} ({confidence:.2f})"
                    # Get text size
                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    # Text background
                    cv2.rectangle(original_image, (x1, y1 - 20), (x1 + tw, y1), color, -1)
                    # Text
                    cv2.putText(original_image, label_text, (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Save Result
            save_name = output_path / img_path.name
            cv2.imwrite(str(save_name), original_image)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    print(f"\nProcessing complete! Check outputs in {output_dir}")

if __name__ == "__main__":
    run_pipeline()
