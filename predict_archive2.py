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

def run_large_batch(
    source_dir="archive-2/image",
    yolo_path="best.pt",
    resnet_path="severity_model_resnet18.pth",
    output_dir="archive2_results",
    class_names=['high', 'low', 'medium'] 
):
    # Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load YOLO
    print("Loading YOLO model...")
    yolo_model = YOLO(yolo_path)

    # 2. Load ResNet
    print("Loading ResNet model...")
    resnet_model = models.resnet18(pretrained=False)
    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_ftrs, len(class_names))
    resnet_model.load_state_dict(torch.load(resnet_path, map_location=device))
    resnet_model = resnet_model.to(device)
    resnet_model.eval()

    # Transforms
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
    # Using iterator for memory efficiency if list is huge, but 11k is fine in list
    image_files = [
        p for p in source_path.rglob("*") 
        if p.suffix.lower() in image_extensions
    ]

    print(f"Found {len(image_files)} images in {source_dir}. Starting large batch processing...")

    for img_path in tqdm(image_files, desc="Batch Processing"):
        try:
            # Read Image
            original_image = cv2.imread(str(img_path))
            if original_image is None:
                continue
            
            # YOLO Inference
            results = yolo_model(original_image, verbose=False)

            has_detection = False
            for r in results:
                boxes = r.boxes
                if len(boxes) > 0:
                    has_detection = True
                
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Crop
                    h, w = original_image.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    crop = original_image[y1:y2, x1:x2]
                    
                    if crop.size == 0:
                        continue
                        
                    # ResNet Inference
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(crop_rgb)
                    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        outputs = resnet_model(input_tensor)
                        _, preds = torch.max(outputs, 1)
                        severity_idx = preds.item()
                        severity_label = class_names[severity_idx]
                        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][severity_idx].item()

                    # Visualization
                    color = (0, 255, 0) 
                    if severity_label == 'high':
                        color = (0, 0, 255) # Red
                    elif severity_label == 'medium':
                        color = (0, 165, 255) # Orange
                    elif severity_label == 'low':
                        color = (0, 255, 255) # Yellow
                        
                    cv2.rectangle(original_image, (x1, y1), (x2, y2), color, 2)
                    
                    label_text = f"{severity_label} ({confidence:.2f})"
                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(original_image, (x1, y1 - 20), (x1 + tw, y1), color, -1)
                    cv2.putText(original_image, label_text, (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Save Result
            # To avoid name collision, use parent folder name in filename
            save_name = output_path / f"{img_path.parent.name}_{img_path.name}"
            # Or just keep original name if unique enough, but safely:
            # save_name = output_path / img_path.name
            
            # Since user provided "archive-2/image", structure is likely flat or random.
            # Let's stick to unique naming to be safe given 11k files.
            
            cv2.imwrite(str(save_name), original_image)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    print(f"\nProcessing complete! Check outputs in {output_dir}")

if __name__ == "__main__":
    run_large_batch()
