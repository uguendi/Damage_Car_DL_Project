import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np

def label_cropped_dataset(
    source_dir="cropped_damages",
    resnet_path="severity_model_resnet18.pth",
    output_dir="cropped_damages_results",
    class_names=['high', 'low', 'medium'] # Must match training order
):
    # Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load ResNet
    print("Loading ResNet model...")
    resnet_model = models.resnet18(pretrained=False)
    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_ftrs, len(class_names))
    resnet_model.load_state_dict(torch.load(resnet_path, map_location=device))
    resnet_model = resnet_model.to(device)
    resnet_model.eval()

    # Transforms (Match validation)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Setup Paths
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [
        p for p in source_path.rglob("*") 
        if p.suffix.lower() in image_extensions
    ]

    print(f"Found {len(image_files)} images in {source_dir}. Starting labeling...")

    for img_path in tqdm(image_files, desc="Labeling"):
        try:
            # Read Image (OpenCV)
            original_image = cv2.imread(str(img_path))
            if original_image is None:
                continue

            # Convert to PIL for Torch
            image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image_rgb)

            # Inference
            input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = resnet_model(input_tensor)
                _, preds = torch.max(outputs, 1)
                severity_idx = preds.item()
                severity_label = class_names[severity_idx]
                confidence = torch.nn.functional.softmax(outputs, dim=1)[0][severity_idx].item()

            # Colors
            color = (0, 255, 0) # Default Green
            if severity_label == 'high':
                color = (0, 0, 255) # Red
            elif severity_label == 'medium':
                color = (0, 165, 255) # Orange (BGR)
            elif severity_label == 'low':
                color = (0, 255, 255) # Yellow

            # Draw Annotation
            h, w = original_image.shape[:2]
            
            # Thick Border
            cv2.rectangle(original_image, (0, 0), (w-1, h-1), color, 4)
            
            # Text Label with Background
            label_text = f"{severity_label} ({confidence:.2f})"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            
            # Ensure text fits inside or put at top left
            cv2.rectangle(original_image, (0, 0), (tw + 4, th + 8), color, -1)
            cv2.putText(original_image, label_text, (2, 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Save
            save_name = output_path / img_path.name
            cv2.imwrite(str(save_name), original_image)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    print(f"\nLabeling complete! Check outputs in {output_dir}")

if __name__ == "__main__":
    label_cropped_dataset()
