import os
import cv2
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

def crop_damages(
    source_dir="dataset_raw_bing",
    model_path="best.pt",
    output_dir="cropped_damages",
    conf_threshold=0.25
):
    """
    Detects damages using YOLOv8 and crops them into individual images.
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Get list of all image files
    source_path = Path(source_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [
        p for p in source_path.rglob("*") 
        if p.suffix.lower() in image_extensions
    ]
    
    print(f"Found {len(image_files)} images in {source_dir}. Starting processing...")

    count = 0
    
    for img_path in tqdm(image_files, desc="Processing Images"):
        try:
            # Run inference
            results = model.predict(source=str(img_path), conf=conf_threshold, verbose=False)
            
            for r in results:
                # Load original image for cropping
                # Note: ultralytics might resize, so we use the original image from the prediction result or reload
                # r.orig_img is the original numpy array
                orig_img = r.orig_img
                
                boxes = r.boxes
                for i, box in enumerate(boxes):
                    # Bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Ensure coordinates are within image bounds
                    h, w = orig_img.shape[:2]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    
                    # Crop
                    crop = orig_img[y1:y2, x1:x2]
                    
                    # Skip empty crops
                    if crop.size == 0:
                        continue
                        
                    # Get class name
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    
                    # Create unique filename: originalName_class_conf_index.jpg
                    # Use parent folder name to avoid collisions if filenames are same across folders
                    parent_name = img_path.parent.name
                    stem = img_path.stem
                    file_name = f"{parent_name}_{stem}_{cls_name}_{box.conf[0]:.2f}_{i}.jpg"
                    
                    # Replace spaces and special chars in filename
                    file_name = file_name.replace(" ", "_").replace("/", "-")
                    
                    save_path = output_path / file_name
                    
                    cv2.imwrite(str(save_path), crop)
                    count += 1
                    
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    print(f"\nProcessing complete!")
    print(f"Total damage crops saved: {count}")
    print(f"Saved to: {output_path.absolute()}")

if __name__ == "__main__":
    crop_damages()
