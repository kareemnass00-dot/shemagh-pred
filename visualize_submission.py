"""
Visualize submission_dual.csv on test images.
Draws bounding boxes (Head=Blue, Shemagh=Green) and RIGHT PLACE banner.
"""
import os
import csv
import cv2

# Paths
SUB_FILE = "submission_dual.csv"
TEST_DIR = "data/dal-shemagh-detection-challenge/images/test"
OUT_DIR = "visualized_submission_dual"
os.makedirs(OUT_DIR, exist_ok=True)

# Read submission
rows = []
with open(SUB_FILE, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        rows.append(row)

print(f"Loaded {len(rows)} rows from {SUB_FILE}")

for row in rows:
    fname = row[0]
    right_place = int(row[1])
    pred_str = row[2]
    
    img_path = os.path.join(TEST_DIR, fname)
    if not os.path.exists(img_path):
        continue
    
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    # Draw boxes
    if pred_str != '-':
        parts = pred_str.split()
        for i in range(0, len(parts), 6):
            if i + 5 >= len(parts):
                break
            cls = int(parts[i])
            conf = float(parts[i+1])
            cx = float(parts[i+2]) * w
            cy = float(parts[i+3]) * h
            bw = float(parts[i+4]) * w
            bh = float(parts[i+5]) * h
            
            x1 = int(cx - bw/2)
            y1 = int(cy - bh/2)
            x2 = int(cx + bw/2)
            y2 = int(cy + bh/2)
            
            if cls == 0:  # Head
                color = (255, 100, 0)  # Blue
                label = f"Head {conf:.2f}"
            else:  # Shemagh
                color = (0, 200, 0)  # Green
                label = f"Shem {conf:.2f}"
            
            thickness = 2 if conf > 0.25 else 1
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Banner
    banner_h = 40
    banner_color = (0, 180, 0) if right_place == 1 else (0, 0, 200)
    banner_text = f"RIGHT PLACE: {'YES' if right_place == 1 else 'NO'}"
    cv2.rectangle(img, (0, 0), (w, banner_h), banner_color, -1)
    cv2.putText(img, banner_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Save
    out_name = fname.replace('.jpg', '.jpg')
    cv2.imwrite(os.path.join(OUT_DIR, out_name), img)

print(f"Done! Saved {len(rows)} images to {OUT_DIR}/")
