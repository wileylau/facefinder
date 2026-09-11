import os
import shutil
import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1. Config
REFERENCE_IMAGE = r"C:\Users\Administrator\Documents\Code\Finding Toby\arnold.jpeg"
SOURCE_FOLDER = r"C:\Users\Administrator\Documents\Code\Finding Toby\source"
OUTPUT_FOLDER = r"C:\Users\Administrator\Documents\Code\Finding Toby\found_arnold"
NUM_WORKERS = 8  # Number of parallel CPU threads

def load_image(path):
    try:
        pil_img = Image.open(path).convert('RGB')
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        return None

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 2. Collect ALL image files recursively
all_files = []
for root, _, files in os.walk(SOURCE_FOLDER):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
            all_files.append(os.path.join(root, f))

print(f"📁 Found {len(all_files)} images across '{SOURCE_FOLDER}'")

if not all_files:
    print("❌ No images found. Stopping.")
    exit()

# 3. Initialize Model
print("\n⏳ Loading Face Detection Model...")
app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 4. Load Reference Photo
ref_img = load_image(REFERENCE_IMAGE)
if ref_img is None:
    print(f"❌ Could not load reference image at '{REFERENCE_IMAGE}'")
    exit()

ref_faces = app.get(ref_img)
if not ref_faces:
    print("❌ No face detected in reference photo!")
    exit()

toby_embedding = ref_faces[0].embedding
print("✅ Reference face loaded successfully!\n")

# 5. Worker function for each photo
def process_single_photo(file_path):
    img = load_image(file_path)
    if img is None:
        return None
    
    faces = app.get(img)
    filename = os.path.basename(file_path)
    
    for face in faces:
        sim = np.dot(toby_embedding, face.embedding) / (
            np.linalg.norm(toby_embedding) * np.linalg.norm(face.embedding)
        )
        if sim > 0.40:
            dest_path = os.path.join(OUTPUT_FOLDER, filename)
            # Handle duplicate filenames across subfolders
            if os.path.exists(dest_path):
                dest_path = os.path.join(OUTPUT_FOLDER, f"{hash(file_path)}_{filename}")
                
            shutil.copy(file_path, dest_path)
            return (filename, sim)
    return None

# 6. Run multithreaded pool
print(f"⚡ Processing with {NUM_WORKERS} threads...")
matched_count = 0
processed_count = 0

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {executor.submit(process_single_photo, fp): fp for fp in all_files}
    
    for future in as_completed(futures):
        processed_count += 1
        result = future.result()
        
        if result:
            matched_count += 1
            filename, sim = result
            print(f"[{processed_count}/{len(all_files)}] 👉 [MATCH!] {filename} (Similarity: {sim:.2f})")
        else:
            # Show progress every 50 images
            if processed_count % 50 == 0 or processed_count == len(all_files):
                print(f"Progress: [{processed_count}/{len(all_files)}] photos scanned...")

print(f"\n🎉 Done! Scanned {len(all_files)} photos and found {matched_count} matches of Toby.")