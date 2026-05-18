import os
from PIL import Image
from pathlib import Path

DATA_DIR    = r"E:\wfd"           # original extracted data
CROPPED_DIR = r"E:\wfd_cropped"   # cropped images saved here

exts = {".jpg", ".jpeg", ".png", ".bmp"}

print("Starting crop of full dataset...\n")

total_done = 0

for folder in sorted(os.listdir(DATA_DIR)):
    src = os.path.join(DATA_DIR, folder)
    if not os.path.isdir(src):
        continue

    dst = os.path.join(CROPPED_DIR, folder)
    os.makedirs(dst, exist_ok=True)

    images = [f for f in os.listdir(src)
              if Path(f).suffix.lower() in exts]

    done = 0
    for fname in images:
        try:
            img = Image.open(os.path.join(src, fname)).convert("RGB")
            w, h = img.size
            # keep left half only (real grain, remove white masks)
            cropped = img.crop((0, 0, w // 2, h))
            cropped.save(os.path.join(dst, fname))
            done += 1
        except Exception as e:
            pass

        if done % 5000 == 0 and done > 0:
            print(f"  {folder}: {done}/{len(images)}...")

    total_done += done
    print(f"  ✓  {folder:25s} → {done} images cropped")

print(f"\nDone! Total cropped: {total_done} images")
print(f"Saved to: {CROPPED_DIR}")