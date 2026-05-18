import os, json
import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────
SAMPLE_DIR    = r"E:\Sample Data Cropped"
OUTPUT_CSV    = r"C:\Users\hii\Downloads\Wheat Detection Model\labels.csv"
PROGRESS_FILE = r"C:\Users\hii\Downloads\Wheat Detection Model\.progress.json"

LABEL_MAP = {
    "1": "healthy",
    "2": "broken",
    "3": "fungus_damage",
    "4": "insect_damage",
    "5": "foreign_matter",
    "6": "black_tip",
    "7": "shriveled",
    "s": "skip"
}
# ─────────────────────────────────────────────────────────────

exts = {".jpg", ".jpeg", ".png", ".bmp"}

def load_images():
    all_images = []
    for folder in sorted(os.listdir(SAMPLE_DIR)):
        fp = os.path.join(SAMPLE_DIR, folder)
        if not os.path.isdir(fp):
            continue
        for f in sorted(os.listdir(fp)):
            if Path(f).suffix.lower() in exts:
                all_images.append((os.path.join(fp, f), folder))
    return all_images

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"index": 0, "labels": {}}

def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)

def save_csv(progress, all_images):
    rows = []
    for path, folder in all_images:
        label = progress["labels"].get(path)
        if label:
            rows.append({
                "filepath": path,
                "folder":   folder,
                "filename": os.path.basename(path),
                "label":    label
            })
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    return len(rows)

class Labeler:
    def __init__(self, root, images, progress):
        self.root     = root
        self.images   = images
        self.progress = progress
        self.idx      = progress["index"]
        self.total    = len(images)

        root.title("Wheat Grain Labeler")
        root.configure(bg="#111111")
        root.geometry("750x650")

        # ── top progress info ──
        self.info = tk.StringVar()
        tk.Label(root, textvariable=self.info,
                 bg="#111111", fg="#c2c0b6",
                 font=("Helvetica", 13)).pack(pady=(14, 2))

        # ── progress bar ──
        self.progress_frame = tk.Frame(root, bg="#111111")
        self.progress_frame.pack(fill="x", padx=40, pady=4)
        self.progress_canvas = tk.Canvas(self.progress_frame, height=8,
                                         bg="#2c2c2a", highlightthickness=0)
        self.progress_canvas.pack(fill="x")

        # ── folder badge ──
        self.badge = tk.StringVar()
        tk.Label(root, textvariable=self.badge,
                 bg="#1D9E75", fg="#ffffff",
                 font=("Helvetica", 11, "bold"),
                 padx=12, pady=5).pack(pady=(4, 0))

        # image canvas
        self.canvas = tk.Canvas(root, width=520, height=400,
                                bg="#000000", highlightthickness=0)
        self.canvas.pack(pady=10)

        # status
        self.status = tk.StringVar(value="Press a key to label this image")
        tk.Label(root, textvariable=self.status,
                 bg="#111111", fg="#888780",
                 font=("Helvetica", 11)).pack()

        # key legend
        legend1 = "1=healthy   2=broken   3=fungus_damage   4=insect_damage"
        legend2 = "5=foreign_matter   6=black_tip   7=shriveled   s=skip   b=back   q=quit"
        tk.Label(root, text=legend1,
                 bg="#111111", fg="#5F5E5A",
                 font=("Helvetica", 10)).pack(pady=(6, 0))
        tk.Label(root, text=legend2,
                 bg="#111111", fg="#5F5E5A",
                 font=("Helvetica", 10)).pack(pady=(2, 8))

        root.bind("<Key>", self.on_key)
        self.show()

    def update_progress_bar(self):
        self.progress_canvas.update_idletasks()
        w = self.progress_canvas.winfo_width()
        if w < 2:
            return
        filled = int((self.idx / self.total) * w)
        self.progress_canvas.delete("all")
        self.progress_canvas.create_rectangle(0, 0, filled, 8,
                                               fill="#1D9E75", outline="")

    def show(self):
        if self.idx >= self.total:
            self.finish()
            return

        path, folder = self.images[self.idx]
        done    = len([v for v in self.progress["labels"].values() if v != "skip"])
        skipped = len([v for v in self.progress["labels"].values() if v == "skip"])

        self.info.set(
            f"Image {self.idx + 1} of {self.total}   |   "
            f"Labeled: {done}   Skipped: {skipped}   "
            f"Remaining: {self.total - self.idx}"
        )
        self.badge.set(f"  Current Folder: {folder}  ")
        self.update_progress_bar()

        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((520, 400), Image.LANCZOS)
            self.tkimg = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            x = (520 - img.width)  // 2
            y = (400 - img.height) // 2
            self.canvas.create_image(x, y, anchor="nw", image=self.tkimg)

            existing = self.progress["labels"].get(path)
            if existing:
                self.status.set(
                    f"Already labeled as: '{existing}'  —  press new key to change"
                )
            else:
                self.status.set("Press a key to label this image")

        except Exception as e:
            self.canvas.delete("all")
            self.canvas.create_text(260, 200,
                                    text=f"Cannot open image:\n{e}",
                                    fill="#E24B4A",
                                    font=("Helvetica", 11))

    def on_key(self, event):
        k = event.char.lower()

        # quit and save
        if k == "q":
            self.finish()
            return

        # go back
        if k == "b":
            if self.idx > 0:
                self.idx -= 1
                prev_path = self.images[self.idx][0]
                removed   = self.progress["labels"].pop(prev_path, None)
                self.progress["index"] = self.idx
                save_progress(self.progress)
                self.status.set(f"Went back — removed label '{removed}'")
                self.show()
            else:
                self.status.set("Already at the first image!")
            return

        # label
        if k in LABEL_MAP:
            label = LABEL_MAP[k]
            path  = self.images[self.idx][0]
            self.progress["labels"][path] = label
            self.idx += 1
            self.progress["index"] = self.idx
            save_progress(self.progress)

            # auto save CSV every 20 images
            if self.idx % 20 == 0:
                save_csv(self.progress, self.images)
                print(f"Auto saved at image {self.idx}")

            self.show()

    def finish(self):
        count = save_csv(self.progress, self.images)
        save_progress(self.progress)
        messagebox.showinfo(
            "Progress Saved!",
            f"✓ {count} labels saved to labels.csv\n\n"
            f"You can close and resume anytime.\n"
            f"Just run the script again to continue."
        )
        self.root.destroy()


# MAIN 
if __name__ == "__main__":
    print("Loading images...")
    all_images = load_images()
    print(f"Found {len(all_images)} images across all folders\n")

    for folder in sorted(set(f for _, f in all_images)):
        count = sum(1 for _, fl in all_images if fl == folder)
        print(f"  {folder:25s} → {count} images")

    print()
    progress = load_progress()

    if progress["index"] > 0:
        print(f"Resuming from image {progress['index']} "
              f"({len(progress['labels'])} already labeled)")
    else:
        print("Starting fresh labeling session")

    print("\nOpening labeling window...")
    root = tk.Tk()
    app  = Labeler(root, all_images, progress)
    root.mainloop()