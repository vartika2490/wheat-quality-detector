
import os, shutil

data_dir = r"E:\wfd_cropped"

# correct mapping based on what you confirmed:
# foreign_matter  → black_tip      (row 1 = black tip grains)
# healthy         → healthy        (row 2 = healthy) already correct
# temp_broken     → insect_damage  (row 3 = holes = insect damage)
# temp_fungus_damage → fungus_damage (row 4 = white patches = fungus)
# temp_fungus_damage2 → shriveled  (row 5 = pink/shriveled)
# temp_healthy    → DELETE         (row 6 = duplicate healthy)
# temp_shriveled  → broken         (row 7 = broken pieces)
# foreign_matter  → DELETE         (not in dataset)

rename_map = {
    "foreign_matter"     : "black_tip",
    "temp_broken"        : "insect_damage",
    "temp_fungus_damage" : "fungus_damage",
    "temp_fungus_damage2": "shriveled",
    "temp_shriveled"     : "broken",
}

delete_list = ["temp_healthy"]

print("Renaming folders...")
for old, new in rename_map.items():
    old_path = os.path.join(data_dir, old)
    new_path = os.path.join(data_dir, new)
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f"  Renamed: {old} → {new}")
    else:
        print(f"  Not found: {old}")

print("\nDeleting duplicate folders...")
for folder in delete_list:
    path = os.path.join(data_dir, folder)
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"  Deleted: {folder}")

print("\nFinal folders:")
for f in sorted(os.listdir(data_dir)):
    full = os.path.join(data_dir, f)
    if os.path.isdir(full):
        count = len(os.listdir(full))
        print(f"  {f:25s} → {count} images")