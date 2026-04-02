import os
import shutil
import pandas as pd

# Paths
BASE_DIR = "data"
IMG_DIR = os.path.join(BASE_DIR, "images")
CSV_PATH = os.path.join(BASE_DIR, "labels.csv")

# Load labels
df = pd.read_csv(CSV_PATH)

# Column names check (HAM10000 usually has these)
# image_id | dx
print(df.head())

# Create class folders
classes = df["dx"].unique()
for c in classes:
    os.makedirs(os.path.join(IMG_DIR, c), exist_ok=True)

# Move images into class folders
moved = 0
for _, row in df.iterrows():
    img_name = row["image_id"] + ".jpg"
    label = row["dx"]

    src = os.path.join(IMG_DIR, img_name)
    dst = os.path.join(IMG_DIR, label, img_name)

    if os.path.exists(src):
        shutil.move(src, dst)
        moved += 1

print(f"Done ✅ Moved {moved} images into class folders.")
