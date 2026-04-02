# check_dataset.py
# Quick sanity check:
# - Do the images in data/images match the labels in data/labels.csv?
# - What does the class distribution look like?

import os
import pandas as pd

IMAGE_DIR = "data/images"
CSV_PATH = "data/labels.csv"

# 1) Load the metadata CSV
df = pd.read_csv(CSV_PATH)
print("CSV loaded ✅")
print("Rows:", len(df))
print("Columns:", list(df.columns))

# 2) Count how many image files we have
files_in_folder = os.listdir(IMAGE_DIR)
image_set = set(files_in_folder)

print("\nImages folder scanned ✅")
print("Files in data/images:", len(files_in_folder))

# 3) In HAM10000, image_id is stored without ".jpg"
# Convert image_id -> filename so it matches the actual file names
df["filename"] = df["image_id"].astype(str) + ".jpg"

# 4) Check if any filenames in the CSV are missing from the folder
missing = df[~df["filename"].isin(image_set)]

print("\nMatch check ✅")
print("Missing images:", len(missing))

if len(missing) > 0:
    print("\nSome missing filenames:")
    print(missing["filename"].head(10).to_string(index=False))

# 5) Class distribution (dx note: dataset is imbalanced)
print("\nClass distribution (dx) ✅")
print(df["dx"].value_counts())
