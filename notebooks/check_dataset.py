

import os
import pandas as pd

IMAGE_DIR = "data/images"
CSV_PATH = "data/labels.csv"


df = pd.read_csv(CSV_PATH)
print("CSV loaded ✅")
print("Rows:", len(df))
print("Columns:", list(df.columns))


files_in_folder = os.listdir(IMAGE_DIR)
image_set = set(files_in_folder)

print("\nImages folder scanned ✅")
print("Files in data/images:", len(files_in_folder))


df["filename"] = df["image_id"].astype(str) + ".jpg"


missing = df[~df["filename"].isin(image_set)]

print("\nMatch check ✅")
print("Missing images:", len(missing))

if len(missing) > 0:
    print("\nSome missing filenames:")
    print(missing["filename"].head(10).to_string(index=False))


print("\nClass distribution (dx) ✅")
print(df["dx"].value_counts())
