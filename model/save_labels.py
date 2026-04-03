import os, json

DATA_DIR = "data/images"  
classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

with open("model/labels.json", "w") as f:
    json.dump({str(i): c for i, c in enumerate(classes)}, f, indent=2)

print("Saved model/labels.json:", classes)
