import os
import random
import subprocess
from pathlib import Path
import sys


DATA_DIR = Path("data/images")
OUT_DIR = Path("outputs/gradcam_examples")
GRADCAM_SCRIPT = Path("model/gradcam.py")


N_PER_CLASS = {
    "mel": 3,
    "nv": 3,
    "bcc": 2,
    "bkl": 2,
    "akiec": 1,
    "df": 1,
    "vasc": 1,
}

SEED = 42


random.seed(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)

def pick_images(class_name, n):
    class_dir = DATA_DIR / class_name
    imgs = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
    if not imgs:
        print(f"⚠️ No images found in {class_dir}")
        return []
    return random.sample(imgs, min(n, len(imgs)))

def run_gradcam(img_path):
    cmd = [sys.executable, str(GRADCAM_SCRIPT), "--img", str(img_path)]
    subprocess.run(cmd, check=False)

def main():
    print("Generating Grad-CAM pack...")
    for cls, n in N_PER_CLASS.items():
        picked = pick_images(cls, n)
        print(f"{cls}: picked {len(picked)}")
        for p in picked:
            run_gradcam(p)

    print("\n✅ Done. Check:")
    print("  outputs/gradcam/  (all saved overlays)")
    print("Now we’ll copy the best ones into outputs/gradcam_examples manually for the report.")

if __name__ == "__main__":
    main()
