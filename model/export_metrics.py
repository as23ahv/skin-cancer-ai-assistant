import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

MODEL_PATH = "model/skin_cancer_model_v2.keras"
LABELS_PATH = "model/labels.json"
DATA_DIR = "data/images"
OUT_DIR = "outputs/metrics"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

def load_labels():
    with open(LABELS_PATH, "r") as f:
        m = json.load(f)
    return [m[str(i)] for i in range(len(m))]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    labels = load_labels()
    model = tf.keras.models.load_model(MODEL_PATH)

    ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

    y_true = []
    y_pred = []

    for x, y in ds:
        preds = model.predict(x, verbose=0)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    
    report = classification_report(y_true, y_pred, target_names=labels, digits=3)
    with open(os.path.join(OUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation=45, values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=200)
    plt.close()


    acc = float((y_true == y_pred).mean())
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")

    print("✅ Export complete:")
    print(f" - {OUT_DIR}/classification_report.txt")
    print(f" - {OUT_DIR}/confusion_matrix.png")
    print(f" - {OUT_DIR}/summary.txt")

if __name__ == "__main__":
    main()
