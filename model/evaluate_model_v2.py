import os
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score
)

import matplotlib.pyplot as plt


# -------------------- CONFIG --------------------
SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

DATA_DIR = Path("data/images")
MODEL_PATH = Path("model/skin_cancer_model_v2.keras")

OUT_DIR = Path("model/eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# If you want *identical* splits every time, keep these files after first run:
SPLIT_DIR = OUT_DIR / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.80  # only used if split files don't exist
# ------------------------------------------------


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def list_classes(data_dir: Path):
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    if not classes:
        raise RuntimeError(f"No class folders found in {data_dir}")
    return classes


def build_file_list(data_dir: Path, classes):
    paths, labels = [], []
    for idx, c in enumerate(classes):
        class_dir = data_dir / c
        for p in class_dir.iterdir():
            if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                paths.append(str(p))
                labels.append(idx)
    if len(paths) == 0:
        raise RuntimeError(f"No images found under {data_dir}")
    return np.array(paths), np.array(labels, dtype=np.int64)


def save_split(paths, labels, split_name: str):
    np.save(SPLIT_DIR / f"{split_name}_paths.npy", paths)
    np.save(SPLIT_DIR / f"{split_name}_labels.npy", labels)


def load_split(split_name: str):
    p = np.load(SPLIT_DIR / f"{split_name}_paths.npy", allow_pickle=True)
    y = np.load(SPLIT_DIR / f"{split_name}_labels.npy", allow_pickle=True)
    return p, y


def ensure_splits(all_paths, all_labels):
    """
    Creates deterministic splits ONCE and saves them.
    Next runs load exact same test set -> perfect reproducibility.
    """
    test_paths_file = SPLIT_DIR / "test_paths.npy"
    test_labels_file = SPLIT_DIR / "test_labels.npy"

    if test_paths_file.exists() and test_labels_file.exists():
        train_paths, train_labels = load_split("train")
        test_paths, test_labels = load_split("test")
        return train_paths, train_labels, test_paths, test_labels

    # Create deterministic shuffle
    perm = np.random.permutation(len(all_paths))
    all_paths = all_paths[perm]
    all_labels = all_labels[perm]

    n_train = int(len(all_paths) * TRAIN_RATIO)
    train_paths = all_paths[:n_train]
    train_labels = all_labels[:n_train]
    test_paths = all_paths[n_train:]
    test_labels = all_labels[n_train:]

    save_split(train_paths, train_labels, "train")
    save_split(test_paths, test_labels, "test")

    return train_paths, train_labels, test_paths, test_labels


def load_img(path, label):
    img_bytes = tf.io.read_file(path)
    # decode_image supports jpeg + png (unlike decode_jpeg)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def make_dataset(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def plot_confusion_matrix(cm, class_names, out_path: Path):
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # annotate
    thresh = cm.max() * 0.6 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(j, i, str(val), ha="center", va="center",
                     color="white" if val > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """
    ECE for multiclass using max prob as confidence.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if np.any(mask):
            bin_acc = np.mean(accuracies[mask])
            bin_conf = np.mean(confidences[mask])
            ece += np.abs(bin_acc - bin_conf) * (np.sum(mask) / len(y_true))
    return float(ece)


def plot_reliability(probs: np.ndarray, y_true: np.ndarray, out_path: Path, n_bins: int = 10):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    bin_accs = []
    bin_confs = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if np.any(mask):
            bin_centers.append((lo + hi) / 2)
            bin_accs.append(np.mean(accuracies[mask]))
            bin_confs.append(np.mean(confidences[mask]))

    fig = plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1])  # perfect calibration line
    if bin_centers:
        plt.plot(bin_confs, bin_accs, marker="o")
    plt.title("Reliability Diagram")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def melanoma_metrics(classes, y_true, y_pred):
    """
    If you have a class named 'mel' or 'melanoma' (case-insensitive),
    compute sensitivity/specificity melanoma vs rest.
    """
    name_map = {c.lower(): i for i, c in enumerate(classes)}
    mel_idx = None
    for key in name_map:
        if key in ["mel", "melanoma"]:
            mel_idx = name_map[key]
            break
    if mel_idx is None:
        return None  # not available

    y_true_bin = (np.array(y_true) == mel_idx).astype(int)
    y_pred_bin = (np.array(y_pred) == mel_idx).astype(int)

    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0  # recall
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "melanoma_class_index": mel_idx,
        "sensitivity_recall": float(sensitivity),
        "specificity": float(specificity),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }


def main():
    seed_everything(SEED)

    classes = list_classes(DATA_DIR)
    print("Classes:", classes)

    all_paths, all_labels = build_file_list(DATA_DIR, classes)
    train_paths, train_labels, test_paths, test_labels = ensure_splits(all_paths, all_labels)

    print(f"Total images: {len(all_paths)} | Train: {len(train_paths)} | Test: {len(test_paths)}")

    test_ds = make_dataset(test_paths, test_labels)

    model = tf.keras.models.load_model(MODEL_PATH)

    y_true = []
    y_pred = []
    y_prob = []

    for x, y in test_ds:
        probs = model.predict(x, verbose=0)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_prob.append(probs)

    y_prob = np.concatenate(y_prob, axis=0)

    # --- Core metrics ---
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    report_dict = classification_report(y_true, y_pred, target_names=classes, digits=4, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    # --- ROC-AUC (OvR) if possible ---
    auc_ovr = None
    try:
        # One-hot y_true
        y_true_oh = np.eye(len(classes))[np.array(y_true)]
        auc_ovr = roc_auc_score(y_true_oh, y_prob, average="macro", multi_class="ovr")
    except Exception:
        auc_ovr = None

    # --- Calibration ---
    ece = expected_calibration_error(y_prob, np.array(y_true), n_bins=15)

    # --- Melanoma metrics if present ---
    mel = melanoma_metrics(classes, y_true, y_pred)

    # --- Save artifacts ---
    np.save(OUT_DIR / "y_true.npy", np.array(y_true))
    np.save(OUT_DIR / "y_pred.npy", np.array(y_pred))
    np.save(OUT_DIR / "y_prob.npy", np.array(y_prob))

    with open(OUT_DIR / "classification_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "auc_ovr_macro": None if auc_ovr is None else float(auc_ovr),
        "ece": float(ece),
        "num_test_samples": int(len(y_true)),
        "classes": classes,
        "melanoma_vs_rest": mel
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(cm, classes, OUT_DIR / "confusion_matrix.png")
    plot_reliability(y_prob, np.array(y_true), OUT_DIR / "reliability.png", n_bins=10)

    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Macro F1:   {macro_f1:.4f}")
    print(f"Weighted F1:{weighted_f1:.4f}")
    print(f"AUC OvR:    {auc_ovr if auc_ovr is not None else 'N/A'}")
    print(f"ECE:        {ece:.4f}")
    if mel:
        print(f"Melanoma sensitivity/recall: {mel['sensitivity_recall']:.4f}")
        print(f"Melanoma specificity:        {mel['specificity']:.4f}")

    print(f"\nSaved to: {OUT_DIR.resolve()}")
    print("- metrics.json")
    print("- classification_report.json")
    print("- confusion_matrix.png")
    print("- reliability.png")
    print("- y_true.npy / y_pred.npy / y_prob.npy")
    print("- splits/*.npy (for reproducible test set)")


if __name__ == "__main__":
    main()