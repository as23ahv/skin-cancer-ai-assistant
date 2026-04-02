import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "model/skin_cancer_model_v2.keras"
LABELS_PATH = "model/labels.json"
OUT_DIR = "outputs/gradcam"
IMG_SIZE = (224, 224)

BASE_MODEL_NAME = "mobilenetv2_1.00_224"
GAP_NAME = "global_average_pooling2d"
DROPOUT_NAME = "dropout"
DENSE_NAME = "dense"
OUT_NAME = "dense_1"


def load_labels():
    with open(LABELS_PATH, "r") as f:
        m = json.load(f)
    return [m[str(i)] for i in range(len(m))]


def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize(IMG_SIZE)
    arr = np.array(img_resized).astype(np.float32) / 255.0
    return img, arr


def overlay_heatmap(original_pil, heatmap, alpha=0.45):
    """
    Create a simple red/blue overlay without matplotlib.
    heatmap: (H,W) in [0,1]
    """
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(original_pil.size)
    heat = np.array(heatmap_img)

    # red for high attention, blue for low attention
    heat_rgb = np.stack([heat, np.zeros_like(heat), 255 - heat], axis=-1).astype(np.uint8)
    heat_pil = Image.fromarray(heat_rgb)

    return Image.blend(original_pil.convert("RGB"), heat_pil, alpha=alpha)


def build_grad_model(model):
    """
    Rebuilds a clean connected graph by CALLING the layers again.
    Outputs:
      - feature_maps: output of MobileNetV2 base (conv feature map)
      - preds: final softmax predictions
    """
    base = model.get_layer(BASE_MODEL_NAME)
    gap = model.get_layer(GAP_NAME)
    drop = model.get_layer(DROPOUT_NAME)
    dense = model.get_layer(DENSE_NAME)
    out = model.get_layer(OUT_NAME)

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))

    feature_maps = base(inputs, training=False)
    x = gap(feature_maps)
    x = drop(x, training=False)
    x = dense(x)
    preds = out(x)

    return tf.keras.Model(inputs=inputs, outputs=[feature_maps, preds])


def make_gradcam(grad_model, img_array, class_index=None):
    """
    grad_model returns (feature_maps, preds)
    img_array: (224,224,3) float32 in [0,1]
    class_index: int OR tf.Tensor OR None
    """
    img_tensor = tf.expand_dims(img_array, axis=0)

    with tf.GradientTape() as tape:
        feature_maps, preds = grad_model(img_tensor, training=False)

        # --- FIX: ensure class_index is always a tensor ---
        if class_index is None:
            class_index = tf.argmax(preds[0])
        else:
            class_index = tf.convert_to_tensor(class_index)

        loss = preds[:, class_index]

    grads = tape.gradient(loss, feature_maps)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    feature_maps = feature_maps[0]  # (h,w,c)
    heatmap = tf.reduce_sum(feature_maps * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), int(class_index.numpy()), preds.numpy()[0]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="Full path to an image")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    model = tf.keras.models.load_model(MODEL_PATH)
    labels = load_labels()

    grad_model = build_grad_model(model)

    original_pil, img_array = preprocess_image(args.img)
    heatmap, pred_idx, probs = make_gradcam(grad_model, img_array)

    pred_label = labels[pred_idx]
    confidence = float(probs[pred_idx])

    overlay = overlay_heatmap(original_pil, heatmap, alpha=0.45)

    base_name = os.path.splitext(os.path.basename(args.img))[0]
    out_path = os.path.join(OUT_DIR, f"gradcam_{pred_label}_{base_name}.png")
    overlay.save(out_path)

    print(f"✅ Pred: {pred_label} ({confidence:.3f})")
    print(f"✅ Saved: {out_path}")


if __name__ == "__main__":
    main()
