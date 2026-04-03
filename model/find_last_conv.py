import tensorflow as tf

MODEL_PATH = "model/skin_cancer_model_v2.keras"
model = tf.keras.models.load_model(MODEL_PATH)


base = model.get_layer("mobilenetv2_1.00_224")


conv_like = []
for l in base.layers:
    name = l.name.lower()
    if "conv" in name or "depthwise" in name:
        conv_like.append(l.name)

print("Total layers in base:", len(base.layers))
print("\nLast 30 conv-like layer names in base:\n")
for n in conv_like[-30:]:
    print(n)


if "Conv_1" in [l.name for l in base.layers]:
    print("\n✅ Found expected last conv layer: Conv_1")
