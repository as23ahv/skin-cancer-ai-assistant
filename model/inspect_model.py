import tensorflow as tf

MODEL_PATH = "model/skin_cancer_model_v2.keras"

model = tf.keras.models.load_model(MODEL_PATH)

print("\nLast 30 layers:\n")
for layer in model.layers[-30:]:
    print(layer.name, layer.__class__.__name__)
