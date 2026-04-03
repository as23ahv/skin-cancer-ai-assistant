import sys
from pathlib import Path
import io
import json
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt

from model.gradcam import build_grad_model, make_gradcam, overlay_heatmap


from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


MODEL_PATH = "model/skin_cancer_model_v2.keras"
LABELS_PATH = "model/labels.json"
IMG_SIZE = (224, 224)


DEFAULT_LOW_CONF_THRESHOLD = 0.55
MED_CONF_THRESHOLD = 0.75


st.set_page_config(page_title="Skin Cancer AI Assistant", layout="wide")


CUSTOM_CSS = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* App background */
.stApp {
  background: radial-gradient(circle at 20% 10%, rgba(99,102,241,0.18), transparent 40%),
              radial-gradient(circle at 80% 20%, rgba(16,185,129,0.14), transparent 40%),
              radial-gradient(circle at 50% 90%, rgba(236,72,153,0.10), transparent 45%);
}

/* Card style */
.card {
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 18px;
  padding: 16px 18px;
  background: rgba(255,255,255,0.04);
  box-shadow: 0 10px 30px rgba(0,0,0,0.12);
}

/* Badge */
.badge {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
  border: 1px solid rgba(255,255,255,0.18);
  background: rgba(255,255,255,0.06);
}

/* Title */
.bigtitle {
  font-size: 32px;
  font-weight: 800;
  letter-spacing: -0.5px;
  margin-bottom: 8px;
}

/* Subtle text */
.subtle { opacity: 0.82; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  

def add_chat(role: str, content: str):
    st.session_state.chat_history.append({"role": role, "content": content})



@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    grad_model = build_grad_model(model)
    return model, grad_model


@st.cache_data
def load_labels():
    with open(LABELS_PATH, "r") as f:
        m = json.load(f)
    return [m[str(i)] for i in range(len(m))]


def preprocess(img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype(np.float32) / 255.0
    return arr


def confidence_band(conf: float, low_thr: float) -> str:
    if conf < low_thr:
        return "Low"
    if conf < MED_CONF_THRESHOLD:
        return "Medium"
    return "High"


def make_topk_bar(labels, probs, k=5):
    topk = np.argsort(probs)[::-1][:k]
    top_labels = [labels[int(i)] for i in topk][::-1]
    top_probs = [float(probs[int(i)]) for i in topk][::-1]

    fig = plt.figure()
    plt.barh(top_labels, top_probs)
    plt.xlim(0, 1)
    plt.xlabel("Probability")
    plt.tight_layout()
    return fig


def generate_pdf_report(
    original_img: Image.Image,
    overlay_img: Image.Image,
    pred_label: str,
    conf: float,
    topk_items,
) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, height - 50, "Skin Cancer AI Assistant - Prediction Report")

    c.setFont("Helvetica", 10)
    c.drawString(40, height - 70, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, height - 110, "Prediction")
    c.setFont("Helvetica", 11)
    c.drawString(40, height - 130, f"Class: {pred_label}")
    c.drawString(40, height - 150, f"Confidence: {conf*100:.2f}%")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, height - 185, "Top Predictions")
    c.setFont("Helvetica", 10)

    y = height - 205
    for name, p in topk_items:
        c.drawString(40, y, f"- {name}: {p*100:.2f}%")
        y -= 14

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y - 10, "Images")
    y_img_top = y - 25

    orig_reader = ImageReader(original_img.convert("RGB"))
    overlay_reader = ImageReader(overlay_img.convert("RGB"))

    c.setFont("Helvetica", 10)
    c.drawString(40, y_img_top, "Original")
    c.drawImage(orig_reader, 40, y_img_top - 210, width=240, height=200, preserveAspectRatio=True, mask="auto")

    c.drawString(320, y_img_top, "Grad-CAM Overlay")
    c.drawImage(overlay_reader, 320, y_img_top - 210, width=240, height=200, preserveAspectRatio=True, mask="auto")

    c.setFont("Helvetica", 9)
    c.drawString(40, 40, "Disclaimer: Educational use only. Not a medical device. Do not use for diagnosis.")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()



with st.sidebar:
    st.markdown("### 🧠 Controls")
    show_gradcam = st.toggle("Show Grad-CAM", value=True)
    topk_k = st.slider("Top-K predictions", 3, 8, 5)
    low_conf_threshold = st.slider("Low confidence threshold", 0.30, 0.80, float(DEFAULT_LOW_CONF_THRESHOLD), 0.01)

    st.markdown("---")
    st.markdown("### 💬 Your Chats")

    
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Clear chat"):
            st.session_state.chat_history = []
    with col_b:
        st.caption(f"{len(st.session_state.chat_history)} msgs")

    
    if len(st.session_state.chat_history) == 0:
        st.caption("Your messages and the assistant replies will appear here.")
    else:
        for msg in st.session_state.chat_history[-30:]:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**Assistant:** {msg['content']}")



st.markdown(
    """
<div class="card">
  <div class="bigtitle">🧠 Skin Cancer AI Assistant</div>
  <div class="subtle">
    Upload a dermoscopic image to get a prediction + Grad-CAM explanation.
    <span class="badge">Educational only</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")



left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📤 Upload")
    uploaded = st.file_uploader("Upload skin lesion image", type=["jpg", "png", "jpeg"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧾 Safety")
    st.warning("This tool is for educational purposes only and must not be used for medical diagnosis.")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧩 Output")
    st.write("Upload an image to see results here.")
    st.markdown("</div>", unsafe_allow_html=True)



if uploaded is None:
    
    q = st.chat_input("Ask about the tool (e.g., “what does Grad-CAM mean?”)")
    if q:
        add_chat("user", q)
        add_chat("assistant", "Upload an image first, then I can explain the prediction + heatmap. Grad-CAM highlights areas the model focused on.")
        st.rerun()
    st.stop()



try:
    image = Image.open(uploaded).convert("RGB")
except UnidentifiedImageError:
    st.error("That file doesn’t look like a valid image. Try a JPG/PNG.")
    st.stop()
except Exception as e:
    st.error(f"Could not open image: {e}")
    st.stop()


add_chat("user", f"Uploaded image: {getattr(uploaded, 'name', 'image')}")


model, grad_model = load_model()
labels = load_labels()

img_array = preprocess(image)
img_tensor = np.expand_dims(img_array, axis=0)

with st.spinner("Running model inference..."):
    preds = model.predict(img_tensor, verbose=0)[0]

pred_idx = int(np.argmax(preds))
confidence = float(preds[pred_idx])
label = labels[pred_idx]
band = confidence_band(confidence, low_conf_threshold)

topk_idx = np.argsort(preds)[::-1][:topk_k]
topk_items = [(labels[int(i)], float(preds[int(i)])) for i in topk_idx]


overlay = None
if show_gradcam:
    try:
        heatmap, _, _ = make_gradcam(grad_model, img_array, class_index=pred_idx)
        overlay = overlay_heatmap(image, heatmap)
    except Exception as e:
        st.warning(f"Grad-CAM failed (still showing prediction). Details: {e}")
        overlay = None


add_chat("assistant", f"Prediction: {label} ({confidence*100:.2f}%, certainty: {band}).")


out_left, out_right = st.columns([1, 1], gap="large")

with out_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🖼️ Image")
    st.image(image, caption="Uploaded image", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if overlay is not None:
        st.write("")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔥 Grad-CAM")
        st.image(overlay, caption="Model attention heatmap overlay", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with out_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔍 Prediction")

    m1, m2, m3 = st.columns(3)
    m1.metric("Class", label)
    m2.metric("Confidence", f"{confidence*100:.2f}%")
    m3.metric("Certainty", band)

    if confidence < low_conf_threshold:
        st.error(
            "Low confidence prediction — treat this as uncertain. "
            "Consider using a clearer dermoscopic image and/or seeking professional medical advice."
        )
    elif confidence < MED_CONF_THRESHOLD:
        st.warning("Medium confidence — interpret with caution.")
    else:
        st.success("High confidence (model-level) — still not a medical diagnosis.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Top Predictions")
    fig = make_topk_bar(labels, preds, k=topk_k)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💬 Explanation")
    st.write(
        f"The model predicts **{label}** with **{confidence*100:.2f}%** confidence. "
        f"If Grad-CAM is enabled, the highlighted regions indicate areas that contributed most to the prediction."
    )
    st.caption("Tip: If the lesion is tiny, blurry, or lighting is off, confidence will usually drop.")
    st.markdown("</div>", unsafe_allow_html=True)

    if overlay is not None:
        pdf_bytes = generate_pdf_report(
            original_img=image,
            overlay_img=overlay,
            pred_label=label,
            conf=confidence,
            topk_items=topk_items,
        )
        st.write("")
        st.download_button(
            label="⬇️ Download Prediction Report (PDF)",
            data=pdf_bytes,
            file_name=f"skin_ai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


user_q = st.chat_input("Ask a question about your result (e.g., “why low confidence?”)")
if user_q:
    add_chat("user", user_q)

    q_lower = user_q.lower()

    
    top3_idx = np.argsort(preds)[::-1][:3]
    top3_lines = "\n".join([f"- {labels[int(i)]}: {float(preds[int(i)])*100:.2f}%" for i in top3_idx])

    def detailed_output_explanation():
        return (
            f"Here’s a detailed breakdown of what you’re seeing:\n\n"
            f"**1) Predicted class**\n"
            f"- The model’s top prediction is **{label}**.\n\n"
            f"**2) Confidence (probability)**\n"
            f"- The confidence shown (**{confidence*100:.2f}%**) is the model’s estimated probability for **{label}**.\n"
            f"- This is not a guarantee — it’s how strongly the model leans toward that class based on patterns it learned.\n\n"
            f"**3) Certainty level (Low / Medium / High)**\n"
            f"- Your app labels this as **{band}** based on the confidence thresholds.\n"
            f"  - **Low**: below your slider threshold ({low_conf_threshold:.2f})\n"
            f"  - **Medium**: between that threshold and {MED_CONF_THRESHOLD:.2f}\n"
            f"  - **High**: above {MED_CONF_THRESHOLD:.2f}\n\n"
            f"**4) Top predictions (what else it considered)**\n"
            f"{top3_lines}\n\n"
            f"**5) Why confidence might only be Medium**\n"
            f"- Different lesion classes can share similar colour/texture patterns.\n"
            f"- Lighting, focus/blur, zoom level, hair, shadows, or reflections can reduce certainty.\n"
            f"- If the lesion features aren’t strongly distinctive, the model spreads probability across multiple classes.\n\n"
            f"**6) Grad-CAM (if enabled)**\n"
            f"- The heatmap overlay shows *where* the model focused most.\n"
            f"- Brighter/stronger regions = areas that influenced the decision more.\n"
            f"- This improves transparency (XAI), but it still doesn’t mean diagnosis.\n\n"
            f"If you want, tell me what you want explained more: **confidence**, **top predictions**, or **Grad-CAM**."
        )

    
    wants_detail = any(
        phrase in q_lower
        for phrase in [
            "explain in detail", "explain the output", "explain output", "better explanation",
            "elaborate", "more detail", "go deeper", "explain more", "detailed"
        ]
    )

    if wants_detail:
        reply = detailed_output_explanation()

    
    elif "grad" in q_lower or "cam" in q_lower or "heatmap" in q_lower:
        reply = (
            f"Grad-CAM is the visual explanation. It highlights regions that most influenced the prediction.\n\n"
            f"- Bright/strong areas = higher influence\n"
            f"- It helps you understand *where the model looked*, not prove a diagnosis.\n\n"
            f"In your case the model predicted **{label}** with **{confidence*100:.2f}%** confidence ({band})."
        )
    elif "low confidence" in q_lower or "not confident" in q_lower or "uncertain" in q_lower:
        reply = (
            f"Your confidence is **{confidence*100:.2f}%** ({band}). Common reasons confidence drops:\n"
            f"- blurry image / low resolution\n"
            f"- lighting/reflections/shadows\n"
            f"- lesion is small or zoomed out\n"
            f"- overlapping visual patterns between classes\n\n"
            f"Try a clearer dermoscopic image and see if confidence increases."
        )
    elif "top" in q_lower and ("3" in q_lower or "pred" in q_lower):
        reply = (
            f"Here are the **top 3 predictions** the model considered:\n\n"
            f"{top3_lines}\n\n"
            f"This is useful when confidence isn’t super high because you can see what it was torn between."
        )
    elif "confidence" in q_lower:
        reply = (
            f"**Confidence = {confidence*100:.2f}%** for **{label}**.\n\n"
            f"It’s the model’s probability estimate for that class. If multiple classes look similar, "
            f"the model spreads probability across them, lowering the top confidence.\n\n"
            f"Your certainty label is **{band}** based on your thresholds."
        )
    elif "what is" in q_lower or "meaning" in q_lower:
        reply = "I can explain terms like confidence, classes, and Grad-CAM. Ask what you want explained and I’ll break it down."
    else:
        reply = (
            "Ask me things like:\n"
            "- “Explain the output in detail”\n"
            "- “Why is confidence medium?”\n"
            "- “What does Grad-CAM show?”\n"
            "- “Show top 3 predictions”\n\n"
            "I can explain the model output (confidence, top predictions, Grad-CAM). I can’t provide medical advice or diagnosis."
        )

    add_chat("assistant", reply)
    st.rerun()