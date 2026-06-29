"""
Corn Disease Detection — Streamlit demo.

A convolutional neural network that classifies a corn (maize) leaf image into
one of four classes: Blight, Common Rust, Gray Leaf Spot, or Healthy.

Run locally:   streamlit run streamlit_app.py
"""

import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# Keep TensorFlow quiet in the deployed logs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from keras.models import load_model  # noqa: E402  (import after env var is set)

# --- Constants ---------------------------------------------------------------

MODEL_PATH = "model.h5"
INPUT_SIZE = (255, 255)  # the model's internal Resizing layer expects this

# Class order MUST match the training order from the notebook:
# tf.keras.preprocessing.image_dataset_from_directory -> ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
CLASS_NAMES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]
CLASS_LABELS = {
    "Blight": "Northern Corn Leaf Blight",
    "Common_Rust": "Common Rust",
    "Gray_Leaf_Spot": "Gray Leaf Spot",
    "Healthy": "Healthy",
}

# Bundled example images (filename -> true label) for one-click trials.
EXAMPLES = {
    "static/Corn_Blight (120).JPG": "Blight",
    "static/Corn_Common_Rust (28).jpg": "Common_Rust",
    "static/Corn_Gray_Spot (14).jpg": "Gray_Leaf_Spot",
    "static/Corn_Health (13).jpg": "Healthy",
}


# --- Model -------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model…")
def get_model():
    # compile=False: the model was saved with an older Keras whose serialized
    # loss config can't be rebuilt by Keras 3. We only need inference, so we
    # skip recompilation entirely.
    return load_model(MODEL_PATH, compile=False)


def predict(img: Image.Image):
    """Return (probabilities dict, predicted_class) for a PIL image."""
    rgb = img.convert("RGB").resize(INPUT_SIZE)
    arr = np.array(rgb).reshape((1, INPUT_SIZE[0], INPUT_SIZE[1], 3))
    # NOTE: we feed raw 0-255 values on purpose — the model has a Rescaling(1/255)
    # layer baked in, so rescaling here would double-normalize the input.
    preds = get_model().predict(arr, verbose=0)[0]
    probs = {name: float(p) for name, p in zip(CLASS_NAMES, preds)}
    top = max(probs, key=probs.get)
    return probs, top


# --- UI ----------------------------------------------------------------------

st.set_page_config(page_title="Corn Disease Detection", page_icon="🌽", layout="centered")

st.title("🌽 Corn Disease Detection")
st.caption(
    "A CNN that classifies maize leaf images as **Blight**, **Common Rust**, "
    "**Gray Leaf Spot**, or **Healthy**. Educational demo — not agronomic advice."
)


def render_result(img: Image.Image, true_label: str | None = None):
    probs, top = predict(img)
    col_img, col_pred = st.columns([1, 1])
    with col_img:
        st.image(img, caption="Input image", use_container_width=True)
    with col_pred:
        confidence = probs[top]
        st.metric("Prediction", CLASS_LABELS[top], f"{confidence:.1%} confidence")
        if true_label:
            ok = top == true_label
            st.write(("✅ Correct" if ok else "❌ Misclassified")
                     + f" — true label: **{CLASS_LABELS[true_label]}**")
        st.write("**Class probabilities**")
        ordered = dict(sorted(probs.items(), key=lambda kv: kv[1], reverse=True))
        st.bar_chart({CLASS_LABELS[k]: v for k, v in ordered.items()})


uploaded = st.file_uploader("Upload a corn-leaf image", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    render_result(Image.open(uploaded))
else:
    st.write("**…or try a bundled example:**")
    cols = st.columns(len(EXAMPLES))
    for col, (path, label) in zip(cols, EXAMPLES.items()):
        if col.button(CLASS_LABELS[label], use_container_width=True):
            if Path(path).exists():
                render_result(Image.open(path), true_label=label)
            else:
                st.warning(f"Example not found: {path}")


# --- Model card / honest performance section ---------------------------------

with st.expander("📊 Model & performance details"):
    st.markdown(
        """
**Architecture** — 5-block convolutional network (Conv2D → MaxPool ×5 → Dense),
~278K parameters, with `Resizing` + `Rescaling` and on-the-fly augmentation
(`RandomFlip`, `RandomRotation`) baked in as model layers.

**Training data** — 4,188 images across 4 classes (Kaggle corn/maize leaf
dataset), split 80/10/10 train/val/test. The classes are **imbalanced**:

| Class | Images |
|---|---|
| Common Rust | 1,047 |
| Blight | 928 |
| Healthy | 935 |
| Gray Leaf Spot | **440** |

**Training** — 30 epochs, Adam, sparse categorical cross-entropy.

**Performance (honest numbers):**
- **~75% test accuracy** (`model.evaluate` on the held-out test set).
- ~94% validation accuracy during training — optimistic relative to test, a
  sign of mild overfitting.
        """
    )

with st.expander("⚠️ Known limitations & next steps"):
    st.markdown(
        """
- **Gray Leaf Spot is the weak class.** It has the fewest training images (440,
  ~40% of Common Rust's count) and is visually similar to Common Rust, so the
  model sometimes confuses the two — consistent with live testing on samples.
- **Class imbalance** isn't corrected; class weights or targeted augmentation
  for Gray Leaf Spot would likely help.
- **Mild overfitting** (val 94% vs test 75%) — more aggressive regularization /
  early stopping, or a pretrained backbone (transfer learning), are natural next
  steps.
- The model only knows these 4 maize classes — out-of-distribution images
  (other crops, non-leaves) will still be forced into one of the four buckets.
        """
    )
