# 🌽 Corn Disease Detection

A deep-learning web app that classifies a **corn (maize) leaf image** into one of four
categories — **Northern Corn Leaf Blight, Common Rust, Gray Leaf Spot, or Healthy** —
using a convolutional neural network trained in TensorFlow/Keras.

**▶️ Live demo:** _<!-- TODO: paste your Streamlit Cloud URL here after deploying, e.g. https://corn-disease-detection.streamlit.app -->_

<!--
  TODO (optional but recommended): add a screenshot or GIF of the app.
  The easiest way: open this README on GitHub, click the pencil (Edit), and
  drag an image into the text area — GitHub uploads it and inserts the link.
  Then replace the line below:
-->
<!-- ![App demo](assets/demo.png) -->

---

## What it does

Upload a photo of a maize leaf and the model returns the predicted disease class
with a confidence score and a probability bar chart across all four classes. Four
bundled example images let anyone try it in one click — no upload required.

There are **two front-ends** in this repo:

| App | File | Use it for |
|---|---|---|
| **Streamlit** (primary) | `streamlit_app.py` | The clean, deployable demo. Confidence chart, example gallery, model card. |
| **Flask** (original) | `app.py` + `templates/index.html` | A minimal server-rendered version of the same classifier. |

---

## Quickstart

```bash
# 1. Create an environment (Python 3.10–3.12; TensorFlow has no 3.13/3.14 wheels yet)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run streamlit_app.py
```

Then open http://localhost:8501.

To run the Flask version instead: `python app.py` → http://127.0.0.1:5000.

---

## Model

A compact CNN built and trained in [`dieseasePrediction.ipynb`](dieseasePrediction.ipynb).

- **Architecture:** 5 × (Conv2D → MaxPool) → Flatten → Dense(64) → Dense(4, softmax),
  ~**278K** parameters. Input resizing, rescaling, and on-the-fly augmentation
  (`RandomFlip`, `RandomRotation`) are baked in as model layers, so the saved model
  takes a raw RGB image and handles preprocessing internally.
- **Training:** 30 epochs, Adam, sparse categorical cross-entropy.
- **Saved weights:** `model.h5` (loaded with `compile=False` for Keras 3 compatibility).

### Dataset

~**4,188** labelled maize-leaf images across 4 classes (Kaggle corn/maize leaf disease
dataset), split **80 / 10 / 10** train / val / test. The classes are **imbalanced**:

| Class | Images |
|---|---:|
| Common Rust | 1,047 |
| Healthy | 935 |
| Blight | 928 |
| **Gray Leaf Spot** | **440** |

### Performance

| Metric | Value |
|---|---|
| **Test accuracy** (`model.evaluate`) | **~75%** |
| Validation accuracy (training) | ~94% |

The gap between validation and test accuracy points to mild **overfitting**, and the
under-represented **Gray Leaf Spot** class is the model's weakest — it's sometimes
confused with Common Rust (visually similar lesions + fewest training samples).

---

## 🐞 Engineering highlight: a hidden evaluation bug

The original notebook reported **two contradictory test accuracies**: `model.evaluate`
said **~75%**, but a manual `classification_report` said **~35%** (barely above the 25%
random baseline for 4 classes).

**Root cause:** the test pipeline was built with `.shuffle()`, which reshuffles on
*every* iteration (`reshuffle_each_iteration=True` by default). The evaluation code
iterated the test set twice — once for `model.predict()` and once to collect the true
labels — so the two passes came back in **different orders**, silently misaligning
`y_true` and `y_pred`. `model.evaluate` was unaffected because it compares predictions
and labels within a single pass.

**Fix:** drop `.shuffle()` from the val/test pipelines and collect predictions and
labels in one aligned pass, then plot a real confusion matrix. See the commit history
and the evaluation cells in the notebook.

---

## Project structure

```
.
├── streamlit_app.py          # Streamlit demo (primary front-end)
├── app.py                    # Flask app (alternative front-end)
├── templates/index.html      # Flask template
├── dieseasePrediction.ipynb  # Training + evaluation notebook
├── model.h5                  # Trained CNN weights
├── static/                   # Bundled example leaf images
├── requirements.txt
└── README.md
```

---

## Limitations & next steps

- **Class imbalance** isn't corrected — class weights or targeted augmentation for
  Gray Leaf Spot would likely lift its recall.
- **Mild overfitting** — stronger regularization / early stopping, or a pretrained
  backbone (transfer learning, e.g. MobileNetV2/EfficientNet) are natural improvements.
- **Closed-world model** — it only knows these 4 maize classes; out-of-distribution
  images are still forced into one of the four buckets.

---

## Acknowledgements

Dataset: Kaggle corn / maize leaf disease dataset. This project is for educational and
demonstration purposes — not agronomic advice.
