# Skin Cancer AI Assistant

AI-powered skin cancer classification system using MobileNetV2 with GradCAM explainability and Streamlit interface.

**University of Hertfordshire — BSc Computer Science — 6COM2017 Artificial Intelligence Project**

## Overview

This project implements an interactive web application for classifying dermoscopic skin lesion images into seven diagnostic categories using a transfer-learning convolutional neural network. The system includes:

- **MobileNetV2 classifier** trained on the HAM10000 dataset (10,015 images, 7 classes)
- **GradCAM explainability** — visual heatmaps showing which image regions influenced each prediction
- **Streamlit web interface** with image upload, chatbot Q&A, confidence bands, top-K predictions and PDF report export
- **Comprehensive evaluation** with accuracy, F1 scores, AUC, confusion matrix, calibration error and melanoma-specific metrics

### Prerequisites

- Python 3.10+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/as23ahv/skin-cancer-ai-assistant.git
cd skin-cancer-ai-assistant
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

3. Download the HAM10000 dataset and place images in `data/images/` organised by class folder. Run `notebooks/prepare_ham10000.py` if starting from the raw download.

### Running the Application

```bash
cd chatbot
streamlit run app.py
```

The application will open in your browser. Upload a dermoscopic image to receive a classification prediction with GradCAM visualisation.

### Training the Model

```bash
python model/train_model_v2.py
```

This runs the two-phase training process:
- **Phase 1:** Frozen MobileNetV2 backbone, 12 epochs, lr=0.001
- **Phase 2:** Last 40 layers unfrozen, 8 epochs, lr=0.00001

### Evaluating the Model

```bash
python model/evaluate_model_v2.py
```

Outputs are saved to `model/eval/` including metrics, confusion matrix, and classification report.

### Generating GradCAM Visualisations

```bash
python model/predict_image.py --img path/to/image.jpg
```

## Results

| Metric | Value |
|--------|-------|
| Accuracy | 68.2% |
| AUC (macro, OvR) | 0.886 |
| Macro F1 | 0.43 |
| Weighted F1 | 0.69 |
| ECE | 0.031 |
| Melanoma Sensitivity | 0.316 |
| Melanoma Specificity | 0.952 |

## Key Technologies

- **TensorFlow / Keras** — model training and inference
- **MobileNetV2** — pretrained backbone (ImageNet weights)
- **Streamlit** — interactive web application
- **scikit-learn** — evaluation metrics and class weighting
- **ReportLab** — PDF report generation
- **Matplotlib** — visualisation

## Disclaimer

This system is for **educational and research purposes only**. It must not be used as a medical diagnostic tool. Always consult a qualified healthcare professional for medical advice.

## Author

A Subhani — University of Hertfordshire, 2025-26