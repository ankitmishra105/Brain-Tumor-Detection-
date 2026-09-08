

# Brain Tumor MRI Classification — 4-Class Deep Learning Model

A deep learning system that classifies brain MRI scans into four categories using **transfer learning**, deployed as an interactive **Streamlit web app**.

## Overview

This project uses a pretrained CNN (VGG16, fine-tuned) to classify brain MRI images into 4 classes with **~94% test accuracy**. The trained model is wrapped in a simple web interface where users can upload an MRI scan and get an instant prediction.

## Dataset

The dataset contains 7,200 human brain MRI images divided into four classes:

- Glioma
- Meningioma
- Pituitary tumor
- No tumor

| Class | Training | Testing | Total |
|---|---|---|---|
| Glioma | 1,400 | 400 | 1,800 |
| Meningioma | 1,400 | 400 | 1,800 |
| Pituitary tumor | 1,400 | 400 | 1,800 |
| No tumor | 1,400 | 400 | 1,800 |
| **Total** | **5,600** | **1,600** | **7,200** |

Classes are perfectly balanced across both training and testing sets.

```
Training/
    glioma/
    meningioma/
    pituitary/
    notumor/
Testing/
    glioma/
    meningioma/
    pituitary/
    notumor/
```

## Approach

- **Transfer learning** with a VGG16 backbone pretrained on ImageNet
- Images resized to 224x224 and augmented (flip, rotation, zoom, contrast) to reduce overfitting
- **Two-stage training:**
  1. Train a new classification head on top of the frozen VGG16 backbone
  2. Fine-tune the last convolutional block of VGG16 at a low learning rate
- Evaluated on a fully held-out test set (1,600 images) never seen during training

## Model Performance

- **Test Accuracy: ~94%**
- Evaluated with a full classification report (precision, recall, F1-score per class) and a confusion matrix to check for class-wise misclassification patterns
- Training/validation accuracy and loss curves tracked across both training stages to confirm stable convergence without significant overfitting

## Tech Stack

- **Model:** TensorFlow / Keras, VGG16 (transfer learning)
- **App:** Streamlit
- **Other:** NumPy, Pillow, scikit-learn (metrics), Matplotlib, Seaborn (visualizations)

## Project Structure

```
mri_det/
    app.py                          # Streamlit inference app
    train_mri_tumor_model.py        # Training pipeline
    mri_tumor_vgg16_final.keras     # Trained model
    requirements.txt
    README.md
```

## Running the App

```bash
python -m venv myvenv
myvenv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
streamlit run app.py
```

Upload an MRI scan in the browser UI to get a predicted class with confidence scores.

## Disclaimer

This project is for educational and portfolio purposes only. It is not a certified medical diagnostic tool and should not be used for real clinical decision-making.
