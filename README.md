# Brain Tumor Detection System

This project is a deep-learning-based brain tumor detection system that classifies human brain MRI images into one of four categories. It uses a VGG16 model and a Streamlit web application to provide predictions from uploaded MRI images.

## Dataset

The dataset contains **7,200 human brain MRI images** divided into four classes:

- Glioma
- Meningioma
- Pituitary tumor
- No tumor

## Model Performance

The VGG16 model achieved a **test accuracy of 91.56%** (approximately **92%**) on unseen test images.

The accuracy was calculated as follows:

```text
Correct predictions: 1,465
Total test images: 1,600
Accuracy: 1,465 / 1,600 = 91.56%
```

## Project Files

- `app.py` — Streamlit web application
- `requirements.txt` — Python dependencies
- `vgg16_brain_tumor.keras` — trained VGG16 model

## Run Locally

1. Clone this repository.
2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the Streamlit application:

   ```bash
   streamlit run app.py
   ```

4. Upload a JPG or PNG brain MRI image to receive a predicted class and confidence score.

## Disclaimer

This project is intended for educational and research purposes only. It is not a medical diagnostic system and must not replace evaluation by a qualified medical professional.
