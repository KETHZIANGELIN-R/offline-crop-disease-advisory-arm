
# Offline Crop Disease Detection and Advisory

**Bharat AI-SoC Student Challenge 2026–27**  
**Problem Statement 1**

## 1. Project Overview

This project is a prototype for detecting crop leaf diseases using an AI image classification model and providing basic farmer advisory in English and Tamil.

The application is designed to perform local model inference after the initial model download.

## 2. Key Features

- Crop leaf image upload
- AI-based disease classification
- 38 PlantVillage disease and healthy-condition classes
- Model output score display
- English and Tamil language support
- Basic rule-based farmer advisory
- Automatic first-time model download
- Local inference after model download

## 3. Technology Stack

- Python
- Streamlit
- TensorFlow
- MobileNetV2
- NumPy
- Pillow

## 4. Model Information

Model: MobileNetV2  
Dataset label set: PlantVillage (38 classes)

Model download:
https://huggingface.co/ashikaasriarun/MobileNet-V2-PlantVillage/resolve/main/best_mnv2_pv_original.keras

The model is downloaded automatically on the first application run and stored locally in the `models` directory.

## 5. Project Structure

```text
offline-crop-disease-advisory-arm/
│
├── app.py
├── requirements.txt
├── README.md
│
├── documentation/
│   └── labels.json
│
└── models/
    └── plant_disease_model.keras
```

The model file is downloaded automatically and does not need to be uploaded to GitHub.

## 6. Installation and Execution

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

Internet access is required for the initial model download. After downloading the model and installing dependencies, local inference can run without internet.

## 7. Current Implementation Status

This is a laptop-based working prototype.

Implemented:
- Image upload
- MobileNetV2 model inference
- Predicted crop and condition display
- English and Tamil advisory
- Local model storage

## 8. Limitations

- The prototype has not undergone formal accuracy evaluation.
- The displayed model score is not a verified accuracy measurement.
- The 40% threshold is an experimental prototype setting.
- Advisory messages are general precautions, not confirmed medical or agricultural diagnoses.
- Raspberry Pi 5, Arm hardware and AI accelerator deployment have not yet been implemented or tested.

## 9. Future Scope

- Deployment on Arm-based edge devices
- Raspberry Pi 5 integration
- Model optimization for edge inference
- Field image validation
- Expanded regional-language advisory
- Improved disease classification and evaluation

## 10. Disclaimer

This application provides preliminary AI predictions for demonstration and educational purposes. Farmers should consult qualified agricultural experts before taking treatment decisions.
