# AI-Based ECG Analysis

A deep learning-powered system for automated ECG interpretation and cardiovascular risk prediction. This project provides a comprehensive web-based platform for doctors and medical staff to analyze ECG signals, detect cardiac abnormalities, and assess patient risk.

---

## 📖 About The Project

Cardiovascular diseases (CVDs) are a leading global cause of death. Early and accurate diagnosis is critical for effective treatment. This project automates the complex task of ECG analysis using advanced deep learning models to assist medical professionals.

The system aims to:
*   Detect various arrhythmias, signs of heart attacks, and other cardiac abnormalities from raw ECG signals.
*   Predict the 10-year cardiovascular disease risk using the Framingham Risk Score.
*   Generate comprehensive PDF reports for review by doctors and for patient records.
*   Provide a real-time monitoring dashboard for visualizing ECG data streams.

![Project Screenshot](https://via.placeholder.com/800x450.png?text=Add+Project+Screenshot+Here)
*(Add a screenshot of your web dashboard here)*

## ✨ Key Features

### 1. ECG Signal Processing
*   **Noise Removal:** Employs Butterworth and Wavelet filters to clean raw ECG signals.
    ```python
    from neurokit2 import ecg_clean
    cleaned_ecg = ecg_clean(raw_signal, sampling_rate=100)
    ```
*   **Feature Extraction:** Uses the Pan-Tompkins algorithm for robust R-peak and QRS complex detection.
*   **Vital Sign Calculation:** Computes Heart Rate Variability (HRV), QT intervals, and PR intervals.

### 2. AI Classification (CNN Model)
*   A Convolutional Neural Network (CNN) model classifies ECG signals into multiple categories:
    - Normal
    - Supraventricular Tachycardia (SVT)
    - Atrial Fibrillation
    - Ventricular Fibrillation
    - Ventricular Tachycardia
    - Heart Block
*   The model is trained on industry-standard datasets, including **MIT-BIH Arrhythmia** and **PTB-XL**.

### 3. Cardiovascular Risk Assessment
*   **Framingham Risk Score:** Calculates the 10-year risk of developing cardiovascular disease.
*   **GRACE Score:** Estimates mortality risk for patients post-heart attack.

### 4. Interactive Web Dashboard
*   **Real-time Visualization:** Uses Plotly to render interactive ECG charts.
*   **Secure Multi-user Portal:** A Flask-based backend with a secure login system for both doctors and staff, backed by a MySQL database.
*   **Patient Management:** Allows for creating, updating, and managing patient profiles and their associated ECG records.

## 🛠️ Tech Stack

| Category          | Technologies Used                                     |
|-------------------|-------------------------------------------------------|
| AI / ML           | TensorFlow, Keras, Scikit-learn                       |
| ECG Processing    | NeuroKit2, WFDB                                       |
| Backend           | Flask, MySQL                                          |
| Visualization     | Plotly, Matplotlib                                    |
| Deployment        | Docker, AWS                                           |

