# Disease Prediction and Drug Recommendation

## Overview

This project is a web-based machine learning application for predicting diseases and recommending drugs. It uses the Random Forest Classifier to make predictions based on user input and integrates the Gemini API to provide relevant information about predicted diseases, including causes, symptoms, and possible treatments.

---

## Features

- **Disease Prediction:** Predicts possible diseases from user-entered symptoms using trained ML models.
- **Drug Recommendation:** Suggests drugs and treatments for diagnosed diseases.
- **Gemini API Integration:** Fetches detailed disease information (causes, symptoms, treatments).
- **Interactive Web Interface:** Built with HTML, CSS, and JavaScript for user-friendly experience.
- **Jupyter Notebooks:** Contains exploratory data analysis, model training, and evaluation.
- **Extensible Design:** Easy to add more diseases, drugs, or models.

---

## Tech Stack

- **Machine Learning:** Python (Random Forest Classifier, scikit-learn, pandas, numpy)
- **Web Frontend:** HTML, CSS, JavaScript
- **Backend:** Python (Flask)
- **API:** Gemini API
- **Notebook Environment:** Jupyter Notebook

---

## Getting Started

### Prerequisites

- Python >= 3.7
- Jupyter Notebook
- pip (Python package manager)
- Gemini API credentials (if deploying API features)

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Kuldeep-Tapodhan/Disease-prediction-and-drug-recommendation.git
    cd Disease-prediction-and-drug-recommendation
    ```

2. **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **(Optional) Set up Gemini API:**
    - Register and obtain API credentials.
    - Add credentials to your environment or configuration file as instructed in the code.

### Running the Project

#### 1. Run Jupyter Notebooks

- Open Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
- Explore the notebooks for data exploration, model training, and evaluation.

#### 2. Run the Web Application

- Execute the main Python script:
    ```bash
    python main1.py
    ```
- Open your browser and go to `http://localhost:5000` (or as shown in script output).

---

## Usage

1. **Enter symptoms** in the web interface.
2. **View predicted disease** and recommended drugs.
3. **Access additional details** about the disease via Gemini API integration.

---


## Dataset

- The project uses medical datasets (symptoms, diseases, drugs) for training and inference.
- See `data/` directory and related notebooks for details about preprocessing and feature engineering.

---

## Model Details

- **Random Forest Classifier**: Used for multi-class disease prediction.
- **Training & Evaluation**: Included in Jupyter notebooks.
- **Performance Metrics**: Accuracy, precision, recall, F1-score.

---

## API Integration

- **Gemini API**: Retrieves contextual information about diseases.
- **Configuration**: See code comments for how to set up API access.

---


## Acknowledgements

- Gemini API for medical data
- Scikit-learn for machine learning utilities
- Contributors and dataset providers

---

