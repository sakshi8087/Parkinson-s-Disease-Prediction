#data-science-project

# Parkinson's Disease Prediction

This project aims to build a machine learning model to predict the presence and severity of Parkinson's Disease based on patient data. It includes a web-based user interface for easy input of medical parameters, and the prediction is made using machine learning models.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Description](#model-description)


## Project Overview
Parkinson's Disease is a progressive disorder that affects movement. Early detection and accurate diagnosis are critical for treatment. This project uses machine learning algorithms to predict the level of Parkinson's Disease in a patient based on relevant medical parameters.

The main features of the project include:
- A user-friendly web interface to input patient data.
- Backend machine learning model that predicts the severity of the disease.
- Data preprocessing and feature selection to ensure optimal model performance.

## Technologies Used
- **Programming Language**: Python
- **Web Framework**: Flask
- **Machine Learning Libraries**: Xboost, Pandas, NumPy
- **Frontend**: HTML, CSS, JavaScript
- **Visualization**: Matplotlib, Seaborn

## Dataset
The dataset used in this project contains medical data points such as voice measurements and motor functions associated with Parkinson’s disease. The dataset consists of multiple features, including:
- Jitter (%)
- Jitter (Abs)
- Shimmer
- HNR (Harmonics to Noise Ratio)
- RPDE (Recurrence Period Density Entropy)
- DFA (Detrended Fluctuation Analysis)

You can download the dataset from [UCI Parkinson's Dataset Repository](https://archive.ics.uci.edu/ml/datasets/parkinsons).

## Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/parkinsons-prediction.git
    cd parkinsons-prediction
    ```
2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
  
    ```
4. Run the Flask/Django app:
    ```bash
    flask run  # or `python manage.py runserver` if using Django
    ```

## Usage
1. Open the application in a web browser at `http://127.0.0.1:5000/` (for Flask) 
2. Enter the relevant medical parameters in the input fields.
3. Click on the "Predict" button to get the prediction result.
4. The system will output the likelihood and severity of Parkinson's Disease based on the input.

## Model Description
The machine learning model is trained using supervised learning algorithms, including:
- **Random Forest Classifier**: For high accuracy and feature importance extraction.
- **Support Vector Machines (SVM)**: To classify Parkinson's disease severity.
- **K-Nearest Neighbors (KNN)**: As a baseline model for classification.

Feature selection was performed using techniques like **Information Gain** and **Chi-square** to ensure the most relevant medical features were used for prediction.



