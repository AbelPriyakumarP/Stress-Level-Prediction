# Student Stress Level Predictor

![image alt](https://github.com/AbelPriyakumarP/Stress-Level-Prediction/blob/feaf1667136dabebacde7f74feaa171e60a1c14f/Screenshot%202025-05-11%20214616.png)

A **Streamlit-based web application** that predicts a student's stress level (**Low**, **Moderate**, or **High**) using a pre-trained **Random Forest Classifier**. The app leverages a dataset of student activities and GPA to provide insights into stress triggers, with features like prediction probabilities and feature importance visualization.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview
The **Student Stress Level Predictor** is a machine learning-powered web application built with **Streamlit**. It predicts a student's stress level based on six numerical features: **Study Hours Per Day**, **Extracurricular Hours Per Day**, **Sleep Hours Per Day**, **Social Hours Per Day**, **Physical Activity Hours Per Day**, and **GPA**. The app uses a pre-trained **RandomForestClassifier**, **MinMaxScaler**, and **LabelEncoder** to deliver accurate predictions and insights.

Key goals:
- Provide an intuitive interface for users to input student data and receive stress level predictions.
- Visualize prediction probabilities and feature importance to explain model decisions.
- Offer actionable insights into factors driving student stress, such as GPA and study hours.

## Features
- **Interactive Input Form**: Users enter values for six features via sliders, with validation (e.g., total hours ≤ 24, GPA ≤ 4.0).
- **Categorical Prediction**: Predicts stress level as **Low**, **Moderate**, or **High**.
- **Prediction Probabilities**: Displays probabilities for each stress level in the order Low, Moderate, High.
- **Feature Importance**: Visualizes which features (e.g., GPA, Study Hours) most influence predictions using a bar plot.
- **Input Validation**: Ensures realistic inputs, such as capping Physical Activity at 12.9 hours (based on outlier analysis).
- **Error Handling**: Robust checks for file loading, scaling, and prediction errors.

## Architecture
The project follows a modular architecture integrating data preprocessing, machine learning, and web development:

```plaintext
+-------------------+       +-------------------+       +-------------------+
|   Dataset         | ----> |   Preprocessing   | ----> |   ML Model        |
| (dataset_.csv)    |       | (MinMaxScaler,    |       | (RandomForest     |
|                   |       |  LabelEncoder)    |       |  Classifier)      |
+-------------------+       +-------------------+       +-------------------+
                                                        |
                                                        v
+-------------------+       +-------------------+       +-------------------+
|   Model Saving    | <---- |   Streamlit App   | <---- |   Predictions     |
| (Joblib: .pkl)    |       | (app.py)          |       | & Visualizations  |
+-------------------+       +-------------------+       +-------------------+

Data Layer: Loads dataset_.csv with six numerical features and a categorical target (Stress_Level).
Preprocessing Layer: Applies MinMaxScaler to scale features and LabelEncoder to encode Stress_Level (Low, Moderate, High).
Model Layer: Trains a RandomForestClassifier on scaled features to predict encoded stress levels.
Serialization Layer: Saves the model, scaler, and encoder as .pkl files using Joblib.
Application Layer: A Streamlit app (app.py) loads the saved files, collects user inputs, scales them, predicts stress levels, and visualizes results.
Dataset
The dataset (dataset_.csv) contains 2000 entries (after removing a header row) with the following columns:

Study_Hours_Per_Day: Hours spent studying (0–24).
Extracurricular_Hours_Per_Day: Hours spent on extracurricular activities (0–24).
Sleep_Hours_Per_Day: Hours slept (0–24).
Social_Hours_Per_Day: Hours spent socializing (0–24).
Physical_Activity_Hours_Per_Day: Hours spent on physical activity (0–12.9, capped due to outliers).
GPA: Grade Point Average (0–4.0).
Stress_Level: Target variable (Low, Moderate, High).
Key insights from analysis:

GPA, Study_Hours_Per_Day, and Social_Hours_Per_Day are the most influential predictors (per Random Forest feature importance).
Outliers in Physical_Activity_Hours_Per_Day were capped at 12.9 using IQR analysis.
Note: The dataset is not included in this repository due to size or privacy constraints. Ensure you have dataset_.csv for training or use the provided pre-trained model files.

Installation
Follow these steps to set up the project locally:

Clone the Repository:
bash

Copy
git clone https://github.com/your-username/stress-level-predictor.git
cd stress-level-predictor
Create a Virtual Environment (recommended):
bash

Copy
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:
bash

Copy
pip install -r requirements.txt
The requirements.txt includes:
text

Copy
streamlit==1.29.0
pandas==2.2.2
numpy==2.0.2
joblib==1.4.2
matplotlib==3.9.2
scikit-learn==1.5.1
Place Model Files:
Ensure random_forest_model.pkl, minmax_scaler.pkl, and label_encoder.pkl are in the project root directory. These are provided in the repository or generated from the Jupyter notebook (Untitled74 (1).ipynb).
Usage
Run the Streamlit App:
bash

Copy
streamlit run app.py
This starts a local server, typically at http://localhost:8501.
Interact with the App:
Open the provided URL in a web browser.
Enter values for the six features using the input form:
Study Hours Per Day: 0–24 (e.g., 8.0)
Extracurricular Hours Per Day: 0–24 (e.g., 2.0)
Sleep Hours Per Day: 0–24 (e.g., 7.0)
Social Hours Per Day: 0–24 (e.g., 3.0)
Physical Activity Hours Per Day: 0–12.9 (e.g., 4.0)
GPA: 0–4.0 (e.g., 3.5)
Click Predict Stress Level to view:
The predicted stress level (e.g., "Moderate").
Prediction probabilities (e.g., Low: 20%, Moderate: 50%, High: 30%).
A feature importance plot showing key predictors (e.g., GPA, Study Hours).
Input Validation:
The app ensures total hours (Study + Extracurricular + Sleep + Social + Physical Activity) do not exceed 24.
GPA is capped at 4.0, and Physical Activity is capped at 12.9 to align with the training data.
Project Structure
text

Copy
stress-level-predictor/
│
├── app.py                    # Streamlit application code
├── Untitled74 (1).ipynb      # Jupyter notebook for data analysis and model training
├── random_forest_model.pkl    # Pre-trained Random Forest model
├── minmax_scaler.pkl         # Pre-trained MinMaxScaler
├── label_encoder.pkl         # Pre-trained LabelEncoder
├── requirements.txt          # Project dependencies
├── assets/                   # Static files (e.g., screenshots)
│   └── screenshot.png
├── README.md                 # Project documentation
Model Details
Algorithm: RandomForestClassifier (from scikit-learn).
Features: 6 numerical features (Study_Hours_Per_Day, Extracurricular_Hours_Per_Day, Sleep_Hours_Per_Day, Social_Hours_Per_Day, Physical_Activity_Hours_Per_Day, GPA).
Target: Stress_Level (categorical: Low, Moderate, High).
Preprocessing:
MinMaxScaler: Scales features to [0, 1] for consistent model input.
LabelEncoder: Encodes Stress_Level (Low → 0, Moderate → 1, High → 2) for training; inverse-transformed for predictions.
Feature Importance (from notebook):
GPA: Highest importance, indicating academic performance strongly influences stress.
Study_Hours_Per_Day and Social_Hours_Per_Day: Significant contributors.
Sleep_Hours_Per_Day, Extracurricular_Hours_Per_Day, Physical_Activity_Hours_Per_Day: Lower impact.
Serialization: Model, scaler, and encoder saved with Joblib for deployment.
Note: Model performance metrics (e.g., accuracy, confusion matrix) are available in the notebook (Untitled74 (1).ipynb) but not displayed in the app. Future versions may include these.

Future Improvements
Model Metrics: Display accuracy, precision, recall, or confusion matrix in the app.
Enhanced Visualizations: Add a bar chart for prediction probabilities.
Input Range Warnings: Alert users if inputs deviate from training data ranges (e.g., GPA < 2.25).
Export Functionality: Allow users to download predictions as CSV or PDF.
Model Tuning: Experiment with hyperparameters or other algorithms (e.g., XGBoost) for better performance.
Deployment: Host the app on Streamlit Cloud or Heroku for public access.
Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Make changes and commit (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a Pull Request.
Please ensure code follows PEP 8 standards and includes tests where applicable.

License
This project is licensed under the .

Acknowledgments
scikit-learn: For the RandomForestClassifier and preprocessing tools.
Streamlit: For the intuitive web app framework.
Pandas and NumPy: For efficient data handling.
Matplotlib: For visualization support.
Thanks to the open-source community for inspiration and resources.
Feel free to star ⭐ this repository if you find it useful! For questions or feedback, open an issue or contact me at [your-email@example.com].

text

Copy

### Instructions to Use
1. **Create a GitHub Repository**:
   - Initialize a repository (e.g., `stress-level-predictor`) on GitHub.
   - Push your project files (`app.py`, `Untitled74 (1).ipynb`, `random_forest_model.pkl`, `minmax_scaler.pkl`, `label_encoder.pkl`).

2. **Add the README**:
   - Create a file named `README.md` in the repository root.
   - Copy and paste the entire content above into `README.md`.
   - Save and commit the file.

3. **Create Supporting Files**:
   - **requirements.txt**:
     ```text
     streamlit==1.29.0
     pandas==2.2.2
     numpy==2.0.2
     joblib==1.4.2
     matplotlib==3.9.2
     scikit-learn==1.5.1
Save this as requirements.txt in the repository root.

Push to GitHub:
Commit and push all files:
bash

Copy
git add .
git commit -m "Initial commit with Streamlit app, README, and supporting files"
git push origin main
Optional Enhancements:
.gitignore: Create a .gitignore file to exclude unnecessary files:
text

Copy
venv/
__pycache__/
*.pyc
dataset_.csv
This excludes the virtual environment, Python cache, and the dataset (if private).
Screenshot: If you can’t create a screenshot yet, run the app locally, capture the interface, and add it to assets/.
Deployment: To make the app publicly accessible, deploy it to Streamlit Cloud:
Push the repository to GitHub.
Sign into Streamlit Cloud, connect your repo, and deploy app.py.
Add the deployed URL to the README under the Usage section (e.g., "Access the deployed app at [your-streamlit-url]").
