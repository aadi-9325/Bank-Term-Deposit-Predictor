# Bank Term Deposit Prediction System

An end-to-end Machine Learning project that predicts whether a bank customer will subscribe to a term deposit using marketing campaign data. This project includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and deployment using Streamlit.

---

## Project Overview

This project uses the **Bank Marketing Dataset** to build a predictive machine learning model that helps banks identify potential customers likely to subscribe to a term deposit.

The application provides:

* Customer subscription prediction
* Probability-based prediction output
* Interactive Streamlit web interface
* Real-time inference using trained ML model

---

## Features

* Exploratory Data Analysis (EDA)
* Data preprocessing and feature engineering
* Logistic Regression based prediction model
* ROC-AUC evaluation metric
* Streamlit deployment
* Interactive UI for customer input
* Real-time probability prediction

---

## Tech Stack

### Languages & Libraries

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit
* Pickle

---

## Dataset

Dataset Used:

* Bank Marketing Dataset

Source:

* https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv

---

## Machine Learning Workflow

### Data Preprocessing

* Handling categorical variables
* Encoding features
* Feature scaling
* Train/CV/Test split

### Models Explored

* Logistic Regression
* KNN
* SVM
* Random Forest

### Evaluation Metrics

* ROC-AUC Score
* Accuracy
* Precision
* Recall
* F1 Score

---

## Project Structure

```bash
Bank-Term-Deposit-Prediction/
│
├── app.py
├── final_bank_model.pkl
├── response_encoder_dict.pkl
├── requirements.txt
├── README.md
└── notebooks/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/bank-term-deposit-prediction.git
```

Move into the project directory:

```bash
cd bank-term-deposit-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Streamlit App

```bash
streamlit run app.py
```

---

## Streamlit Deployment

This project is deployed using Streamlit Community Cloud.

Deployment Steps:

1. Push project to GitHub
2. Connect GitHub repository to Streamlit Cloud
3. Select `app.py`
4. Deploy

---

## Screenshots

### Home Page

* Customer information input form
* Economic indicators section
* Prediction button

### Prediction Output

* Subscription probability
* Prediction status

---

## Model Performance

| Model               | ROC-AUC Score |
| ------------------- | ------------- |
| Logistic Regression | 0.93          |
| KNN                 | Evaluated     |
| SVM                 | Evaluated     |

---

## Key Learnings

* End-to-end ML workflow
* Feature engineering
* Model evaluation using ROC-AUC
* Streamlit deployment
* Debugging deployment issues
* Handling categorical variables
* Real-world ML project structure

---

## Future Improvements

* Add SHAP Explainability
* Batch CSV prediction
* Model comparison dashboard
* Docker deployment
* Cloud deployment using AWS/GCP
* Better UI/UX

---

## Author

Aadi Ghodke

---

## License

This project is for educational and learning purposes.
