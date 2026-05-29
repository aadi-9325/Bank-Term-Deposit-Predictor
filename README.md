# Bank Term Deposit Predictor

A responsive Streamlit app that predicts whether a bank client is likely to subscribe to a term deposit after a direct-marketing campaign.

The original case study describes Portuguese banking direct-marketing phone-call data where the target is whether the client subscribed to a term deposit. It uses the UCI `bank-additional-full.csv` dataset with 41,188 records, 20 independent variables, and a binary target class of `yes` or `no`. The reference project also identifies AUC ROC as the main evaluation metric.

## Project Files

- `app.py` - Streamlit web app with the prediction form, preprocessing logic, and UI styling.
- `final_bank_model.pkl` - trained calibrated machine-learning model.
- `response_encoder_dict.pkl` - response-encoding dictionary for categorical input fields.
- `requirements.txt` - Python dependencies needed to run the app.

## Features

- Clean responsive Streamlit interface.
- Client profile, campaign, and economic-indicator input sections.
- Response encoding for categorical variables.
- Model feature alignment using the saved model's expected `feature_names_in_`.
- Subscription probability output with a clear high/low likelihood message.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run The App

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Input Variables

The app accepts customer, campaign, and economic indicators similar to the bank marketing dataset:

- Customer profile: age, job, marital status, education, default, housing loan, personal loan.
- Campaign details: contact type, month, day of week, campaign contacts, previous contacts, previous outcome, days since last contact.
- Economic indicators: employment variation rate, consumer price index, consumer confidence index, Euribor 3M, number of employed.

## Model Output

The app returns:

- Probability of subscription.
- Predicted class:
  - High likelihood of subscription.
  - Low likelihood of subscription.

## Notes

- The saved model was trained with an XGBoost estimator. The app forces the loaded estimator to CPU mode so predictions run reliably on local machines without CUDA.
- The app builds a named `pandas.DataFrame` in the exact feature order expected by the saved model, helping prevent mismatches between training-time and app-time predictions.
