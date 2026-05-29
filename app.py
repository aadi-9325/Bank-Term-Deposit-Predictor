import streamlit as st
import pandas as pd
import pickle
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Bank Term Deposit Predictor",
    page_icon=":material/account_balance:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        :root {
            --bg: #f6f7fb;
            --panel: #ffffff;
            --ink: #1f2937;
            --muted: #64748b;
            --line: #e2e8f0;
            --accent: #0f766e;
            --accent-soft: #ccfbf1;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(20, 184, 166, 0.12), transparent 30rem),
                linear-gradient(180deg, #f8fafc 0%, var(--bg) 100%);
            color: var(--ink);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2.25rem;
            padding-bottom: 3rem;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .app-hero {
            padding: 1.5rem 0 1.1rem;
            border-bottom: 1px solid var(--line);
            margin-bottom: 1.25rem;
        }

        .app-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            color: var(--accent);
            background: var(--accent-soft);
            border: 1px solid rgba(15, 118, 110, 0.18);
            border-radius: 999px;
            padding: 0.35rem 0.7rem;
            font-size: 0.84rem;
            font-weight: 700;
            margin-bottom: 0.9rem;
        }

        .app-title {
            font-size: clamp(2rem, 4vw, 3.35rem);
            line-height: 1.05;
            font-weight: 800;
            letter-spacing: 0;
            color: #111827;
            margin: 0;
        }

        .app-subtitle {
            color: var(--muted);
            font-size: 1.05rem;
            max-width: 760px;
            margin: 0.8rem 0 0;
        }

        .section-label {
            color: #0f172a;
            font-size: 1.02rem;
            font-weight: 750;
            margin: 1rem 0 0.25rem;
        }

        div[data-testid="stForm"] {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1.2rem;
            box-shadow: 0 18px 55px rgba(15, 23, 42, 0.08);
        }

        div[data-testid="stForm"] label p {
            color: #334155;
            font-size: 0.9rem;
            font-weight: 650;
        }

        div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] input {
            border-radius: 7px;
        }

        .stButton > button {
            width: 100%;
            min-height: 3.05rem;
            border-radius: 7px;
            border: 0;
            background: linear-gradient(135deg, #0f766e 0%, #2563eb 100%);
            color: white;
            font-weight: 800;
            letter-spacing: 0;
            box-shadow: 0 14px 28px rgba(37, 99, 235, 0.22);
        }

        .stButton > button:hover {
            border: 0;
            color: white;
            filter: brightness(1.03);
        }

        .result-card {
            background: #ffffff;
            border: 1px solid var(--line);
            border-left: 5px solid var(--accent);
            border-radius: 8px;
            padding: 1rem 1.15rem;
            box-shadow: 0 14px 38px rgba(15, 23, 42, 0.08);
            margin-top: 1.2rem;
        }

        .result-label {
            color: var(--muted);
            font-size: 0.86rem;
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.2rem;
        }

        .result-value {
            color: #0f172a;
            font-size: clamp(1.8rem, 3.2vw, 2.6rem);
            font-weight: 850;
            line-height: 1.1;
            margin: 0;
        }

        .result-copy {
            color: var(--muted);
            font-size: 0.98rem;
            margin-top: 0.5rem;
        }

        @media (max-width: 640px) {
            .block-container {
                padding: 1.1rem 0.8rem 2rem;
            }

            div[data-testid="stForm"] {
                padding: 0.9rem;
            }

            .app-hero {
                padding-top: 0.7rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# 1. Load the model and encoder
@st.cache_resource
def load_assets():
    with open(APP_DIR / 'final_bank_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(APP_DIR / 'response_encoder_dict.pkl', 'rb') as f:
        encoder = pickle.load(f)

    # The saved XGBoost model was trained with device='cuda'. On this machine the
    # Streamlit app is serving CPU inputs, so force every saved estimator to CPU.
    for calibrated_classifier in getattr(model, "calibrated_classifiers_", []):
        estimator = getattr(calibrated_classifier, "estimator", None)
        if hasattr(estimator, "set_params"):
            estimator.set_params(device="cpu")

    return model, encoder

model, feat_dict = load_assets()

def transform_input(raw_data, encoder_dict, feature_columns):
    # Numerical features (same as training order)
    # We use the dots here as they match the training feature names
    numerical_cols = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']

    features = {col: raw_data[col] for col in numerical_cols}

    # Add response coded categorical features (Class 0 and Class 1 probabilities)
    for col in categorical_cols:
        val = raw_data[col]
        # Get prob vector from dict, or [0.5, 0.5] if unknown
        prob_vec = encoder_dict.get(col, {}).get(val, [0.5, 0.5])
        features[f"{col}_0"] = prob_vec[0]
        features[f"{col}_1"] = prob_vec[1]

    return pd.DataFrame([[features[col] for col in feature_columns]], columns=feature_columns)


def predict_subscription_probability(model, transformed_x):
    class_labels = list(model.classes_)
    positive_class_index = class_labels.index(1)
    prediction_prob = model.predict_proba(transformed_x)[0, positive_class_index]
    prediction = model.predict(transformed_x)[0]
    return prediction_prob, prediction

st.markdown(
    """
    <section class="app-hero">
        <div class="app-kicker">Marketing intelligence</div>
        <h1 class="app-title">Bank Term Deposit Predictor</h1>
        <p class="app-subtitle">
            Estimate how likely a client is to subscribe using campaign history,
            customer profile details, and current economic indicators.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

# Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-label">Client Profile</div>', unsafe_allow_html=True)
        age = st.number_input("Age", 18, 100, 35)
        job = st.selectbox("Job", ['admin.', 'blue-collar', 'technician', 'services', 'management', 'retired', 'entrepreneur', 'self-employed', 'housemaid', 'unemployed', 'student', 'unknown'])
        marital = st.selectbox("Marital Status", ['married', 'single', 'divorced', 'unknown'])
        education = st.selectbox("Education", ['university.degree', 'high.school', 'basic.9y', 'professional.course', 'basic.4y', 'basic.6y', 'unknown', 'illiterate'])
        default = st.selectbox("Credit in Default?", ['no', 'unknown', 'yes'])
        housing = st.selectbox("Housing Loan?", ['yes', 'no', 'unknown'])
        loan = st.selectbox("Personal Loan?", ['no', 'yes', 'unknown'])

    with col2:
        st.markdown('<div class="section-label">Campaign Details</div>', unsafe_allow_html=True)
        contact = st.selectbox("Contact Type", ['cellular', 'telephone'])
        month = st.selectbox("Month", ['may', 'jul', 'aug', 'jun', 'nov', 'apr', 'oct', 'sep', 'mar', 'dec'])
        day_of_week = st.selectbox("Day of Week", ['mon', 'tue', 'wed', 'thu', 'fri'])
        poutcome = st.selectbox("Previous Outcome", ['nonexistent', 'failure', 'success'])
        campaign = st.number_input("Campaign Contacts", 1, 50, 1)
        pdays = st.number_input("Days since last contact (999=Never)", 0, 999, 999)
        previous = st.number_input("Previous Contacts", 0, 10, 0)

    st.markdown('<div class="section-label">Economic Indicators</div>', unsafe_allow_html=True)
    e_col1, e_col2, e_col3 = st.columns(3)
    emp_var_rate = e_col1.number_input("Emp. Var. Rate", value=1.1)
    cons_price_idx = e_col2.number_input("Cons. Price Index", value=93.99)
    cons_conf_idx = e_col3.number_input("Cons. Conf. Index", value=-36.4)
    euribor3m = e_col1.number_input("Euribor 3M", value=4.85)
    nr_employed = e_col2.number_input("Nr. Employed", value=5191.0)

    submit = st.form_submit_button("Predict Subscription")

if submit:
    input_data = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'month': month,
        'day_of_week': day_of_week,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome,
        'emp.var.rate': emp_var_rate,
        'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx,
        'euribor3m': euribor3m,
        'nr.employed': nr_employed
    }

    try:
        feature_columns = [str(col) for col in model.feature_names_in_]
        transformed_x = transform_input(input_data, feat_dict, feature_columns)
        prediction_prob, prediction = predict_subscription_probability(model, transformed_x)

        if prediction == 1:
            status_text = "Target audience: high likelihood of subscription."
            accent = "#0f766e"
        else:
            status_text = "Low likelihood of subscription."
            accent = "#2563eb"

        st.markdown(
            f"""
            <div class="result-card" style="border-left-color: {accent};">
                <div class="result-label">Subscription Probability</div>
                <p class="result-value">{prediction_prob:.2%}</p>
                <div class="result-copy">{status_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"Prediction Error: {e}")
