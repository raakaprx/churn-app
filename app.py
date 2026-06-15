import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# ===============================
# CUSTOM CSS
# ===============================
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.metric-card {
    background-color: #1f2937;
    padding: 20px;
    border-radius: 15px;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load("churn_model.pkl")

# ===============================
# TRAINING COLUMNS
# ===============================
TRAINING_COLUMNS = [
    'SeniorCitizen',
    'tenure',
    'MonthlyCharges',
    'TotalCharges',
    'gender_Female',
    'gender_Male',
    'InternetService_DSL',
    'InternetService_Fiber optic',
    'InternetService_No',
    'PaymentMethod_Bank transfer (automatic)',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check',
    'Partner_No',
    'Partner_Yes',
    'Dependents_No',
    'Dependents_Yes',
    'PhoneService_No',
    'PhoneService_Yes',
    'PaperlessBilling_No',
    'PaperlessBilling_Yes',
    'MultipleLines_No',
    'MultipleLines_Yes',
    'OnlineSecurity_No',
    'OnlineSecurity_Yes',
    'OnlineBackup_No',
    'OnlineBackup_Yes',
    'DeviceProtection_No',
    'DeviceProtection_Yes',
    'TechSupport_No',
    'TechSupport_Yes',
    'StreamingTV_No',
    'StreamingTV_Yes',
    'StreamingMovies_No',
    'StreamingMovies_Yes',
    'tenure_range',
    'Contract_Month-to-month',
    'Contract_One year',
    'Contract_Two year'
]

# ===============================
# PREPROCESS
# ===============================
def preprocess(df):

    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'],
        errors='coerce'
    )

    df.replace(
        ['No internet service', 'No phone service'],
        'No',
        inplace=True
    )

    cat_cols = [
        'gender',
        'Partner',
        'Dependents',
        'PhoneService',
        'MultipleLines',
        'InternetService',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies',
        'PaperlessBilling',
        'PaymentMethod',
        'Contract'
    ]

    df = pd.concat(
        [df, pd.get_dummies(df[cat_cols])],
        axis=1
    )

    df.drop(columns=cat_cols, inplace=True)

    condition = [
        ((df.tenure >= 0) & (df.tenure <= 12)),
        ((df.tenure > 12) & (df.tenure <= 24)),
        ((df.tenure > 24) & (df.tenure <= 36)),
        ((df.tenure > 36) & (df.tenure <= 48)),
        ((df.tenure > 48) & (df.tenure <= 60)),
        ((df.tenure > 60))
    ]

    choice = [0,1,2,3,4,5]

    df['tenure_range'] = np.select(
        condition,
        choice
    )

    df['MonthlyCharges'] = np.log1p(df['MonthlyCharges'])
    df['TotalCharges'] = np.log1p(df['TotalCharges'])

    return df


# ===============================
# SIDEBAR
# ===============================
with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
        width=120
    )

    st.title("Customer Churn")

    st.markdown("---")

    st.info("""
    Upload dataset customer CSV
    untuk memprediksi customer churn.
    """)

# ===============================
# HEADER
# ===============================
st.title("📊 Customer Churn Prediction Dashboard")

st.markdown("""
Dashboard untuk memprediksi pelanggan yang berpotensi berhenti menggunakan layanan.
""")

st.divider()

# ===============================
# UPLOAD
# ===============================
uploaded_file = st.file_uploader(
    "📂 Upload Dataset CSV",
    type=["csv"]
)

# ===============================
# PROCESS
# ===============================
if uploaded_file:

    raw_df = pd.read_csv(uploaded_file)

    st.subheader("📋 Dataset Preview")

    st.dataframe(raw_df.head())

    processed_df = preprocess(raw_df.copy())

    processed_df = processed_df.drop(
        columns=["customerID", "Churn"],
        errors="ignore"
    )

    # samakan kolom dengan training
    processed_df = processed_df.reindex(
        columns=TRAINING_COLUMNS,
        fill_value=False
    )

    # handle NaN
    processed_df = processed_df.fillna(0)

    prediction = model.predict(processed_df)

    probability = model.predict_proba(processed_df)

    result = raw_df.copy()

    result["Prediction"] = prediction

    result["Churn Probability (%)"] = (
        probability[:,1] * 100
    ).round(2)

    total = len(result)

    churn = (prediction == 1).sum()

    retained = (prediction == 0).sum()

    # ==========================
    # KPI
    # ==========================
    st.subheader("📈 Summary")

    col1,col2,col3 = st.columns(3)

    col1.metric(
        "👥 Total Customer",
        total
    )

    col2.metric(
        "⚠️ Churn",
        churn
    )

    col3.metric(
        "✅ Retained",
        retained
    )

    st.divider()

    # ==========================
    # CHART
    # ==========================
    chart_df = pd.DataFrame({
        "Status":["Churn","Retained"],
        "Count":[churn,retained]
    })

    c1,c2 = st.columns(2)

    with c1:

        fig_bar = px.bar(
            chart_df,
            x="Status",
            y="Count",
            color="Status",
            title="Customer Distribution"
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )

    with c2:

        fig_pie = px.pie(
            chart_df,
            names="Status",
            values="Count",
            hole=0.5,
            title="Churn Percentage"
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True
        )

    st.divider()

    # ==========================
    # TOP RISK CUSTOMER
    # ==========================
    st.subheader("🚨 Top 10 High Risk Customer")

    risk_df = result.sort_values(
        "Churn Probability (%)",
        ascending=False
    )

    st.dataframe(
        risk_df.head(10)
    )

    # ==========================
    # RESULT TABLE
    # ==========================
    st.subheader("📄 Prediction Result")

    st.dataframe(result)

    # ==========================
    # DOWNLOAD
    # ==========================
    csv = result.to_csv(index=False)

    st.download_button(
        "📥 Download Result CSV",
        csv,
        "prediction_result.csv",
        "text/csv"
    )