"""
Customer Churn Prediction Dashboard
====================================
A complete, portfolio-ready Streamlit app.
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark theme overrides */
    .main { background-color: #0e1117; }
    
    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #1a1d2e 0%, #16213e 100%);
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00d9c0;
        line-height: 1.1;
    }
    .kpi-label {
        font-size: 0.78rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 4px;
    }
    .kpi-delta {
        font-size: 0.82rem;
        color: #64ffda;
        margin-top: 6px;
    }

    /* Section headers */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ccd6f6;
        border-left: 3px solid #00d9c0;
        padding-left: 12px;
        margin: 24px 0 16px 0;
    }

    /* Prediction result */
    .churn-high {
        background: linear-gradient(135deg, #ff4b4b22, #ff6b6b11);
        border: 1px solid #ff4b4b66;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .churn-low {
        background: linear-gradient(135deg, #00d9c022, #00d9c011);
        border: 1px solid #00d9c066;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .prob-number {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1;
    }
    
    /* Insight cards */
    .insight-card {
        background: #1a1d2e;
        border: 1px solid #2d3561;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 6px 0;
        font-size: 0.88rem;
        color: #8892b0;
    }
    .insight-card strong { color: #ccd6f6; }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── Load assets ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model/churn_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/telco_churn.csv")

@st.cache_data
def load_processed():
    return pd.read_csv("data/processed_churn.csv")

@st.cache_data
def load_metrics():
    with open("model/metrics.json") as f:
        return json.load(f)

@st.cache_data
def load_feature_importances():
    with open("model/feature_importances.json") as f:
        return json.load(f)

@st.cache_data
def load_feature_names():
    return joblib.load("model/feature_names.pkl")

model       = load_model()
df_raw      = load_data()
df_proc     = load_processed()
metrics     = load_metrics()
feat_imp    = load_feature_importances()
feat_names  = load_feature_names()

# ─── Sidebar — Prediction Inputs ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Predict Churn")
    st.markdown("Enter a customer's details to get their churn probability.")
    st.markdown("---")

    # Demographic
    st.markdown("**👤 Demographics**")
    gender      = st.selectbox("Gender",       ["Male", "Female"])
    senior      = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner     = st.selectbox("Has Partner",  ["Yes", "No"])
    dependents  = st.selectbox("Has Dependents", ["No", "Yes"])
    tenure      = st.slider("Tenure (months)", 1, 72, 12)

    st.markdown("**📶 Services**")
    phone_svc   = st.selectbox("Phone Service", ["Yes", "No"])
    multi_lines = st.selectbox("Multiple Lines",
        ["No", "Yes", "No phone service"])
    internet    = st.selectbox("Internet Service",
        ["Fiber optic", "DSL", "No"])

    if internet != "No":
        online_sec   = st.selectbox("Online Security",    ["No", "Yes"])
        online_bkp   = st.selectbox("Online Backup",      ["No", "Yes"])
        device_prot  = st.selectbox("Device Protection",  ["No", "Yes"])
        tech_sup     = st.selectbox("Tech Support",       ["No", "Yes"])
        stream_tv    = st.selectbox("Streaming TV",       ["No", "Yes"])
        stream_mov   = st.selectbox("Streaming Movies",   ["No", "Yes"])
    else:
        online_sec = online_bkp = device_prot = tech_sup = \
            stream_tv = stream_mov = "No internet service"

    st.markdown("**💳 Account**")
    contract    = st.selectbox("Contract Type",
        ["Month-to-month", "One year", "Two year"])
    paperless   = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment     = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_ch  = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5)
    total_ch    = st.number_input("Total Charges ($)",
        min_value=18.0, max_value=9000.0,
        value=float(round(monthly_ch * tenure, 2)))

    predict_btn = st.button("⚡ Predict Churn", type="primary", use_container_width=True)

# ─── Helper: build feature vector ─────────────────────────────────────────────
def build_feature_vector():
    """Map sidebar inputs to model feature vector."""
    row = {
        "gender":           1 if gender == "Male" else 0,
        "SeniorCitizen":    1 if senior == "Yes" else 0,
        "Partner":          1 if partner == "Yes" else 0,
        "Dependents":       1 if dependents == "Yes" else 0,
        "tenure":           tenure,
        "PhoneService":     1 if phone_svc == "Yes" else 0,
        "MonthlyCharges":   monthly_ch,
        "TotalCharges":     total_ch,
    }

    # One-hot encoded columns (drop_first=True was used in training)
    # MultipleLines
    row["MultipleLines_No phone service"] = 1 if multi_lines == "No phone service" else 0
    row["MultipleLines_Yes"]              = 1 if multi_lines == "Yes" else 0

    # InternetService
    row["InternetService_Fiber optic"] = 1 if internet == "Fiber optic" else 0
    row["InternetService_No"]          = 1 if internet == "No" else 0

    # OnlineSecurity
    row["OnlineSecurity_No internet service"] = 1 if online_sec == "No internet service" else 0
    row["OnlineSecurity_Yes"]                 = 1 if online_sec == "Yes" else 0

    # OnlineBackup
    row["OnlineBackup_No internet service"] = 1 if online_bkp == "No internet service" else 0
    row["OnlineBackup_Yes"]                 = 1 if online_bkp == "Yes" else 0

    # DeviceProtection
    row["DeviceProtection_No internet service"] = 1 if device_prot == "No internet service" else 0
    row["DeviceProtection_Yes"]                 = 1 if device_prot == "Yes" else 0

    # TechSupport
    row["TechSupport_No internet service"] = 1 if tech_sup == "No internet service" else 0
    row["TechSupport_Yes"]                 = 1 if tech_sup == "Yes" else 0

    # StreamingTV
    row["StreamingTV_No internet service"] = 1 if stream_tv == "No internet service" else 0
    row["StreamingTV_Yes"]                 = 1 if stream_tv == "Yes" else 0

    # StreamingMovies
    row["StreamingMovies_No internet service"] = 1 if stream_mov == "No internet service" else 0
    row["StreamingMovies_Yes"]                 = 1 if stream_mov == "Yes" else 0

    # Contract
    row["Contract_One year"] = 1 if contract == "One year" else 0
    row["Contract_Two year"] = 1 if contract == "Two year" else 0

    # PaperlessBilling
    row["PaperlessBilling"] = 1 if paperless == "Yes" else 0

    # PaymentMethod
    row["PaymentMethod_Credit card (automatic)"] = 1 if payment == "Credit card (automatic)" else 0
    row["PaymentMethod_Electronic check"]        = 1 if payment == "Electronic check" else 0
    row["PaymentMethod_Mailed check"]            = 1 if payment == "Mailed check" else 0

    # Build df aligned to training columns
    x = pd.DataFrame([row])
    for col in feat_names:
        if col not in x.columns:
            x[col] = 0
    return x[feat_names]

# ─── Main Content ─────────────────────────────────────────────────────────────
st.markdown("# 📡 Customer Churn Prediction Dashboard")
st.markdown(
    "*Telco dataset · Gradient Boosting Classifier · AUC {:.3f}*".format(metrics["auc"])
)
st.markdown("---")

# ── KPI Cards ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{metrics['total_customers']:,}</div>
        <div class="kpi-label">Total Customers</div>
    </div>""", unsafe_allow_html=True)

with c2:
    pct = metrics['churn_rate'] * 100
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{pct:.1f}%</div>
        <div class="kpi-label">Churn Rate</div>
        <div class="kpi-delta">↑ High-risk segment</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">${metrics['avg_monthly_charges']:.0f}</div>
        <div class="kpi-label">Avg Monthly Charges</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{metrics['avg_tenure']:.0f}mo</div>
        <div class="kpi-label">Avg Customer Tenure</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔬 Feature Analysis", "📋 Data Explorer"])

# ════════════════════════ TAB 1: OVERVIEW ═════════════════════════════════════
with tab1:

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-title">Churn by Contract Type</div>',
                    unsafe_allow_html=True)
        contract_churn = df_raw.groupby("Contract")["Churn"].apply(
            lambda x: (x == "Yes").mean() * 100
        ).reset_index()
        contract_churn.columns = ["Contract", "Churn Rate (%)"]
        contract_churn["Churn Rate (%)"] = contract_churn["Churn Rate (%)"].round(1)

        # Simple bar chart via st.bar_chart after pivoting
        st.dataframe(
            contract_churn.style.background_gradient(cmap="RdYlGn_r", subset=["Churn Rate (%)"]),
            use_container_width=True, hide_index=True
        )

        st.markdown('<div class="section-title">Churn by Internet Service</div>',
                    unsafe_allow_html=True)
        inet_churn = df_raw.groupby("InternetService")["Churn"].apply(
            lambda x: (x == "Yes").mean() * 100
        ).reset_index()
        inet_churn.columns = ["Internet Service", "Churn Rate (%)"]
        inet_churn["Churn Rate (%)"] = inet_churn["Churn Rate (%)"].round(1)
        st.dataframe(
            inet_churn.style.background_gradient(cmap="RdYlGn_r", subset=["Churn Rate (%)"]),
            use_container_width=True, hide_index=True
        )

    with col_right:
        st.markdown('<div class="section-title">Churn by Payment Method</div>',
                    unsafe_allow_html=True)
        pay_churn = df_raw.groupby("PaymentMethod")["Churn"].apply(
            lambda x: (x == "Yes").mean() * 100
        ).reset_index()
        pay_churn.columns = ["Payment Method", "Churn Rate (%)"]
        pay_churn["Churn Rate (%)"] = pay_churn["Churn Rate (%)"].round(1)
        pay_churn = pay_churn.sort_values("Churn Rate (%)", ascending=False)
        st.dataframe(
            pay_churn.style.background_gradient(cmap="RdYlGn_r", subset=["Churn Rate (%)"]),
            use_container_width=True, hide_index=True
        )

        st.markdown('<div class="section-title">Tenure Distribution by Churn</div>',
                    unsafe_allow_html=True)
        tenure_churn = df_raw.groupby("Churn")["tenure"].describe()[["mean", "50%", "min", "max"]].round(1)
        tenure_churn.columns = ["Mean", "Median", "Min", "Max"]
        tenure_churn.index.name = "Churn"
        st.dataframe(tenure_churn, use_container_width=True)

        st.markdown('<div class="section-title">Monthly Charges by Churn</div>',
                    unsafe_allow_html=True)
        mc_churn = df_raw.groupby("Churn")["MonthlyCharges"].describe()[["mean", "50%"]].round(2)
        mc_churn.columns = ["Mean ($)", "Median ($)"]
        mc_churn.index.name = "Churn"
        st.dataframe(mc_churn, use_container_width=True)

    # Churn rate chart
    st.markdown('<div class="section-title">Churn Rate by Tenure Band</div>',
                unsafe_allow_html=True)
    df_raw["tenure_band"] = pd.cut(df_raw["tenure"],
        bins=[0, 6, 12, 24, 36, 72],
        labels=["0–6 mo", "6–12 mo", "12–24 mo", "24–36 mo", "36–72 mo"])
    tenure_band_churn = df_raw.groupby("tenure_band", observed=True)["Churn"].apply(
        lambda x: round((x == "Yes").mean() * 100, 1)
    )
    st.bar_chart(tenure_band_churn, color="#00d9c0", height=220)

# ════════════════════════ TAB 2: FEATURE ANALYSIS ═════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Top 15 Churn Predictors</div>',
                unsafe_allow_html=True)

    top_feat = dict(list(feat_imp.items())[:15])
    feat_df  = pd.DataFrame({
        "Feature":    list(top_feat.keys()),
        "Importance": [round(v, 5) for v in top_feat.values()]
    }).sort_values("Importance")

    # Clean up feature names for display
    feat_df["Feature"] = (feat_df["Feature"]
        .str.replace("_", " ")
        .str.replace("Yes", "✓")
        .str.replace("No internet service", "(no internet)")
    )

    st.bar_chart(
        feat_df.set_index("Feature")["Importance"],
        color="#ff6b6b",
        height=420,
        horizontal=True,
    )

    st.markdown('<div class="section-title">Key Insights</div>', unsafe_allow_html=True)
    insights = [
        ("📅 Contract Type", "Month-to-month customers churn at ~3–4× the rate of two-year contracts. Nudging customers to longer contracts is the highest-ROI retention move."),
        ("🌐 Fiber Optic Internet", "Fiber customers have disproportionately high churn — likely driven by competition and higher prices. Satisfaction programs here matter."),
        ("💳 Electronic Check", "Customers paying by electronic check churn more. Auto-pay methods (bank transfer, credit card) correlate with lower churn, possibly due to friction in cancellation."),
        ("⏳ Tenure", "The first 6 months are the danger zone. Churn drops sharply after month 12. Onboarding programs targeting new customers have outsized impact."),
        ("🛡️ Online Security & Tech Support", "Customers with these add-ons churn significantly less. These services create 'stickiness' beyond just the core internet connection."),
    ]
    for title, body in insights:
        st.markdown(f"""
        <div class="insight-card">
            <strong>{title}</strong><br>{body}
        </div>""", unsafe_allow_html=True)

# ════════════════════════ TAB 3: DATA EXPLORER ════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Raw Dataset (7,043 customers)</div>',
                unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        churn_filter = st.multiselect("Filter by Churn", ["Yes", "No"], default=["Yes", "No"])
    with col_f2:
        contract_filter = st.multiselect(
            "Filter by Contract",
            df_raw["Contract"].unique().tolist(),
            default=df_raw["Contract"].unique().tolist()
        )
    with col_f3:
        internet_filter = st.multiselect(
            "Filter by Internet",
            df_raw["InternetService"].unique().tolist(),
            default=df_raw["InternetService"].unique().tolist()
        )

    filtered = df_raw[
        df_raw["Churn"].isin(churn_filter) &
        df_raw["Contract"].isin(contract_filter) &
        df_raw["InternetService"].isin(internet_filter)
    ]

    st.caption(f"Showing {len(filtered):,} of {len(df_raw):,} customers")
    st.dataframe(filtered, use_container_width=True, height=380)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "⬇ Download Filtered Data (CSV)",
            filtered.to_csv(index=False),
            "churn_filtered.csv",
            "text/csv",
            use_container_width=True,
        )

# ─── Prediction Panel (runs when button clicked) ──────────────────────────────
if predict_btn:
    X_input  = build_feature_vector()
    prob     = model.predict_proba(X_input)[0][1]
    pct_prob = round(prob * 100, 1)

    st.markdown("---")
    st.markdown("## ⚡ Churn Prediction Result")

    res_col, factors_col = st.columns([1, 1], gap="large")

    with res_col:
        if prob >= 0.5:
            st.markdown(f"""
            <div class="churn-high">
                <div class="prob-number" style="color:#ff4b4b">{pct_prob}%</div>
                <div style="font-size:1.2rem;font-weight:600;margin:8px 0;color:#ff6b6b">⚠️ HIGH CHURN RISK</div>
                <div style="color:#8892b0;font-size:0.88rem">This customer is likely to leave. Immediate retention action recommended.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="churn-low">
                <div class="prob-number" style="color:#00d9c0">{pct_prob}%</div>
                <div style="font-size:1.2rem;font-weight:600;margin:8px 0;color:#64ffda">✅ LOW CHURN RISK</div>
                <div style="color:#8892b0;font-size:0.88rem">This customer is likely to stay. Standard engagement is sufficient.</div>
            </div>""", unsafe_allow_html=True)

        # Risk level bar
        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(prob, text=f"Churn Probability: {pct_prob}%")

    with factors_col:
        st.markdown('<div class="section-title">Contributing Risk Factors</div>',
                    unsafe_allow_html=True)

        risk_factors = []
        if contract == "Month-to-month":
            risk_factors.append(("🔴 High", "Month-to-month contract (biggest churn driver)"))
        elif contract == "One year":
            risk_factors.append(("🟡 Med", "One-year contract (moderate risk)"))
        else:
            risk_factors.append(("🟢 Low", "Two-year contract (strong retention signal)"))

        if internet == "Fiber optic":
            risk_factors.append(("🔴 High", "Fiber optic internet (high-churn segment)"))

        if payment == "Electronic check":
            risk_factors.append(("🟡 Med", "Electronic check payment (correlates with higher churn)"))
        elif payment in ["Bank transfer (automatic)", "Credit card (automatic)"]:
            risk_factors.append(("🟢 Low", "Auto-pay method (retention signal)"))

        if tenure <= 6:
            risk_factors.append(("🔴 High", f"New customer ({tenure} months — danger zone)"))
        elif tenure <= 12:
            risk_factors.append(("🟡 Med", f"Early-stage customer ({tenure} months)"))
        else:
            risk_factors.append(("🟢 Low", f"Established customer ({tenure} months tenure)"))

        if online_sec == "No" and internet != "No":
            risk_factors.append(("🟡 Med", "No online security add-on"))
        if tech_sup == "No" and internet != "No":
            risk_factors.append(("🟡 Med", "No tech support add-on"))

        for level, desc in risk_factors:
            st.markdown(f"""
            <div class="insight-card">
                <strong>{level}</strong> — {desc}
            </div>""", unsafe_allow_html=True)

        # Retention suggestions
        st.markdown('<div class="section-title">Retention Suggestions</div>',
                    unsafe_allow_html=True)
        if prob >= 0.5:
            suggestions = []
            if contract == "Month-to-month":
                suggestions.append("🎁 Offer a discount to upgrade to a 1 or 2-year contract")
            if internet == "Fiber optic" and online_sec == "No":
                suggestions.append("🛡️ Offer free 3-month trial of Online Security")
            if payment == "Electronic check":
                suggestions.append("💳 Incentivize switching to auto-pay (e.g. $5/mo discount)")
            if tenure <= 12:
                suggestions.append("📞 Schedule proactive check-in call with customer success")
            if not suggestions:
                suggestions.append("📧 Send personalized loyalty offer via email")
                suggestions.append("⭐ Invite to loyalty rewards program")
            for s in suggestions:
                st.markdown(f'<div class="insight-card">{s}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="insight-card">✅ Customer appears satisfied. '
                'Standard monthly touchpoint is sufficient. '
                'Consider upselling add-on services.</div>',
                unsafe_allow_html=True
            )

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#4a5568;font-size:0.78rem'>"
    "Built with Streamlit · Gradient Boosting Classifier · Telco Churn Dataset"
    "</div>",
    unsafe_allow_html=True
)
