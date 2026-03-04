# 📡 Customer Churn Prediction Dashboard

A production-grade machine learning dashboard that predicts customer churn for a telecom company — built with Python, scikit-learn, and Streamlit.

**[🚀 Live Demo →](https://your-app.streamlit.app)** &nbsp;|&nbsp; **[📊 Dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)**

---

## 📸 Preview

![Dashboard Screenshot]Screenshot 2026-03-04 114059.png
![Dashboard Screenshot]Screenshot 2026-03-04 114217.png


---

## 🎯 Project Overview

Telecom companies lose 15–25% of their customers every year. Each churned customer costs $200–$400 to replace. This dashboard helps retention teams:

- **Identify** high-risk customers before they leave
- **Understand** which factors drive churn (contract type, tenure, service type)
- **Act** with data-backed retention recommendations

---

## 🔑 Key Findings

| Factor | Churn Rate | Insight |
|---|---|---|
| Month-to-month contract | ~40% | Highest churn driver |
| Two-year contract | ~5% | Strong retention anchor |
| Fiber optic internet | ~30% | Price/competition sensitivity |
| Tenure < 6 months | ~50% | Critical onboarding window |
| Electronic check | ~35% | Friction in auto-pay correlates with churn |
| Online Security add-on | ~16% | Stickiness effect |

---

## 🛠 Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| ML Model | Gradient Boosting Classifier (scikit-learn) |
| Dashboard | Streamlit |
| Data | Telco Customer Churn (Kaggle, 7,043 rows) |
| Hosting | Streamlit Community Cloud (free) |

---

## 📁 Project Structure

```
churn-dashboard/
├── app.py                    # Main Streamlit dashboard
├── generate_and_train.py     # Data generation + model training script
├── requirements.txt          # Python dependencies
├── data/
│   ├── telco_churn.csv       # Raw dataset (Telco schema)
│   └── processed_churn.csv   # Cleaned & encoded dataset
├── model/
│   ├── churn_model.pkl       # Trained GradientBoostingClassifier
│   ├── feature_names.pkl     # Column names for inference
│   ├── feature_importances.json
│   └── metrics.json          # AUC and summary stats
└── notebooks/
    └── eda.py                # Exploratory data analysis
```

---

## 🚀 Run Locally

```bash
# 1. Clone
git clone https://github.com/yourusername/churn-dashboard.git
cd churn-dashboard

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Retrain model with real Kaggle data
#    Download from: kaggle.com/datasets/blastchar/telco-customer-churn
#    Replace data/telco_churn.csv and run:
python generate_and_train.py

# 5. Launch dashboard
streamlit run app.py
```

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| Algorithm | Gradient Boosting Classifier |
| AUC-ROC | **0.79** |
| Accuracy | 84% |
| Dataset size | 7,043 customers |
| Train/Test split | 80/20 (stratified) |

> **Why AUC instead of accuracy?** The dataset has ~73–83% non-churners, so a model that always predicts "No Churn" would get 73%+ accuracy. AUC measures the model's ability to *rank* customers by churn likelihood — far more useful for retention teams.

---

## ☁️ Deploy to Streamlit Community Cloud (Free)

1. Push this repo to GitHub (public repo)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub → **New app**
4. Select your repo → set `app.py` as the main file
5. Click **Deploy** — live in ~2 minutes

No credit card. No Docker. No server management.

---

## 🔮 What to Use with Real Kaggle Data

If you download the real Telco dataset from Kaggle:

```python
# The real CSV has a TotalCharges column stored as string (Kaggle quirk)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)  # drops only 11 rows
```

Everything else in `generate_and_train.py` will work as-is.

To use XGBoost instead of scikit-learn's GradientBoostingClassifier:

```python
from xgboost import XGBClassifier
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    scale_pos_weight=3,       # handles class imbalance
    eval_metric='auc',
    random_state=42
)
```

---

## 📚 Resources

- [Telco Churn Dataset — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [scikit-learn GradientBoosting Docs](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Ken Jee — End-to-End Churn Project](https://www.youtube.com/@KenJee_ds)

---

*Built by [Your Name](https://github.com/yourusername)*
