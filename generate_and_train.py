"""
Generate realistic Telco-like churn dataset and train model.
Matches the schema of WA_Fn-UseC_-Telco-Customer-Churn.csv from Kaggle.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import json, os

np.random.seed(42)
N = 7043

# ── Demographic features ──────────────────────────────────────────────────────
gender        = np.random.choice(['Male', 'Female'], N)
senior        = np.random.choice([0, 1], N, p=[0.84, 0.16])
partner       = np.random.choice(['Yes', 'No'], N, p=[0.48, 0.52])
dependents    = np.random.choice(['Yes', 'No'], N, p=[0.30, 0.70])
tenure        = np.random.choice(range(0, 73), N)
tenure        = np.where(tenure == 0, 1, tenure)  # min 1 month

# ── Service features ──────────────────────────────────────────────────────────
phone_service = np.random.choice(['Yes', 'No'], N, p=[0.90, 0.10])

# Multiple lines only meaningful if phone service
multiple_lines = np.where(
    phone_service == 'No', 'No phone service',
    np.random.choice(['Yes', 'No'], N, p=[0.42, 0.58])
)

internet_service = np.random.choice(
    ['DSL', 'Fiber optic', 'No'], N, p=[0.34, 0.44, 0.22]
)

def internet_addon(internet, yes_p=0.45):
    return np.where(
        internet == 'No', 'No internet service',
        np.random.choice(['Yes', 'No'], N, p=[yes_p, 1-yes_p])
    )

online_security   = internet_addon(internet_service, 0.29)
online_backup     = internet_addon(internet_service, 0.34)
device_protection = internet_addon(internet_service, 0.34)
tech_support      = internet_addon(internet_service, 0.29)
streaming_tv      = internet_addon(internet_service, 0.38)
streaming_movies  = internet_addon(internet_service, 0.39)

# ── Account features ──────────────────────────────────────────────────────────
# Short-tenure customers more likely to be month-to-month
def get_contract_p(t):
    if t < 12:   return [0.75, 0.15, 0.10]
    elif t < 36: return [0.50, 0.30, 0.20]
    else:        return [0.25, 0.35, 0.40]

contract = np.array([
    np.random.choice(['Month-to-month', 'One year', 'Two year'], p=get_contract_p(t))
    for t in tenure
])

paperless_billing = np.random.choice(['Yes', 'No'], N, p=[0.59, 0.41])

payment_method = np.random.choice(
    ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
    N, p=[0.34, 0.23, 0.22, 0.21]
)

# Monthly charges: fiber > DSL > No internet; vary by addons
base_charge = np.where(internet_service == 'Fiber optic', 70,
              np.where(internet_service == 'DSL', 45, 20))
addon_count = (
    (online_security == 'Yes').astype(int) +
    (online_backup == 'Yes').astype(int) +
    (device_protection == 'Yes').astype(int) +
    (tech_support == 'Yes').astype(int) +
    (streaming_tv == 'Yes').astype(int) +
    (streaming_movies == 'Yes').astype(int) +
    (multiple_lines == 'Yes').astype(int)
)
monthly_charges = base_charge + addon_count * 5 + np.random.normal(0, 3, N)
monthly_charges = np.round(np.clip(monthly_charges, 18, 120), 2)

total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 10, N), 2)
total_charges = np.clip(total_charges, 18, 9000)

# ── Churn label (realistic business logic) ───────────────────────────────────
churn_score = (
    - 0.04 * tenure
    + 0.8  * (contract == 'Month-to-month').astype(int)
    - 0.5  * (contract == 'Two year').astype(int)
    + 0.5  * (internet_service == 'Fiber optic').astype(int)
    + 0.3  * (payment_method == 'Electronic check').astype(int)
    - 0.3  * (online_security == 'Yes').astype(int)
    - 0.3  * (tech_support == 'Yes').astype(int)
    + 0.2  * senior.astype(int)
    - 0.2  * (partner == 'Yes').astype(int)
    - 0.1  * (dependents == 'Yes').astype(int)
    + 0.01 * monthly_charges
    + np.random.normal(0, 0.3, N)
    - 1.5   # baseline offset → ~26% churn
)
churn_prob = 1 / (1 + np.exp(-churn_score))
churn = (np.random.random(N) < churn_prob).astype(int)

# ── Build DataFrame ───────────────────────────────────────────────────────────
df = pd.DataFrame({
    'customerID':        [f'CUST-{i:04d}' for i in range(N)],
    'gender':            gender,
    'SeniorCitizen':     senior,
    'Partner':           partner,
    'Dependents':        dependents,
    'tenure':            tenure,
    'PhoneService':      phone_service,
    'MultipleLines':     multiple_lines,
    'InternetService':   internet_service,
    'OnlineSecurity':    online_security,
    'OnlineBackup':      online_backup,
    'DeviceProtection':  device_protection,
    'TechSupport':       tech_support,
    'StreamingTV':       streaming_tv,
    'StreamingMovies':   streaming_movies,
    'Contract':          contract,
    'PaperlessBilling':  paperless_billing,
    'PaymentMethod':     payment_method,
    'MonthlyCharges':    monthly_charges,
    'TotalCharges':      total_charges,
    'Churn':             np.where(churn == 1, 'Yes', 'No'),
})

os.makedirs('data', exist_ok=True)
os.makedirs('model', exist_ok=True)

df.to_csv('data/telco_churn.csv', index=False)
print(f"Dataset saved: {df.shape}")
print(df['Churn'].value_counts())

# ── Preprocessing ─────────────────────────────────────────────────────────────
df2 = df.drop('customerID', axis=1).copy()

binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
for col in ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
    df2[col] = df2[col].map(binary_map)

df2['Churn'] = df2['Churn'].map({'Yes': 1, 'No': 0})

df2 = pd.get_dummies(df2, columns=[
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaymentMethod'
], drop_first=True)

df2.to_csv('data/processed_churn.csv', index=False)
print(f"\nProcessed dataset: {df2.shape}")

# ── Train/Test split ──────────────────────────────────────────────────────────
X = df2.drop('Churn', axis=1)
y = df2['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ── Train GradientBoostingClassifier (XGBoost equivalent, built-in sklearn) ──
model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred       = model.predict(X_test)

auc = roc_auc_score(y_test, y_pred_proba)
print(f"\n✅ Model AUC: {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# ── Save model and metadata ───────────────────────────────────────────────────
joblib.dump(model, 'model/churn_model.pkl')
joblib.dump(list(X.columns), 'model/feature_names.pkl')

# Feature importances
feat_imp = dict(zip(X.columns, model.feature_importances_))
feat_imp_sorted = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))
with open('model/feature_importances.json', 'w') as f:
    json.dump(feat_imp_sorted, f, indent=2)

# Save model metrics
metrics = {
    'auc': round(auc, 4),
    'total_customers': int(len(df)),
    'churn_rate': round(float(df['Churn'].eq('Yes').mean()), 4),
    'avg_monthly_charges': round(float(df['MonthlyCharges'].mean()), 2),
    'avg_tenure': round(float(df['tenure'].mean()), 1),
    'test_size': len(X_test),
}
with open('model/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n✅ Model saved to model/churn_model.pkl")
print(f"✅ Metrics: {metrics}")
