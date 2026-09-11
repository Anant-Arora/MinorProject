import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# ===========================================================
# STEP 1: Load data
# ===========================================================
df = pd.read_csv("data/synthetic/transactions.csv")
df["date"] = pd.to_datetime(df["date"])

# ===========================================================
# STEP 2: FEATURE ENGINEERING
# A flat amount threshold isn't fair across users with very
# different typical spending levels. So instead of raw amount
# alone, we compute how unusual a transaction is RELATIVE TO
# THAT USER'S OWN typical spending (a z-score): how many
# standard deviations away from their own average is this
# transaction? Standard practice in fraud/anomaly detection —
# uses only each user's own amounts, no label leakage.
# ===========================================================
def add_user_amount_zscore(df):
    df = df.copy()
    user_mean = df.groupby("user_id")["amount"].transform("mean")
    user_std = df.groupby("user_id")["amount"].transform("std")
    user_std = user_std.replace(0, 1)  # avoid divide-by-zero edge case
    df["amount_zscore"] = (df["amount"] - user_mean) / user_std
    return df

df = add_user_amount_zscore(df)

FEATURE_COLUMNS = ["amount", "amount_zscore"]

# ===========================================================
# STEP 3: TRAIN / TEST SPLIT BY USER
# ===========================================================
TRAIN_USERS = ["user_1", "user_2", "user_3"]
TEST_USERS = ["user_4", "user_5"]

train_df = df[df["user_id"].isin(TRAIN_USERS)]
test_df = df[df["user_id"].isin(TEST_USERS)]

X_train = train_df[FEATURE_COLUMNS]
X_test = test_df[FEATURE_COLUMNS]
y_test = test_df["is_anomaly"]  # only used for EVALUATION, never given to the model

print(f"Training on {len(X_train)} transactions (users 1-3)")
print(f"Testing on {len(X_test)} transactions (users 4-5, NEVER seen during training)")

# ===========================================================
# STEP 4: TRAIN Isolation Forest (UNSUPERVISED — it never
# sees is_anomaly labels, only the feature values)
# ===========================================================
train_anomaly_rate = train_df["is_anomaly"].mean()
contamination = max(train_anomaly_rate, 0.01)

print(f"Estimated contamination (expected anomaly rate): {contamination:.4f}")

model = IsolationForest(
    n_estimators=100,
    contamination=contamination,
    random_state=42,
)
model.fit(X_train)

# ===========================================================
# STEP 5: PREDICT on held-out test users
# Isolation Forest outputs -1 for anomaly, 1 for normal —
# we convert to 1 = anomaly, 0 = normal.
# ===========================================================
raw_predictions = model.predict(X_test)
y_pred = np.where(raw_predictions == -1, 1, 0)

# ===========================================================
# STEP 6: EVALUATE against ground truth
# ===========================================================
print("\n" + "=" * 55)
print("EVALUATION ON UNSEEN TEST USERS (user_4, user_5)")
print("=" * 55)
print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"], zero_division=0))

print("Confusion Matrix:")
print("             Predicted:0  Predicted:1")
cm = confusion_matrix(y_test, y_pred)
print(f"Actual:0        {cm[0][0]:>5}        {cm[0][1]:>5}")
print(f"Actual:1        {cm[1][0]:>5}        {cm[1][1]:>5}")

# ===========================================================
# STEP 7: Show which specific transactions were flagged
# ===========================================================
test_df = test_df.copy()
test_df["predicted_anomaly"] = y_pred

flagged = test_df[test_df["predicted_anomaly"] == 1][
    ["user_id", "date", "description", "amount", "amount_zscore", "is_anomaly"]
]
print("\nTransactions flagged as anomalies:")
print(flagged.to_string(index=False))

test_df.to_csv("data/synthetic/anomaly_results.csv", index=False)
print("\nSaved to: data/synthetic/anomaly_results.csv")