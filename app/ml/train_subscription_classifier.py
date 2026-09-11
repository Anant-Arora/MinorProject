import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ===========================================================
# STEP 1: Load data and build FEATURES per (user, merchant)
# group. Instead of a hand-written rule, we describe each
# group with numbers, and let a model learn the pattern.
# ===========================================================
df = pd.read_csv("data/synthetic/transactions.csv")
df["date"] = pd.to_datetime(df["date"])

def clean_merchant(description):
    return re.sub(r'/\d{10,}/?$', '', description)

df["merchant_clean"] = df["description"].apply(clean_merchant)

def build_features(df):
    rows = []
    grouped = df.sort_values("date").groupby(["user_id", "merchant_clean"])

    for (user_id, merchant), group in grouped:
        amounts = group["amount"]
        dates = group["date"].sort_values()
        gaps = dates.diff().dropna().dt.days

        mean_gap = gaps.mean() if len(gaps) > 0 else 0
        std_gap = gaps.std() if len(gaps) > 1 else 0

        mean_amount = amounts.mean()
        std_amount = amounts.std() if len(amounts) > 1 else 0
        amount_cv = (std_amount / mean_amount) if mean_amount > 0 else 0

        rows.append({
            "user_id": user_id,
            "merchant_clean": merchant,
            "occurrences": len(group),
            "mean_gap_days": round(mean_gap, 2),
            "std_gap_days": round(std_gap, 2),
            "mean_amount": round(mean_amount, 2),
            "amount_cv": round(amount_cv, 4),
            "min_amount": amounts.min(),
            "max_amount": amounts.max(),
            "label": int(group["is_subscription"].max()),
        })

    return pd.DataFrame(rows)

features_df = build_features(df)

print(f"Total (user, merchant) groups: {len(features_df)}")
print(f"Subscriptions: {features_df['label'].sum()} | Non-subscriptions: {(features_df['label']==0).sum()}")
print()

# ===========================================================
# STEP 2: TRAIN / TEST SPLIT BY USER (not random rows!)
# Train on users 1-3, test on users 4-5, so the model is
# proven to generalize to people it never saw.
# ===========================================================
TRAIN_USERS = ["user_1", "user_2", "user_3"]
TEST_USERS = ["user_4", "user_5"]

train_df = features_df[features_df["user_id"].isin(TRAIN_USERS)]
test_df = features_df[features_df["user_id"].isin(TEST_USERS)]

FEATURE_COLUMNS = ["occurrences", "mean_gap_days", "std_gap_days", "mean_amount", "amount_cv"]

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df["label"]
X_test = test_df[FEATURE_COLUMNS]
y_test = test_df["label"]

print(f"Training on {len(X_train)} groups (users 1-3)")
print(f"Testing on {len(X_test)} groups (users 4-5, NEVER seen during training)")
print()

# ===========================================================
# STEP 3: TRAIN the model
# ===========================================================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ===========================================================
# STEP 4: EVALUATE on held-out test users
# ===========================================================
y_pred = model.predict(X_test)

print("=" * 55)
print("EVALUATION ON UNSEEN TEST USERS (user_4, user_5)")
print("=" * 55)
print(classification_report(y_test, y_pred, target_names=["Not Subscription", "Subscription"]))

print("Confusion Matrix:")
print("             Predicted:0  Predicted:1")
cm = confusion_matrix(y_test, y_pred)
print(f"Actual:0        {cm[0][0]:>5}        {cm[0][1]:>5}")
print(f"Actual:1        {cm[1][0]:>5}        {cm[1][1]:>5}")

# ===========================================================
# STEP 5: Feature importance
# ===========================================================
print("\nFeature importance (what the model learned mattered most):")
importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
print(importance)