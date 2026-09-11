import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("data/synthetic/transactions.csv")
df["date"] = pd.to_datetime(df["date"])

def clean_merchant(description):
    return re.sub(r'/\d{10,}/?$', '', description)

df["merchant_clean"] = df["description"].apply(clean_merchant)

def detect_subscriptions(df, min_occurrences=2, expected_interval=30, tolerance=5):
    subscription_groups = []
    grouped = df.sort_values("date").groupby(["user_id", "merchant_clean"])
    for (user_id, merchant_clean), group in grouped:
        if len(group) < min_occurrences:
            continue
        dates = group["date"].sort_values()
        gaps_in_days = dates.diff().dropna().dt.days
        if len(gaps_in_days) == 0:
            continue
        is_consistent = gaps_in_days.apply(lambda d: abs(d - expected_interval) <= tolerance)
        if is_consistent.all():
            subscription_groups.append((user_id, merchant_clean))
    return subscription_groups

subscription_groups = detect_subscriptions(df)
print(f"Confirmed subscriptions found: {len(subscription_groups)}")

def build_price_creep_features(df, subscription_groups):
    rows = []
    for user_id, merchant in subscription_groups:
        group = df[(df["user_id"] == user_id) & (df["merchant_clean"] == merchant)].sort_values("date")
        amounts = group["amount"].values
        occurrence_index = np.arange(len(amounts))

        slope, intercept = np.polyfit(occurrence_index, amounts, 1)
        starting_price = amounts[0]
        slope_pct_per_period = (slope / starting_price) * 100 if starting_price > 0 else 0
        total_change_pct = ((amounts[-1] - amounts[0]) / amounts[0]) * 100 if amounts[0] > 0 else 0
        amount_std = amounts.std() if len(amounts) > 1 else 0
        amount_cv = (amount_std / amounts.mean()) if amounts.mean() > 0 else 0

        rows.append({
            "user_id": user_id, "merchant_clean": merchant,
            "slope_pct_per_period": round(slope_pct_per_period, 4),
            "total_change_pct": round(total_change_pct, 4),
            "amount_cv": round(amount_cv, 4),
            "mean_amount": round(amounts.mean(), 2),
            "label": int(group["is_price_creep"].max()),
        })
    return pd.DataFrame(rows)

features_df = build_price_creep_features(df, subscription_groups)

print(f"Positive (price-creep) examples: {features_df['label'].sum()}")
print(f"Negative (stable-price) examples: {(features_df['label']==0).sum()}")
print()

TRAIN_USERS = ["user_1", "user_2", "user_3"]
TEST_USERS = ["user_4", "user_5"]

train_df = features_df[features_df["user_id"].isin(TRAIN_USERS)]
test_df = features_df[features_df["user_id"].isin(TEST_USERS)]

FEATURE_COLUMNS = ["slope_pct_per_period", "total_change_pct", "amount_cv", "mean_amount"]

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df["label"]
X_test = test_df[FEATURE_COLUMNS]
y_test = test_df["label"]

print(f"Training on {len(X_train)} subscriptions (users 1-3) — {y_train.sum()} price-creep, {(y_train==0).sum()} stable")
print(f"Testing on {len(X_test)} subscriptions (users 4-5, NEVER seen during training) — {y_test.sum()} price-creep, {(y_test==0).sum()} stable")
print()

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("=" * 55)
print("EVALUATION ON UNSEEN TEST USERS (user_4, user_5)")
print("=" * 55)
print(classification_report(y_test, y_pred, target_names=["Stable Price", "Price Creep"], zero_division=0))

print("Confusion Matrix:")
print("             Predicted:0  Predicted:1")
cm = confusion_matrix(y_test, y_pred)
print(f"Actual:0        {cm[0][0]:>5}        {cm[0][1]:>5}")
print(f"Actual:1        {cm[1][0]:>5}        {cm[1][1]:>5}")

print("\nFeature importance (what the model learned mattered most):")
importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
print(importance)