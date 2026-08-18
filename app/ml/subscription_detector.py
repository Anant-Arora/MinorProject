import pandas as pd
import re

# ---------------------------------------------------------
# Load the dataset
# ---------------------------------------------------------
df = pd.read_csv("data/synthetic/transactions.csv")
df["date"] = pd.to_datetime(df["date"])

# ---------------------------------------------------------
# Clean merchant descriptions (remove reference numbers)
# ---------------------------------------------------------
def clean_merchant(description):
    cleaned = re.sub(r'/\d{10,}/?$', '', description)
    return cleaned

df["merchant_clean"] = df["description"].apply(clean_merchant)

# ---------------------------------------------------------
# Detect subscriptions using TIMING regularity, not just
# repetition. A real subscription charges roughly every
# 30 days, consistently. Random frequent spending (like
# ordering Swiggy often) does NOT have consistent gaps.
# ---------------------------------------------------------
def detect_subscriptions(df, min_occurrences=2, expected_interval=30, tolerance=5):
    predictions = []
    grouped = df.sort_values("date").groupby(["user_id", "merchant_clean"])

    for (user_id, merchant_clean), group in grouped:
        if len(group) < min_occurrences:
            continue

        dates = group["date"].sort_values()
        gaps_in_days = dates.diff().dropna().dt.days

        if len(gaps_in_days) == 0:
            continue

        # Every gap must be close to ~30 days (within tolerance)
        is_consistent = gaps_in_days.apply(
            lambda d: abs(d - expected_interval) <= tolerance
        )

        if is_consistent.all():
            predictions.append({
                "user_id": user_id,
                "merchant_clean": merchant_clean,
                "occurrences": len(group),
                "avg_interval_days": round(gaps_in_days.mean(), 1),
                "min_amount": group["amount"].min(),
                "max_amount": group["amount"].max(),
                "predicted_subscription": 1
            })

    return pd.DataFrame(predictions)

# ---------------------------------------------------------
# Run and display
# ---------------------------------------------------------
detected = detect_subscriptions(df)

print(f"Detected {len(detected)} likely subscriptions out of {df.shape[0]} total transactions")
print(detected)