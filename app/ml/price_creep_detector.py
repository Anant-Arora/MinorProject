import pandas as pd
import numpy as np
import re

# ===========================================================
# STEP 1: Load data and clean merchant names (same as before)
# ===========================================================
df = pd.read_csv("data/synthetic/transactions.csv")
df["date"] = pd.to_datetime(df["date"])

def clean_merchant(description):
    return re.sub(r'/\d{10,}/?$', '', description)

df["merchant_clean"] = df["description"].apply(clean_merchant)

# ===========================================================
# STEP 2: Re-identify subscriptions using the SAME timing-
# regularity logic as subscription_detector.py. Price-creep
# only makes sense to check WITHIN an already-confirmed
# subscription, not a random one-off merchant.
# ===========================================================
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
print(f"Confirmed subscriptions to check for price-creep: {len(subscription_groups)}")

# ===========================================================
# STEP 3: For each confirmed subscription, fit a straight
# line (linear regression) through its amounts over time.
# A rising slope = price creeping up. Measured as a
# PERCENTAGE of the starting price per period, since raw
# rupee slope isn't comparable across cheap vs expensive
# subscriptions.
# ===========================================================
PRICE_CREEP_THRESHOLD_PCT = 3.0  # flag if price rises >3% per charge, on average

results = []

for user_id, merchant in subscription_groups:
    group = df[(df["user_id"] == user_id) & (df["merchant_clean"] == merchant)].sort_values("date")
    amounts = group["amount"].values
    occurrence_index = np.arange(len(amounts))

    slope, intercept = np.polyfit(occurrence_index, amounts, 1)

    starting_price = amounts[0]
    slope_pct_per_period = (slope / starting_price) * 100 if starting_price > 0 else 0
    total_change_pct = ((amounts[-1] - amounts[0]) / amounts[0]) * 100 if amounts[0] > 0 else 0

    is_price_creep_predicted = slope_pct_per_period > PRICE_CREEP_THRESHOLD_PCT

    results.append({
        "user_id": user_id,
        "merchant_clean": merchant,
        "first_amount": round(amounts[0], 2),
        "last_amount": round(amounts[-1], 2),
        "total_change_pct": round(total_change_pct, 2),
        "slope_pct_per_period": round(slope_pct_per_period, 2),
        "predicted_price_creep": int(is_price_creep_predicted),
    })

results_df = pd.DataFrame(results)
print("\nPrice trend analysis for each confirmed subscription:")
print(results_df.to_string(index=False))

results_df.to_csv("data/synthetic/price_creep_results.csv", index=False)

# ===========================================================
# STEP 4: EVALUATE against ground truth (is_price_creep)
# ===========================================================
truth = df.groupby(["user_id", "merchant_clean"])["is_price_creep"].max().reset_index()
merged = results_df.merge(truth, on=["user_id", "merchant_clean"])

tp = ((merged["predicted_price_creep"] == 1) & (merged["is_price_creep"] == 1)).sum()
fp = ((merged["predicted_price_creep"] == 1) & (merged["is_price_creep"] == 0)).sum()
fn = ((merged["predicted_price_creep"] == 0) & (merged["is_price_creep"] == 1)).sum()
tn = ((merged["predicted_price_creep"] == 0) & (merged["is_price_creep"] == 0)).sum()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print("\n" + "=" * 50)
print("EVALUATION AGAINST GROUND TRUTH")
print("=" * 50)
print(f"True Positives: {tp} | False Positives: {fp} | False Negatives: {fn} | True Negatives: {tn}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")