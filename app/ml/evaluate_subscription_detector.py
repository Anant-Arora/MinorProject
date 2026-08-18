import pandas as pd
import re

df = pd.read_csv("data/synthetic/transactions.csv")
df["date"] = pd.to_datetime(df["date"])

def clean_merchant(description):
    cleaned = re.sub(r'/\d{10,}/?$', '', description)
    return cleaned

df["merchant_clean"] = df["description"].apply(clean_merchant)

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

        is_consistent = gaps_in_days.apply(lambda d: abs(d - expected_interval) <= tolerance)

        if is_consistent.all():
            predictions.append({
                "user_id": user_id,
                "merchant_clean": merchant_clean,
            })

    return pd.DataFrame(predictions)

detected = detect_subscriptions(df)
predicted_set = set(zip(detected["user_id"], detected["merchant_clean"]))

actual_subscriptions = df[df["is_subscription"] == 1].groupby(
    ["user_id", "merchant_clean"]
).size().reset_index(name="count")
actual_set = set(zip(actual_subscriptions["user_id"], actual_subscriptions["merchant_clean"]))

print(f"ACTUAL subscriptions (ground truth): {len(actual_set)} unique (user, merchant) pairs")
print(f"PREDICTED subscriptions: {len(predicted_set)} unique (user, merchant) pairs")

true_positives = actual_set & predicted_set
false_negatives = actual_set - predicted_set
false_positives = predicted_set - actual_set

print(f"\nTrue Positives: {len(true_positives)}")
print(f"False Negatives: {len(false_negatives)}")
print(f"False Positives: {len(false_positives)}")

if false_negatives:
    print("\nMissed subscriptions:")
    for item in false_negatives:
        print(f"  {item}")

if false_positives:
    print("\nWrongly flagged:")
    for item in false_positives:
        print(f"  {item}")

recall = len(true_positives) / len(actual_set) if actual_set else 0
precision = len(true_positives) / len(predicted_set) if predicted_set else 0

print(f"\nRecall: {recall:.2%}")
print(f"Precision: {precision:.2%}")