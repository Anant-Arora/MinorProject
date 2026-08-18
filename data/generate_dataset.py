import pandas as pd
import random
from datetime import date, timedelta

# ---------------------------------------------------------
# CONFIG: adjust these numbers to control dataset size
# ---------------------------------------------------------
NUM_USERS = 5
MONTHS = 6
START_DATE = date(2026, 2, 1)

# ---------------------------------------------------------
# Merchant templates, styled after real UPI statement formatting
# (based on the pattern from your actual bank statement)
# ---------------------------------------------------------
SUBSCRIPTION_MERCHANTS = [
    "UPI/NETFLIX.COM/UPIIntent/HDFC",
    "UPI/SPOTIFY INDIA/UPIIntent/ICICI",
    "UPI/ANGEL ONE/angelmfcpupa@i/Subscripti/INDUSIND",
    "UPI/AMAZON PRIME/UPIIntent/AXIS",
]

# These merchants will have a subscription that quietly gets pricier over time
PRICE_CREEP_MERCHANTS = [
    "UPI/HOTSTAR SUB/UPIIntent/SBI",
]

NORMAL_MERCHANTS = [
    "UPI/SWIGGY*BLR123/UPIIntent/HDFC",
    "UPI/Blinkit/blinkit.payu@h/UPIIntent/HDFC",
    "UPI/Zomato Order/UPIIntent/ICICI",
    "UPI/Thesmartq1/thesmartq1.pay/UPIIntent/AIRTEL",
    "UPI/BOTTLE LAB TECHNOLOGIES/pinelabs.11443/UPI/AXIS",
    "UPI/Kamod Kumar/UPI/INDIAN",
    "UPI/THE FROTH HOUSE/cf.thefrothhou/NSDL",
    "UPI/CATERING CARE FSS/cateringcare1/HDFC",
    "UPI/M S ISHAN SHARMA/eazypay.8kvdv4/ICICI",
]

def random_normal_amount():
    return round(random.uniform(20, 1200), 2)

def random_anomaly_amount():
    # Deliberately unusual, large one-off amounts
    return round(random.uniform(15000, 60000), 2)

def random_ref():
    return str(random.randint(100000000000, 999999999999))

# ---------------------------------------------------------
# Generate transactions for one user
# ---------------------------------------------------------
def generate_user_transactions(user_id):
    rows = []

    # 1. Subscriptions: same merchant, same amount, once a month
    for merchant in SUBSCRIPTION_MERCHANTS:
        amount = random_normal_amount()
        for m in range(MONTHS):
            txn_date = START_DATE + timedelta(days=30 * m + random.randint(0, 2))
            rows.append({
                "user_id": user_id,
                "date": txn_date,
                "description": f"{merchant}/{random_ref()}/",
                "amount": amount,
                "is_subscription": 1,
                "is_price_creep": 0,
                "is_anomaly": 0,
            })

    # 2. Price-creep: same merchant, amount increases every 2 months
    for merchant in PRICE_CREEP_MERCHANTS:
        base_amount = random_normal_amount()
        for m in range(MONTHS):
            amount = round(base_amount * (1 + 0.08 * (m // 2)), 2)  # +8% every 2 months
            txn_date = START_DATE + timedelta(days=30 * m + random.randint(0, 2))
            rows.append({
                "user_id": user_id,
                "date": txn_date,
                "description": f"{merchant}/{random_ref()}/",
                "amount": amount,
                "is_subscription": 1,
                "is_price_creep": 1,
                "is_anomaly": 0,
            })

    # 3. Normal one-off transactions, several per month
    for m in range(MONTHS):
        for _ in range(random.randint(15, 25)):
            merchant = random.choice(NORMAL_MERCHANTS)
            txn_date = START_DATE + timedelta(days=30 * m + random.randint(0, 29))
            rows.append({
                "user_id": user_id,
                "date": txn_date,
                "description": f"{merchant}/{random_ref()}/",
                "amount": random_normal_amount(),
                "is_subscription": 0,
                "is_price_creep": 0,
                "is_anomaly": 0,
            })

    # 4. A couple of deliberate anomalies
    for _ in range(random.randint(1, 3)):
        merchant = random.choice(NORMAL_MERCHANTS)
        txn_date = START_DATE + timedelta(days=random.randint(0, 30 * MONTHS - 1))
        rows.append({
            "user_id": user_id,
            "date": txn_date,
            "description": f"{merchant}/{random_ref()}/",
            "amount": random_anomaly_amount(),
            "is_subscription": 0,
            "is_price_creep": 0,
            "is_anomaly": 1,
        })

    return rows

# ---------------------------------------------------------
# Generate for all users and save
# ---------------------------------------------------------
all_rows = []
for user_num in range(1, NUM_USERS + 1):
    all_rows.extend(generate_user_transactions(f"user_{user_num}"))

df = pd.DataFrame(all_rows)
df = df.sort_values(["user_id", "date"]).reset_index(drop=True)

output_path = "data/synthetic/transactions.csv"
df.to_csv(output_path, index=False)

print(f"Generated {len(df)} transactions for {NUM_USERS} users.")
print(f"Saved to: {output_path}")
print(df.head(10))