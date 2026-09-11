import pandas as pd
import random
from datetime import date, timedelta

NUM_USERS = 5
MONTHS = 6
START_DATE = date(2026, 2, 1)

STABLE_SUBSCRIPTION_MERCHANTS = [
    "UPI/NETFLIX.COM/UPIIntent/HDFC",
    "UPI/SPOTIFY INDIA/UPIIntent/ICICI",
    "UPI/ANGEL ONE/angelmfcpupa@i/Subscripti/INDUSIND",
    "UPI/AMAZON PRIME/UPIIntent/AXIS",
]

# (merchant_string, rate_per_step, interval_months) — 5 different
# creep patterns so the model sees VARIETY, not one fixed shape
PRICE_CREEP_MERCHANTS = [
    ("UPI/HOTSTAR SUB/UPIIntent/SBI", 0.08, 2),
    ("UPI/GYMFIT CLUB/UPIIntent/HDFC", 0.05, 3),
    ("UPI/CLOUDSTORAGE PRO/UPIIntent/ICICI", 0.10, 2),
    ("UPI/MUSICSTREAM PLUS/UPIIntent/AXIS", 0.03, 1),
    ("UPI/NEWSDIGITAL SUB/UPIIntent/SBI", 0.12, 3),
]

MERCHANT_VARIANT_GROUPS = {
    "Swiggy": ["UPI/SWIGGY*BLR123/UPIIntent/HDFC", "UPI/SWIGGY BANGALORE/UPIIntent/HDFC", "UPI/Bundl Technologies Pvt Ltd/UPIIntent/HDFC"],
    "Zomato": ["UPI/Zomato Order/UPIIntent/ICICI", "UPI/ZOMATO LTD/UPIIntent/ICICI"],
    "Blinkit": ["UPI/Blinkit/blinkit.payu@h/UPIIntent/HDFC", "UPI/BLINK COMMERCE PVT LTD/UPIIntent/HDFC"],
    "Thesmartq1": ["UPI/Thesmartq1/thesmartq1.pay/UPIIntent/AIRTEL"],
    "BottleLab": ["UPI/BOTTLE LAB TECHNOLOGIES/pinelabs.11443/UPI/AXIS"],
    "KamodKumar": ["UPI/Kamod Kumar/UPI/INDIAN"],
    "TheFrothHouse": ["UPI/THE FROTH HOUSE/cf.thefrothhou/NSDL"],
    "CateringCare": ["UPI/CATERING CARE FSS/cateringcare1/HDFC"],
    "IshanSharma": ["UPI/M S ISHAN SHARMA/eazypay.8kvdv4/ICICI"],
}

def random_normal_amount():
    return round(random.uniform(20, 1200), 2)

def random_anomaly_amount():
    return round(random.uniform(15000, 60000), 2)

def random_ref():
    return str(random.randint(100000000000, 999999999999))

def generate_user_transactions(user_id):
    rows = []

    for merchant in STABLE_SUBSCRIPTION_MERCHANTS:
        amount = random_normal_amount()
        for m in range(MONTHS):
            txn_date = START_DATE + timedelta(days=30 * m + random.randint(0, 2))
            rows.append({
                "user_id": user_id, "date": txn_date, "description": f"{merchant}/{random_ref()}/",
                "amount": amount, "is_subscription": 1, "is_price_creep": 0, "is_anomaly": 0,
                "true_canonical_merchant": None,
            })

    for merchant, rate, interval_months in PRICE_CREEP_MERCHANTS:
        base_amount = random_normal_amount()
        for m in range(MONTHS):
            amount = round(base_amount * (1 + rate * (m // interval_months)), 2)
            txn_date = START_DATE + timedelta(days=30 * m + random.randint(0, 2))
            rows.append({
                "user_id": user_id, "date": txn_date, "description": f"{merchant}/{random_ref()}/",
                "amount": amount, "is_subscription": 1, "is_price_creep": 1, "is_anomaly": 0,
                "true_canonical_merchant": None,
            })

    canonical_names = list(MERCHANT_VARIANT_GROUPS.keys())
    for m in range(MONTHS):
        for _ in range(random.randint(15, 25)):
            canonical = random.choice(canonical_names)
            merchant = random.choice(MERCHANT_VARIANT_GROUPS[canonical])
            txn_date = START_DATE + timedelta(days=30 * m + random.randint(0, 29))
            rows.append({
                "user_id": user_id, "date": txn_date, "description": f"{merchant}/{random_ref()}/",
                "amount": random_normal_amount(), "is_subscription": 0, "is_price_creep": 0, "is_anomaly": 0,
                "true_canonical_merchant": canonical,
            })

    for _ in range(random.randint(1, 3)):
        canonical = random.choice(canonical_names)
        merchant = random.choice(MERCHANT_VARIANT_GROUPS[canonical])
        txn_date = START_DATE + timedelta(days=random.randint(0, 30 * MONTHS - 1))
        rows.append({
            "user_id": user_id, "date": txn_date, "description": f"{merchant}/{random_ref()}/",
            "amount": random_anomaly_amount(), "is_subscription": 0, "is_price_creep": 0, "is_anomaly": 1,
            "true_canonical_merchant": canonical,
        })

    return rows

all_rows = []
for user_num in range(1, NUM_USERS + 1):
    all_rows.extend(generate_user_transactions(f"user_{user_num}"))

df = pd.DataFrame(all_rows)
df = df.sort_values(["user_id", "date"]).reset_index(drop=True)

output_path = "data/synthetic/transactions.csv"
df.to_csv(output_path, index=False)

print(f"Generated {len(df)} transactions for {NUM_USERS} users.")
print(f"Saved to: {output_path}")