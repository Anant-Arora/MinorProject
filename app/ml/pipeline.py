import os
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

def run_finsight_pipeline(csv_path: str = "data/synthetic/transactions.csv"):
    if not os.path.exists(csv_path):
        csv_path = "../../data/synthetic/transactions.csv"
        
    print("=" * 60)
    print("FINSIGHT ML MASTER PIPELINE EXECUTION")
    print("=" * 60)
    
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["user_id", "date"]).reset_index(drop=True)

    # ===========================================================
    # STEP 1: Merchant Normalization Pass
    # ===========================================================
    print("\n[1/4] Running Merchant Normalization Engine...")
    def clean_merchant(description):
        # Strip UPI/bank boilerplate tokens
        cleaned = re.sub(r'/\d{10,}/?$', '', description)
        return cleaned.strip()

    df["merchant_clean"] = df["description"].apply(clean_merchant)
    # Using your established ground truth canonical mapping if available, or clean string
    print(f" -> Processed {len(df)} transactions through merchant normalization.")

    # ===========================================================
    # STEP 2: Subscription Classification Engine (Train/Test Split)
    # ===========================================================
    print("\n[2/4] Running Subscription Classification Engine...")
    sub_features = []
    
    for (user_id, merchant), group in df.groupby(["user_id", "merchant_clean"]):
        dates = group["date"].sort_values()
        gaps = dates.diff().dropna().dt.days
        
        occurrences = len(group)
        mean_gap_days = gaps.mean() if len(gaps) > 0 else 0
        std_gap_days = gaps.std() if len(gaps) > 1 else 0
        mean_amount = group["amount"].mean()
        amount_cv = (group["amount"].std() / mean_amount) if mean_amount > 0 else 0
        
        is_sub = group["is_subscription"].max()
        
        sub_features.append({
            "user_id": user_id,
            "merchant_clean": merchant,
            "occurrences": occurrences,
            "mean_gap_days": mean_gap_days,
            "std_gap_days": std_gap_days,
            "mean_amount": mean_amount,
            "amount_cv": amount_cv,
            "is_subscription": is_sub
        })
        
    sub_df = pd.DataFrame(sub_features)
    train_users = ["user_1", "user_2", "user_3"]
    
    X_sub_cols = ["occurrences", "mean_gap_days", "std_gap_days", "amount_cv"]
    sub_train = sub_df[sub_df["user_id"].isin(train_users)]
    sub_test = sub_df[~sub_df["user_id"].isin(train_users)]
    
    sub_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    sub_clf.fit(sub_train[X_sub_cols], sub_train["is_subscription"])
    sub_df["predicted_subscription"] = sub_clf.predict(sub_df[X_sub_cols])
    print(" -> Subscription Classifier trained & predicted successfully.")

    # ===========================================================
    # STEP 3: Price-Creep Classification Engine
    # ===========================================================
    print("\n[3/4] Running Price-Creep Classification Engine...")
    creep_features = []
    
    for (user_id, merchant), group in df.groupby(["user_id", "merchant_clean"]):
        if len(group) < 3:
            continue
            
        amounts = group["amount"].values
        time_idx = np.arange(len(amounts))
        slope, _ = np.polyfit(time_idx, amounts, 1)
        starting_price = amounts[0]
        
        slope_pct = (slope / starting_price) * 100 if starting_price > 0 else 0
        total_change_pct = ((amounts[-1] - amounts[0]) / amounts[0]) * 100 if amounts[0] > 0 else 0
        amount_std = np.std(amounts)
        
        is_creep = group["is_price_creep"].max()
        
        creep_features.append({
            "user_id": user_id,
            "merchant_clean": merchant,
            "slope_pct": slope_pct,
            "total_change_pct": total_change_pct,
            "amount_std": amount_std,
            "is_price_creep": is_creep
        })
        
    creep_df = pd.DataFrame(creep_features)
    if not creep_df.empty:
        creep_train = creep_df[creep_df["user_id"].isin(train_users)]
        creep_test = creep_df[~creep_df["user_id"].isin(train_users)]
        
        X_creep_cols = ["slope_pct", "total_change_pct", "amount_std"]
        creep_clf = RandomForestClassifier(n_estimators=50, random_state=42)
        creep_clf.fit(creep_train[X_creep_cols], creep_train["is_price_creep"])
        creep_df["predicted_price_creep"] = creep_clf.predict(creep_df[X_creep_cols])
        print(" -> Price-Creep Classifier trained & predicted successfully.")
    else:
        creep_df["predicted_price_creep"] = 0

        # ===========================================================
    # STEP 4: Anomaly Detection Engine (Isolation Forest)
    # ===========================================================
    print("\n[4/4] Running Unsupervised Anomaly Detection Engine...")
    df["user_median_amount"] = df.groupby("user_id")["amount"].transform("median")
    df["user_std_amount"] = df.groupby("user_id")["amount"].transform("std").fillna(1.0)
    df["amount_zscore"] = (df["amount"] - df["user_median_amount"]) / df["user_std_amount"]

    iso_feature_cols = ["amount", "amount_zscore"]

    # CRITICAL FIX: split BEFORE fitting, so test users are genuinely
    # unseen. Fitting on the full dataset (train+test combined) means
    # the model has already "seen" the test users, which defeats the
    # entire purpose of the train/test split.
    iso_train_df = df[df["user_id"].isin(train_users)]
    iso_test_df = df[~df["user_id"].isin(train_users)]

    # CRITICAL FIX: estimate contamination from the TRAINING data's
    # actual known anomaly rate, instead of a hardcoded guess (0.02).
    # A contamination rate that doesn't match the real anomaly rate
    # forces the model to flag extra points just to meet that quota,
    # which is exactly what caused the false positives.
    train_anomaly_rate = iso_train_df["is_anomaly"].mean()
    contamination = max(train_anomaly_rate, 0.01)

    iso_forest = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    iso_forest.fit(iso_train_df[iso_feature_cols])  # fit ONLY on train users

    test_preds = iso_forest.predict(iso_test_df[iso_feature_cols])
    df.loc[iso_test_df.index, "predicted_anomaly"] = np.where(test_preds == -1, 1, 0)

    # For train rows, also predict (for completeness in the master CSV),
    # but these were seen during training so shouldn't be used for evaluation
    train_preds = iso_forest.predict(iso_train_df[iso_feature_cols])
    df.loc[iso_train_df.index, "predicted_anomaly"] = np.where(train_preds == -1, 1, 0)

    df["predicted_anomaly"] = df["predicted_anomaly"].astype(int)
    print(" -> Anomaly Detector executed successfully.")

    # ===========================================================
    # STEP 5: Consolidate & Output Master Report
    # ===========================================================
    output_path = "data/synthetic/finsight_master_analysis.csv"
    df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETE")
    print(f"Master results saved to: {output_path}")
    print("=" * 60)
    
    # Quick Evaluation Summary on Anomaly Detection as proof
    test_df_anomaly = df[~df["user_id"].isin(train_users)]
    print("\n[EVALUATION] Anomaly Detection on Unseen Test Users (4-5):")
    print(confusion_matrix(test_df_anomaly["is_anomaly"], test_df_anomaly["predicted_anomaly"]))
    print(classification_report(test_df_anomaly["is_anomaly"], test_df_anomaly["predicted_anomaly"], zero_division=0))

if __name__ == "__main__":
    run_finsight_pipeline()