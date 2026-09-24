import os
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from app.ml.statement_parser import parse_bank_statement

def run_finsight_pipeline(csv_path: str = "data/synthetic/transactions.csv", is_real_statement: bool = False):
    """
    Master pipeline orchestrator. 
    Can process synthetic labeled data (for training/evaluation) 
    or real-world parsed bank statements (for live inference).
    """
    print("=" * 60)
    print(f"FINSIGHT ML MASTER PIPELINE EXECUTION (Mode: {'REAL STATEMENT' if is_real_statement else 'SYNTHETIC DATASET'})")
    print("=" * 60)
    
    if is_real_statement:
        # Use our statement parser for real-world files
        df = parse_bank_statement(csv_path, user_id="real_user_1")
    else:
        if not os.path.exists(csv_path):
            csv_path = "../../data/synthetic/transactions.csv"
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(by=["user_id", "date"]).reset_index(drop=True)
    train_users = ["user_1", "user_2", "user_3"]

    # ===========================================================
    # STEP 1: Merchant Normalization Pass
    # ===========================================================
    print("\n[1/4] Running Merchant Normalization Engine...")
    if not is_real_statement:
        def clean_merchant(description):
            cleaned = re.sub(r'/\d{10,}/?$', '', str(description))
            return cleaned.strip()
        df["merchant_clean"] = df["description"].apply(clean_merchant)
    else:
        # Real statement descriptions are already cleaned by the parser
        df["merchant_clean"] = df["description"]
        
    print(f" -> Processed {len(df)} transactions through merchant normalization.")

    # ===========================================================
    # STEP 2: Subscription Classification Engine
    # ===========================================================
    print("\n[2/4] Running Subscription Classification Engine...")
    sub_features = []
    
    for (user_id, merchant), group in df.groupby(["user_id", "merchant_clean"]):
        dates = pd.to_datetime(group["date"]).sort_values()
        gaps = dates.diff().dropna().dt.days
        
        occurrences = len(group)
        mean_gap_days = gaps.mean() if len(gaps) > 0 else 0
        std_gap_days = gaps.std() if len(gaps) > 1 else 0
        mean_amount = group["amount"].mean()
        amount_cv = (group["amount"].std() / mean_amount) if mean_amount > 0 else 0
        
        is_sub = group["is_subscription"].max() if "is_subscription" in group else 0
        
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
    X_sub_cols = ["occurrences", "mean_gap_days", "std_gap_days", "amount_cv"]
    
    if not is_real_statement:
        sub_train = sub_df[sub_df["user_id"].isin(train_users)]
        sub_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        sub_clf.fit(sub_train[X_sub_cols], sub_train["is_subscription"])
        sub_df["predicted_subscription"] = sub_clf.predict(sub_df[X_sub_cols])
    else:
        # For real unlabelled data, we use a pre-trained fallback or rule-based heuristic if model isn't fitted on real data
        sub_df["predicted_subscription"] = np.where((sub_df["occurrences"] >= 2) & (sub_df["std_gap_days"] <= 5), 1, 0)
        
    print(" -> Subscription Classifier executed successfully.")

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
        
        is_creep = group["is_price_creep"].max() if "is_price_creep" in group else 0
        
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
        if not is_real_statement:
            creep_train = creep_df[creep_df["user_id"].isin(train_users)]
            X_creep_cols = ["slope_pct", "total_change_pct", "amount_std"]
            creep_clf = RandomForestClassifier(n_estimators=50, random_state=42)
            creep_clf.fit(creep_train[X_creep_cols], creep_train["is_price_creep"])
            creep_df["predicted_price_creep"] = creep_clf.predict(creep_df[X_creep_cols])
        else:
            creep_df["predicted_price_creep"] = np.where((creep_df["total_change_pct"] > 5.0), 1, 0)
        print(" -> Price-Creep Classifier executed successfully.")
    else:
        creep_df["predicted_price_creep"] = pd.Series(dtype=int)

    # ===========================================================
    # STEP 4: Anomaly Detection Engine (Isolation Forest)
    # ===========================================================
    print("\n[4/4] Running Unsupervised Anomaly Detection Engine...")
    df["user_median_amount"] = df.groupby("user_id")["amount"].transform("median")
    df["user_std_amount"] = df.groupby("user_id")["amount"].transform("std").fillna(1.0)
    df["amount_zscore"] = (df["amount"] - df["user_median_amount"]) / df["user_std_amount"]
    
    iso_features_col = ["amount", "amount_zscore"]
    iso_forest = IsolationForest(contamination=0.02, random_state=42)
    
    if not is_real_statement:
        train_df = df[df["user_id"].isin(train_users)]
        iso_forest.fit(train_df[iso_features_col])
    else:
        # Fit directly on the real statement sample data for local anomaly detection
        iso_forest.fit(df[iso_features_col])
        
    preds = iso_forest.predict(df[iso_features_col])
    df["predicted_anomaly"] = np.where(preds == -1, 1, 0)
    print(" -> Isolation Forest executed successfully.")

    # ===========================================================
    # STEP 5: Consolidate & Output Master Report
    # ===========================================================
    output_path = "data/synthetic/finsight_master_analysis.csv" if not is_real_statement else "data/real_statement_analysis.csv"
    df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETE")
    print(f"Master results successfully saved to: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    # Test running on synthetic dataset by default, or set is_real_statement=True for real files
    run_finsight_pipeline(csv_path="data/synthetic/transactions.csv", is_real_statement=False)