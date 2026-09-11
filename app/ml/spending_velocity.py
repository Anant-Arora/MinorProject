import os
import pandas as pd
import numpy as np

def monitor_spending_velocity(csv_path: str = "data/synthetic/transactions.csv"):
    if not os.path.exists(csv_path):
        csv_path = "../../data/synthetic/transactions.csv"
        
    print("[INFO] Loading dataset for Spending Velocity & Shortfall Analysis...")
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    
    # Extract Year-Month for monthly aggregation
    df["year_month"] = df["date"].dt.to_period("M")
    
    # ===========================================================
    # STEP 1: Calculate Historical Monthly Baselines per User
    # ===========================================================
    # Group by user and month to get total monthly spend
    monthly_spend = df.groupby(["user_id", "year_month"])["amount"].sum().reset_index()
    
    # Calculate historical average monthly spend and max spend per user
    user_baselines = monthly_spend.groupby("user_id").agg(
        avg_monthly_spend=("amount", "mean"),
        max_monthly_spend=("amount", "max")
    ).reset_index()
    
    # ===========================================================
    # STEP 2: Evaluate Current Month Velocity & Forecast Shortfalls
    # ===========================================================
    # Find the latest month present in the dataset for each user to simulate "current month" evaluation
    results = []
    
    for user_id, group in df.groupby("user_id"):
        latest_month = group["year_month"].max()
        current_month_df = group[group["year_month"] == latest_month]
        
        mtd_spend = current_month_df["amount"].sum()
        max_day = current_month_df["date"].dt.day.max() # Days elapsed or data points span in latest month
        
        # Estimate total days in this specific month
        days_in_month = current_month_df["date"].dt.days_in_month.iloc[0]
        
        # Linear velocity projection: (Spend so far / Days elapsed) * Total days
        # If max_day is small, fallback to days_in_month proportion
        effective_days_elapsed = max(max_day, 1)
        projected_month_spend = (mtd_spend / effective_days_elapsed) * days_in_month
        
        baseline_avg = user_baselines[user_baselines["user_id"] == user_id]["avg_monthly_spend"].values[0]
        
        # Shortfall risk condition: If projected spend exceeds historical average by > 15%
        shortfall_risk = projected_month_spend > (baseline_avg * 1.15)
        
        results.append({
            "user_id": user_id,
            "evaluated_month": str(latest_month),
            "mtd_spend": round(mtd_spend, 2),
            "projected_month_end_spend": round(projected_month_spend, 2),
            "historical_avg_baseline": round(baseline_avg, 2),
            "shortfall_risk_flag": int(shortfall_risk),
            "projected_overspend_pct": round(((projected_month_spend - baseline_avg) / baseline_avg) * 100, 2)
        })
        
    velocity_df = pd.DataFrame(results)
    
    print("\n" + "=" * 50)
    print("SPENDING VELOCITY & SHORTFALL FORECAST REPORT")
    print("=" * 50)
    print(velocity_df.to_string(index=False))
    
    output_path = "data/synthetic/spending_velocity_results.csv"
    velocity_df.to_csv(output_path, index=False)
    print(f"\nSaved velocity results to: {output_path}")
    
    return velocity_df

if __name__ == "__main__":
    monitor_spending_velocity()