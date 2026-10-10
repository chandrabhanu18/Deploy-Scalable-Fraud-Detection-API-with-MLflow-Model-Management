"""
generate_data.py
Generates realistic synthetic financial transaction data for fraud detection model training and evaluation.
Features generated:
- transaction_amount: float (Transaction value in USD)
- time_of_day_hour: int (Hour of the transaction, 0-23)
- num_transactions_1h: int (Number of transactions in the last hour)
- avg_transaction_7d: float (Historical 7-day average transaction amount)
- location_risk_score: float (Location risk score between 0.0 and 1.0)
- is_fraud: int (0 for legitimate, 1 for fraudulent)
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd


def generate_synthetic_data(
    num_samples: int = 5000,
    random_seed: int = 42,
    base_fraud_rate: float = 0.05
) -> pd.DataFrame:
    """
    Generate synthetic transactions with realistic risk patterns.
    """
    np.random.seed(random_seed)

    # 1. Base Feature Generation
    # Transaction amount: log-normal distribution centered around $80 with long tail
    transaction_amounts = np.random.lognormal(mean=4.2, sigma=1.1, size=num_samples)
    transaction_amounts = np.clip(transaction_amounts, 5.0, 10000.0).round(2)

    # Time of day: 0 to 23 hours with realistic day/night probabilities
    hour_probs = np.array([
        0.015, 0.010, 0.008, 0.005, 0.005, 0.010, # 00:00 - 05:00 (late night)
        0.025, 0.040, 0.060, 0.070, 0.075, 0.080, # 06:00 - 11:00 (morning)
        0.085, 0.080, 0.075, 0.070, 0.075, 0.080, # 12:00 - 17:00 (afternoon)
        0.075, 0.065, 0.050, 0.035, 0.025, 0.022  # 18:00 - 23:00 (evening)
    ])
    hour_probs = hour_probs / hour_probs.sum()
    time_of_day_hours = np.random.choice(np.arange(24), size=num_samples, p=hour_probs)

    # Historical 7-day average: correlated with transaction amount
    avg_transaction_7d = np.random.lognormal(mean=4.0, sigma=0.9, size=num_samples)
    avg_transaction_7d = np.clip(avg_transaction_7d, 10.0, 5000.0).round(2)

    # Burst transaction count in last 1h: Poisson distribution
    num_transactions_1h = np.random.poisson(lam=1.5, size=num_samples)
    num_transactions_1h = np.clip(num_transactions_1h, 0, 25)

    # Location risk score: Beta distribution (mostly low, occasional high)
    location_risk_scores = np.random.beta(a=1.5, b=5.0, size=num_samples).round(4)
    location_risk_scores = np.clip(location_risk_scores, 0.01, 0.99)

    df = pd.DataFrame({
        "transaction_amount": transaction_amounts,
        "time_of_day_hour": time_of_day_hours,
        "num_transactions_1h": num_transactions_1h,
        "avg_transaction_7d": avg_transaction_7d,
        "location_risk_score": location_risk_scores,
    })

    # 2. Realistic Fraud Probability Modeling (Multi-factor Risk Score)
    # Factor A: Amount anomaly ratio (spending way above normal average)
    amount_ratio = df["transaction_amount"] / (df["avg_transaction_7d"] + 1e-5)
    amount_risk = np.where(amount_ratio > 4.0, 0.40, np.where(amount_ratio > 2.0, 0.15, 0.02))

    # Factor B: High absolute amount
    high_amount_risk = np.where(df["transaction_amount"] > 3500, 0.35, 0.0)

    # Factor C: Late night / unusual hours (01:00 - 05:00)
    night_risk = np.where((df["time_of_day_hour"] >= 1) & (df["time_of_day_hour"] <= 5), 0.15, 0.0)

    # Factor D: Transaction burst frequency
    velocity_risk = np.where(df["num_transactions_1h"] >= 6, 0.35, np.where(df["num_transactions_1h"] >= 4, 0.15, 0.0))

    # Factor E: High risk location
    location_risk = np.where(df["location_risk_score"] > 0.75, 0.40, np.where(df["location_risk_score"] > 0.50, 0.15, 0.0))

    # Composite fraud probability score with sigmoid transformation
    raw_score = (
        0.30 * amount_risk +
        0.25 * high_amount_risk +
        0.15 * night_risk +
        0.25 * velocity_risk +
        0.35 * location_risk +
        np.random.normal(0, 0.05, size=num_samples)
    )

    # Add strong deterministic rules specified in guidelines
    rule_mask = (df["transaction_amount"] > 3500) & (df["location_risk_score"] > 0.70)
    raw_score = np.where(rule_mask, raw_score + 0.5, raw_score)

    # Normalize to probability between [0, 1]
    fraud_prob = 1 / (1 + np.exp(-(raw_score - 0.45) * 6))

    # Assign binary label based on threshold and base fraud rate calibration
    is_fraud = (np.random.uniform(0, 1, size=num_samples) < fraud_prob).astype(int)

    # Ensure rule-based triggers are strictly positive
    is_fraud = np.where((df["transaction_amount"] > 4000) & (df["location_risk_score"] > 0.8), 1, is_fraud)

    df["is_fraud"] = is_fraud

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic financial transactions for fraud detection.")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of transaction samples to generate")
    parser.add_argument("--output-path", type=str, default="data/raw_transactions.csv", help="Path to save raw CSV file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    output_file = Path(args.output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_samples} synthetic transaction records...")
    df = generate_synthetic_data(num_samples=args.num_samples, random_seed=args.seed)

    df.to_csv(output_file, index=False)
    print(f"Successfully saved {len(df)} transactions to {output_file}")
    print("\nDataset Summary:")
    print(f"- Total Records: {len(df):,}")
    print(f"- Fraud Cases: {df['is_fraud'].sum():,} ({df['is_fraud'].mean() * 100:.2f}%)")
    print(f"- Legitimate Cases: {(df['is_fraud'] == 0).sum():,} ({(1 - df['is_fraud'].mean()) * 100:.2f}%)")
    print(f"- Amount Range: ${df['transaction_amount'].min():.2f} - ${df['transaction_amount'].max():.2f}")
    print(f"- Average Amount: ${df['transaction_amount'].mean():.2f}")


if __name__ == "__main__":
    main()
