"""Generate synthetic customer churn data for pipeline testing."""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_churn_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Create a realistic churn dataset with rule-based label generation.

    Churn probability is elevated when:
    - Tenure < 12 months AND NumProducts == 1  (+35 pp)
    - IsActiveMember == 0                       (+20 pp)
    - MonthlyCharges > 70                       (+10 pp)

    The base rate is calibrated so overall churn ≈ 20 %.

    Parameters
    ----------
    n_samples:
        Number of rows to generate.
    seed:
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    customer_ids = [f"CUST_{i:04d}" for i in range(1, n_samples + 1)]
    age              = rng.integers(25, 71,  size=n_samples)
    tenure           = rng.integers(0,  61,  size=n_samples)
    monthly_charges  = rng.uniform(20, 100,  size=n_samples).round(2)
    total_charges    = (monthly_charges * tenure + rng.normal(0, 50, n_samples)).clip(0).round(2)
    num_products     = rng.integers(1, 5,    size=n_samples)
    has_credit_card  = rng.integers(0, 2,    size=n_samples)
    is_active_member = rng.integers(0, 2,    size=n_samples)
    estimated_salary = rng.uniform(20_000, 150_000, size=n_samples).round(2)
    geography        = rng.choice(["North", "South", "East", "West"], size=n_samples)
    gender           = rng.choice(["Male", "Female"], size=n_samples)

    # Rule-based churn probability — strong, well-separated signals
    p_churn = np.full(n_samples, 0.05)                           # low base rate
    p_churn[(tenure < 12) & (num_products == 1)] += 0.60        # strongest signal
    p_churn[is_active_member == 0]               += 0.35        # second signal
    p_churn[monthly_charges > 70]                += 0.20        # third signal
    # Interaction bonus: inactive + expensive → very high risk
    p_churn[(is_active_member == 0) & (monthly_charges > 70)]   += 0.20
    p_churn = np.clip(p_churn, 0.0, 1.0)

    churn = rng.binomial(1, p_churn).astype(int)

    return pd.DataFrame(
        {
            "CustomerID":       customer_ids,
            "Age":              age,
            "Tenure":           tenure,
            "MonthlyCharges":   monthly_charges,
            "TotalCharges":     total_charges,
            "NumProducts":      num_products,
            "HasCreditCard":    has_credit_card,
            "IsActiveMember":   is_active_member,
            "EstimatedSalary":  estimated_salary,
            "Geography":        geography,
            "Gender":           gender,
            "Churn":            churn,
        }
    )


def save_datasets(output_dir: Path = Path("data"), n_samples: int = 1000) -> None:
    """Generate and save the churn dataset to *output_dir*/churn_sample.csv."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = generate_churn_data(n_samples=n_samples)
    out_path = output_dir / "churn_sample.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(f"Shape: {df.shape}")
    print(f"Churn distribution:\n{df['Churn'].value_counts().to_string()}")
    print(f"Churn rate: {df['Churn'].mean():.1%}")


if __name__ == "__main__":
    save_datasets()
