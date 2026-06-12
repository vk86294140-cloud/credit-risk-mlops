"""Reproducible dataset generation.

The pipeline is built to be fully reproducible and to run offline / in CI, so
the default dataset is *synthetic but realistic*: defaults are driven by a
documented logistic process over the same drivers a credit model cares about
(debt-to-income, delinquencies, utilization, income). Swapping in a real CSV is
a one-line change (`load_dataset(path=...)`) since the schema is shared via
config.

This keeps the focus where a portfolio MLOps project should be: the pipeline,
the serving layer, testing, and reproducibility — not data wrangling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_SEED, TARGET

HOME_OWNERSHIP = ["rent", "own", "mortgage"]
LOAN_PURPOSE = ["debt_consolidation", "credit_card", "home_improvement", "other"]


def generate_dataset(n_rows: int = 20_000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate a synthetic credit-application dataset with a realistic target."""

    rng = np.random.default_rng(seed)

    age = rng.normal(40, 12, n_rows).clip(18, 85)
    annual_income = rng.lognormal(mean=11.0, sigma=0.45, size=n_rows).clip(15_000, 500_000)
    loan_amount = (annual_income * rng.uniform(0.05, 0.6, n_rows)).clip(1_000, 100_000)
    loan_term_months = rng.choice([12, 24, 36, 48, 60], size=n_rows)
    interest_rate = rng.normal(13, 4, n_rows).clip(4, 30)
    debt_to_income = rng.beta(2, 5, n_rows) * 0.8           # 0 - 0.8
    credit_utilization = rng.beta(2, 3, n_rows)             # 0 - 1
    num_open_accounts = rng.poisson(6, n_rows).clip(0, 30)
    num_delinquencies = rng.poisson(0.4, n_rows).clip(0, 15)
    months_employed = rng.gamma(shape=2.0, scale=30, size=n_rows).clip(0, 480)

    home_ownership = rng.choice(HOME_OWNERSHIP, size=n_rows, p=[0.45, 0.2, 0.35])
    loan_purpose = rng.choice(LOAN_PURPOSE, size=n_rows, p=[0.4, 0.3, 0.15, 0.15])

    # Latent default risk: a documented logistic combination of the drivers.
    logit = (
        -4.3
        + 6.5 * debt_to_income
        + 3.2 * credit_utilization
        + 0.85 * num_delinquencies
        + 0.09 * (interest_rate - 13)
        - 0.000008 * (annual_income - 60_000)
        - 0.006 * months_employed
        + 0.000018 * (loan_amount - 15_000)
    )
    # Mortgage holders default a little less; renters a little more.
    logit += np.where(home_ownership == "mortgage", -0.25, 0.0)
    logit += np.where(home_ownership == "rent", 0.15, 0.0)

    prob = 1.0 / (1.0 + np.exp(-logit))
    defaulted = (rng.uniform(0, 1, n_rows) < prob).astype(int)

    df = pd.DataFrame(
        {
            "age": age.round(0),
            "annual_income": annual_income.round(2),
            "loan_amount": loan_amount.round(2),
            "loan_term_months": loan_term_months,
            "interest_rate": interest_rate.round(2),
            "debt_to_income": debt_to_income.round(4),
            "credit_utilization": credit_utilization.round(4),
            "num_open_accounts": num_open_accounts,
            "num_delinquencies": num_delinquencies,
            "months_employed": months_employed.round(1),
            "home_ownership": home_ownership,
            "loan_purpose": loan_purpose,
            TARGET: defaulted,
        }
    )
    # Reorder so feature columns come first, target last.
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]]


def load_dataset(path: str | Path | None = None, **kwargs) -> pd.DataFrame:
    """Load a CSV dataset, or generate the synthetic one when no path is given."""
    if path is not None:
        return pd.read_csv(path)
    return generate_dataset(**kwargs)


def save_dataset(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
