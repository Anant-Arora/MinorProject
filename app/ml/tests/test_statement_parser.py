from pathlib import Path

import pandas as pd

from app.ml.statement_parser import parse_bank_statement


# test_statement_parser.py
# parents[3] points to the project root:
# app/ml/tests/test_statement_parser.py
# parents[0] -> app/ml/tests
# parents[1] -> app/ml
# parents[2] -> app
# parents[3] -> finsight-backend
PROJECT_ROOT = Path(__file__).resolve().parents[3]

SAMPLE_CSV_PATH = PROJECT_ROOT / "data" / "real_sample.csv"


def test_parse_bank_statement():
    """Verify that the real bank statement is parsed correctly."""

    assert SAMPLE_CSV_PATH.is_file(), (
        f"Sample CSV not found: {SAMPLE_CSV_PATH}\n"
        "Expected file location: data/real_sample.csv"
    )

    df = parse_bank_statement(
        str(SAMPLE_CSV_PATH),
        user_id="test_user",
    )

    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "Parsed DataFrame is empty."
    assert len(df) > 0, "No transactions were parsed from the sample file."

    required_columns = {"user_id", "amount", "description"}
    missing_columns = required_columns.difference(df.columns)

    assert not missing_columns, (
        f"Missing required columns: {sorted(missing_columns)}. "
        f"Available columns: {list(df.columns)}"
    )

    assert df["user_id"].notna().all(), (
        "Some parsed transactions have a missing user_id."
    )

    assert (df["user_id"] == "test_user").all(), (
        "Every parsed transaction must have user_id='test_user'."
    )

    assert df["amount"].notna().all(), (
        "Some parsed transactions have a missing amount."
    )

    numeric_amounts = pd.to_numeric(df["amount"], errors="coerce")

    assert numeric_amounts.notna().all(), (
        "The amount column contains non-numeric values."
    )

    assert df["description"].notna().all(), (
        "Some parsed transactions have a missing description."
    )

    assert df["description"].astype(str).str.strip().ne("").all(), (
        "Some parsed transactions have an empty description."
    )