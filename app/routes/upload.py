# app/routes/upload.py

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from sklearn.ensemble import IsolationForest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database import get_db
from app.ml.statement_parser import parse_bank_statement
from app.models.models import Transaction, User


router = APIRouter(
    prefix="/api",
    tags=["Statement Upload & ML Processing"],
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _safe_float(value, default: float = 0.0) -> float:
    """Convert a value to a finite float."""

    try:
        converted = float(value)

        if not np.isfinite(converted):
            return default

        return converted

    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int = 0) -> int:
    """Convert a value to an integer."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_string(value, default: str = "") -> str:
    """Convert missing values into safe strings."""

    if value is None or pd.isna(value):
        return default

    return str(value).strip()


def _make_json_safe_dataframe(df: pd.DataFrame) -> list[dict]:
    """
    Convert a DataFrame to JSON-safe records.

    This prevents NumPy scalar, NaN, and pandas timestamp serialization
    problems in FastAPI responses.
    """

    safe_df = df.copy()

    safe_df = safe_df.replace(
        {
            np.nan: None,
            pd.NaT: None,
        }
    )

    records = safe_df.to_dict(orient="records")

    return jsonable_encoder(records)


@router.post("/upload-statement")
async def upload_statement(
    file: UploadFile = File(...),
    user_id: str = "user_1",
    db: Session = Depends(get_db),
):
    """
    Upload, parse, analyze, persist, and return a bank/UPI statement.

    The endpoint:
    1. Validates the uploaded file.
    2. Saves it temporarily.
    3. Parses the statement.
    4. Runs the current ML inference pipeline.
    5. Creates the user if necessary.
    6. Persists transactions.
    7. Returns JSON-safe analytical results.
    8. Removes the temporary file.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have a filename.",
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV statement files are supported.",
        )

    if not user_id or not user_id.strip():
        raise HTTPException(
            status_code=400,
            detail="user_id cannot be empty.",
        )

    safe_filename = Path(file.filename).name
    temporary_filename = f"{uuid.uuid4().hex}_{safe_filename}"
    file_path = UPLOAD_DIR / temporary_filename

    try:
        # Save the uploaded file temporarily.
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Parse the bank/UPI statement.
        df = parse_bank_statement(
            str(file_path),
            user_id=user_id,
        )

        if df is None or df.empty:
            raise HTTPException(
                status_code=400,
                detail="No valid expense rows found in file.",
            )

        required_columns = {"description", "amount"}

        missing_columns = required_columns.difference(df.columns)

        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Parsed statement is missing required columns: "
                    f"{sorted(missing_columns)}"
                ),
            )

        df = df.copy()

        # Ensure required columns are clean and numeric.
        df["user_id"] = user_id
        df["description"] = (
            df["description"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        df["amount"] = pd.to_numeric(
            df["amount"],
            errors="coerce",
        )

        df = df.dropna(subset=["amount"])
        df = df[df["description"].str.len() > 0]

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="No valid transactions remained after cleaning.",
            )

        # Keep only finite numeric amounts.
        df = df[np.isfinite(df["amount"])]

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="No finite transaction amounts were found.",
            )

        # Subscription detection placeholder.
        #
        # This is only a simple temporary heuristic. Replace it with
        # the completed subscription model when available.
        description_counts = (
            df.groupby("description")["amount"]
            .transform("count")
        )

        df["predicted_subscription"] = (
            description_counts >= 2
        ).astype(int)

        # Price-creep detection placeholder.
        #
        # Replace this with the completed price-creep model.
        df["predicted_price_creep"] = 0

        # Robust anomaly features.
        user_median_amount = float(df["amount"].median())

        user_mad_amount = float(
            np.median(
                np.abs(
                    df["amount"].to_numpy(dtype=float)
                    - user_median_amount
                )
            )
        )

        if user_mad_amount > 0:
            robust_scale = 1.4826 * user_mad_amount

        else:
            standard_deviation = float(
                df["amount"].std(ddof=0)
            )

            robust_scale = (
                standard_deviation
                if standard_deviation > 0
                else 1.0
            )

        df["amount_zscore"] = (
            df["amount"] - user_median_amount
        ) / robust_scale

        df["amount_zscore"] = df["amount_zscore"].replace(
            [np.inf, -np.inf],
            0.0,
        ).fillna(0.0)

        # Isolation Forest requires enough observations.
        if len(df) >= 3:
            anomaly_features = df[
                ["amount", "amount_zscore"]
            ].astype(float)

            contamination = min(
                max(1.0 / len(df), 0.01),
                0.05,
            )

            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
            )

            predictions = iso_forest.fit_predict(
                anomaly_features
            )

            df["predicted_anomaly"] = (
                predictions == -1
            ).astype(int)

        else:
            df["predicted_anomaly"] = 0

        # Ensure the user exists before inserting transactions.
        db_user = (
            db.query(User)
            .filter(User.id == user_id)
            .first()
        )

        if db_user is None:
            db_user = User(id=user_id)
            db.add(db_user)
            db.flush()

        transaction_records: list[Transaction] = []

        for _, row in df.iterrows():
            raw_date = row.get(
                "transaction_date",
                row.get("date"),
            )

            transaction_date = pd.to_datetime(
                raw_date,
                errors="coerce",
            )

            if pd.isna(transaction_date):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "One or more transactions have an invalid "
                        "transaction date."
                    ),
                )

            merchant_clean = row.get(
                "merchant_clean",
                row.get("description", ""),
            )

            transaction = Transaction(
                user_id=user_id,
                transaction_date=transaction_date.date(),
                description=_safe_string(
                    row.get("description")
                ),
                amount=_safe_float(
                    row.get("amount")
                ),
                merchant_clean=_safe_string(
                    merchant_clean
                ),
                predicted_subscription=bool(
                    _safe_int(
                        row.get("predicted_subscription")
                    )
                ),
                predicted_price_creep=bool(
                    _safe_int(
                        row.get("predicted_price_creep")
                    )
                ),
                predicted_anomaly=bool(
                    _safe_int(
                        row.get("predicted_anomaly")
                    )
                ),
            )

            db.add(transaction)
            transaction_records.append(transaction)

        # Commit the user and all transactions atomically.
        db.commit()

        total_spent = _safe_float(
            df["amount"].sum()
        )

        total_transactions = len(df)

        anomalies_detected = int(
            df["predicted_anomaly"].sum()
        )

        subscriptions_detected = int(
            df["predicted_subscription"].sum()
        )

        transactions_list = _make_json_safe_dataframe(df)

        return {
            "status": "success",
            "filename": safe_filename,
            "summary": {
                "total_transactions_processed": total_transactions,
                "total_amount_spent": round(total_spent, 2),
                "subscriptions_flagged": subscriptions_detected,
                "anomalies_flagged": anomalies_detected,
            },
            "transactions": transactions_list,
        }

    except HTTPException:
        db.rollback()
        raise

    except SQLAlchemyError as exc:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail="Database error while saving transactions.",
        ) from exc

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"Error processing statement: {str(exc)}",
        ) from exc

    finally:
        try:
            file.file.close()
        except Exception:
            pass

        if file_path.exists():
            file_path.unlink()