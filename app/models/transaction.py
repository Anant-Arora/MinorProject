"""
Compatibility import for the Transaction model.

The actual SQLAlchemy Transaction model is defined only once in
app.models.models. This file re-exports it so older imports continue
to work without registering the transactions table twice.
"""

from app.models.models import Transaction

__all__ = ["Transaction"]