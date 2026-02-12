"""
Churn Compass - Data Validation Schema

Pandera schemas for validating datasets at pipeline stages.

- RAW_SCHEMA: Validates data immediately after CSV ingestion (before customer_uuid generation)
- TRAINING_SCHEMA: Validates data ready for model training (must have customer_uuid, excludes PII)
"""

from pandera.pandas import Check, Column, DataFrameSchema

RAW_SCHEMA = DataFrameSchema(
    columns={
        "RowNumber": Column(int, nullable=False),
        "CustomerId": Column(int, nullable=False),
        "Surname": Column(str, nullable=False),
        "CreditScore": Column(int, checks=[Check.ge(300), Check.le(850)]),
        "Geography": Column(str, checks=Check.isin(["France", "Spain", "Germany"])),
        "Gender": Column(str, checks=Check.isin(["Male", "Female"])),
        "Age": Column(int, checks=[Check.ge(18), Check.le(100)]),
        "Tenure": Column(int, checks=[Check.ge(0), Check.le(10)]),
        "Balance": Column(float, checks=Check.ge(0)),
        "NumOfProducts": Column(int, checks=[Check.ge(1), Check.le(4)]),
        "HasCrCard": Column(int, checks=Check.isin([0, 1])),
        "IsActiveMember": Column(int, checks=Check.isin([0, 1])),
        "EstimatedSalary": Column(float, checks=[Check.ge(0), Check.le(200_000)]),
        "PointEarned": Column(int, checks=[Check.ge(0)]),
        "CardType": Column(
            str, checks=Check.isin(["Silver", "Gold", "Diamond", "Platinum"])
        ),
        "Exited": Column(int, checks=Check.isin([0, 1])),
    },
    strict=False,  # Allow extra columns (Complain, Satisfaction Score, customer_uuid)
    coerce=True,  # Attempt type coercion
)

TRAINING_SCHEMA = DataFrameSchema(
    columns={
        "customer_uuid": Column(str, nullable=False, unique=True),
        "CreditScore": Column(int, checks=[Check.ge(300), Check.le(850)]),
        "Geography": Column(str, checks=Check.isin(["France", "Spain", "Germany"])),
        "Gender": Column(str, checks=Check.isin(["Male", "Female"])),
        "Age": Column(int, checks=[Check.ge(18), Check.le(100)]),
        "Tenure": Column(int, checks=[Check.ge(0), Check.le(10)]),
        "Balance": Column(float, checks=Check.ge(0)),
        "NumOfProducts": Column(int, checks=[Check.ge(1), Check.le(4)]),
        "HasCrCard": Column(int, checks=Check.isin([0, 1])),
        "IsActiveMember": Column(int, checks=Check.isin([0, 1])),
        "EstimatedSalary": Column(float, checks=[Check.ge(0), Check.le(200_000)]),
        "PointEarned": Column(int, checks=[Check.ge(0)]),
        "CardType": Column(
            str, checks=Check.isin(["Silver", "Gold", "Diamond", "Platinum"])
        ),
        "Exited": Column(int, checks=Check.isin([0, 1])),
    },
    strict=True,  # No PII(RowNumber, CustomerId, Surname) allowed
    coerce=True,
)
