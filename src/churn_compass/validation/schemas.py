"""
Churn Compass - Data Validation Schema

Pandera schemas for validating datasets at various pipeline stages.
    - Raw schema and Training schema
"""

from pandera.pandas import Column, DataFrameSchema, Check

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
        "Exited": Column(int, checks=Check.isin([0, 1])),
    },
    strict=False,   # Allow extra columns (Complain, Satisfaction Score) in raw data
    coerce=True,    # Attempt type coercion
)

TRAINING_SCHEMA = DataFrameSchema(
    columns={k: v for k, v in RAW_SCHEMA.columns.items() if k not in {
        "RowNumber", "CustomerId", "Surname"
    }},
    strict=True,    # Extra columns not allowed
    coerce=True,
)
