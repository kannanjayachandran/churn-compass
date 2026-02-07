"""
Tests for API schemas.

Tests Pydantic model validation contracts for API request/response schemas.
"""

import pytest
from pydantic import ValidationError

from churn_compass.api.schemas import (
    CustomerInput,
    PredictionResponse,
    BatchPredictionRequest,
    TopKRequest,
)


@pytest.fixture
def valid_customer_dict():
    """Valid customer input dictionary."""
    return {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Male",
        "Age": 35,
        "Tenure": 5,
        "Balance": 50000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 75000.0,
        "CardType": "Gold",
    }


# CustomerInput tests
def test_customer_input_accepts_valid_data(valid_customer_dict):
    """Valid customer data should be accepted."""
    customer = CustomerInput(**valid_customer_dict)

    assert customer.CreditScore == 650
    assert customer.Geography == "France"


def test_customer_input_normalizes_cardtype():
    """CardType should be normalized (case-insensitive input)."""
    data = {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Male",
        "Age": 35,
        "Tenure": 5,
        "Balance": 50000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 75000.0,
        "CardType": "GOLD",  # Uppercase
    }

    customer = CustomerInput(**data)

    assert customer.CardType == "Gold"


def test_customer_input_normalizes_geography():
    """Geography should be normalized."""
    data = {
        "CreditScore": 650,
        "Geography": "france",  # Lowercase
        "Gender": "Male",
        "Age": 35,
        "Tenure": 5,
        "Balance": 50000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 75000.0,
        "CardType": "Gold",
    }

    customer = CustomerInput(**data)

    assert customer.Geography == "France"


def test_customer_input_rejects_invalid_creditscore(valid_customer_dict):
    """Credit score outside 300-850 should fail validation."""
    valid_customer_dict["CreditScore"] = 200  # Below 300

    with pytest.raises(ValidationError):
        CustomerInput(**valid_customer_dict)


def test_customer_input_rejects_creditscore_above_max(valid_customer_dict):
    """Credit score above 850 should fail validation."""
    valid_customer_dict["CreditScore"] = 900

    with pytest.raises(ValidationError):
        CustomerInput(**valid_customer_dict)


def test_customer_input_rejects_invalid_geography(valid_customer_dict):
    """Invalid geography should fail validation."""
    valid_customer_dict["Geography"] = "Italy"

    with pytest.raises(ValidationError):
        CustomerInput(**valid_customer_dict)


def test_customer_input_rejects_invalid_gender(valid_customer_dict):
    """Invalid gender should fail validation."""
    valid_customer_dict["Gender"] = "Other"

    with pytest.raises(ValidationError):
        CustomerInput(**valid_customer_dict)


def test_customer_input_rejects_invalid_cardtype(valid_customer_dict):
    """Invalid card type should fail validation."""
    valid_customer_dict["CardType"] = "Bronze"

    with pytest.raises(ValidationError):
        CustomerInput(**valid_customer_dict)


def test_customer_input_rejects_negative_balance(valid_customer_dict):
    """Negative balance should fail validation."""
    valid_customer_dict["Balance"] = -1000.0

    with pytest.raises(ValidationError):
        CustomerInput(**valid_customer_dict)


def test_customer_input_rejects_invalid_age(valid_customer_dict):
    """Age outside 10-100 range should fail."""
    valid_customer_dict["Age"] = 5

    with pytest.raises(ValidationError):
        CustomerInput(**valid_customer_dict)


# PredictionResponse tests
def test_prediction_response_valid_risk_levels():
    """PredictionResponse should accept valid risk levels."""
    for risk in ["low", "medium", "high"]:
        response = PredictionResponse(
            prediction=1,
            probability=0.5,
            risk_level=risk,
        )
        assert response.risk_level == risk


def test_prediction_response_rejects_invalid_risk_level():
    """Invalid risk level should fail validation."""
    with pytest.raises(ValidationError):
        PredictionResponse(
            prediction=1,
            probability=0.5,
            risk_level="critical",  # Invalid
        )


# BatchPredictionRequest tests
def test_batch_request_requires_at_least_one_customer(valid_customer_dict):
    """Batch request must have at least 1 customer."""
    with pytest.raises(ValidationError):
        BatchPredictionRequest(customers=[])


def test_batch_request_accepts_valid_customers(valid_customer_dict):
    """Batch request should accept valid customer list."""
    request = BatchPredictionRequest(customers=[CustomerInput(**valid_customer_dict)])

    assert len(request.customers) == 1


# TopKRequest tests
def test_topk_request_optional_fields():
    """TopKRequest fields should be optional."""
    request = TopKRequest()

    assert request.k is None
    assert request.k_percent is None


def test_topk_request_k_validation():
    """k must be >= 1."""
    with pytest.raises(ValidationError):
        TopKRequest(k=0)


def test_topk_request_k_percent_validation():
    """k_percent must be between 0 and 1."""
    with pytest.raises(ValidationError):
        TopKRequest(k_percent=1.5)
