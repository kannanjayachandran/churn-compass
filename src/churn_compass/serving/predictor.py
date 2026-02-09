"""
Churn Compass - Prediction Service

Handles single and batch predictions with optional SHAP explanations.
"""

from typing import Dict, List, Optional, Union, Any
import re

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from churn_compass import get_settings, setup_logger
from churn_compass.serving import get_model_registry

logger = setup_logger(__name__)


class ChurnPredictor:
    """Churn prediction service"""

    def __init__(self, model: Optional[Pipeline] = None):
        self._settings = get_settings()
        if model is None:
            loader = get_model_registry()
            self.model: Pipeline = loader.get_latest_production_model()
        else:
            self.model = model

        self._explainer: Optional[Any] = None
        logger.info("ChurnPredictor initialized")

    # Core Prediction
    def predict_single(
        self, customer_data: Dict[str, Union[str, int, float]]
    ) -> Dict[str, Any]:
        df = pd.DataFrame([customer_data])
        return self._predict_df(df).iloc[0].to_dict()

    def predict_batch(
        self, df: pd.DataFrame, include_features: bool = False
    ) -> pd.DataFrame:
        if df.empty:
            raise ValueError("Empty DataFrame received for prediction")

        results = self._predict_df(df)

        if include_features:
            results = pd.concat([df.reset_index(drop=True), results], axis=1)

        logger.info(
            "Batch prediction completed",
            extra={
                "rows": len(results),
                "mean_probability": float(results["probability"].mean()),
            },
        )

        return results

    def _predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self.model, "predict_proba"):
            raise RuntimeError("Loaded model does not support predict_proba")

        proba = self.model.predict_proba(df)[:, 1]
        probas = np.clip(proba, 0.0, 1.0)

        threshold = self._settings.prediction_threshold

        return pd.DataFrame(
            {
                "probability": probas,
                "prediction": (probas >= threshold).astype(int),
                "risk_level": self._vectorized_risk_level(probas),
            }
        )

    # Top-K targeting
    def get_top_k_customers(
        self,
        df: pd.DataFrame,
        k: Optional[int] = None,
        k_percent: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Docstring for get_top_k_customers

        :param self: Description
        :param df: Description
        :type df: pd.DataFrame
        :param k: Description
        :type k: Optional[int]
        :param k_percent: Description
        :type k_percent: Optional[float]
        :return: Description
        :rtype: DataFrame
        """
        if k is None and k_percent is None:
            k_percent = self._settings.top_k_percent

        results = self.predict_batch(df, include_features=True)

        if k is None:
            if k_percent is None:
                raise ValueError("Either k or k_percent must be provided")
            k = max(1, int(len(results) * k_percent))

        top_k = results.nlargest(k, "probability")

        logger.info(
            "Top-K customers identified",
            extra={"k": k, "min_prob": float(top_k["probability"].min())},
        )

        return top_k

    # SHAP explanations
    def explain_prediction(
        self,
        customer_data: Dict[str, Union[str, int, float]],
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """
        Docstring for explain_prediction

        :param self: Description
        :param customer_data: Description
        :type customer_data: Dict[str, Union[str, int, float]]
        :param top_n: Description
        :type top_n: int
        :return: Description
        :rtype: Dict[str, Any]
        """
        prediction = self.predict_single(customer_data)

        if not self._settings.enable_shap_explanations:
            logger.info("SHAP explanation disabled in settings")
            return {**prediction, "explanation": "SHAP disabled"}

        try:
            df = pd.DataFrame([customer_data])

            if self._explainer is None:
                self._initialize_explainer()

            X = self._transform_for_shap(df)

            assert self._explainer is not None
            shap_values = self._explainer.shap_values(X)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            contributions = shap_values[0]
            feature_names = self._get_feature_names(
                X.shape[1], input_features=df.columns.tolist()
            )

            importance = sorted(
                zip(feature_names, contributions),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:top_n]

            base_value = self._explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                # If shap_values was a list (multi-class), we took index 1.
                # transform expected_value similarly if needed.
                # Assuming binary classification where we care about the positive class (index 1)
                if len(base_value) > 1:
                    base_value = base_value[1]
                else:
                    base_value = base_value[0]

            return {
                **prediction,
                "explanation": {
                    "top_features": [
                        {
                            "feature": self._sanitize_feature_name(f),
                            "contribution": float(v),
                            "impact": "increase" if v > 0 else "decrease",
                        }
                        for f, v in importance
                    ],
                    "base_value": float(base_value),
                    "prediction_value": float(base_value + sum(contributions)),
                },
            }

        except Exception:
            logger.error("SHAP explanation failed", exc_info=True)
            return {**prediction, "explanation": "Explanation failed"}

    def _initialize_explainer(self) -> None:
        """Docstring for _initialize_explainer"""
        logger.info("Initialize SHAP explainer (Lazy Load)")
        import shap

        if hasattr(self.model, "named_steps") and "model" in self.model.named_steps:
            self._explainer = shap.TreeExplainer(self.model.named_steps["model"])
        else:
            raise RuntimeError("Unsupported model type for SHAP")

    def _transform_for_shap(self, df: pd.DataFrame) -> np.ndarray:
        if hasattr(self.model, "named_steps"):
            return self.model.named_steps["preprocessing"].transform(df)

        return df.values

    def _get_feature_names(
        self, n_features: int, input_features: Optional[List[str]] = None
    ) -> List[str]:
        try:
            preprocessing = self.model.named_steps["preprocessing"]
            # Try standard scikit-learn API first
            return preprocessing.get_feature_names_out(input_features).tolist()
        except Exception:
            try:
                # Fallback to our custom extractor which is more robust to nested pipelines
                from churn_compass.features.feature_engineering_pipeline import (
                    extract_feature_names,
                )

                return extract_feature_names(preprocessing)
            except Exception:
                logger.warning(
                    "All feature name extraction methods failed, using generic names"
                )
                return [f"feature_{i}" for i in range(n_features)]

    # Utilities
    @staticmethod
    def _sanitize_feature_name(name: str) -> str:
        """Sanitize feature names for display"""
        # Strip internal prefixes
        for prefix in ["num__", "cat__", "m__"]:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break

        # Replace underscores with spaces
        name = name.replace("_", " ")

        # Split PascalCase/CamelCase (e.g., NumOfProducts -> Num Of Products)
        name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)

        # Final formatting: Title Case and strip extra spaces
        return name.title().strip()

    @staticmethod
    def _vectorized_risk_level(probabilities: np.ndarray) -> np.ndarray:
        return np.select(
            [probabilities >= 0.7, probabilities >= 0.4],
            ["high", "medium"],
            default="low",
        )

    @staticmethod
    def _get_risk_level(probability: float) -> str:
        if probability >= 0.7:
            return "high"
        if probability >= 0.4:
            return "medium"
        return "low"
