"""
Churn Compass - Model Registry

Thread-safe model loading and caching from MLflow Registry.
"""

from typing import Optional, Dict, Any
import threading

import mlflow
import mlflow.sklearn as ms
from sklearn.pipeline import Pipeline

from churn_compass import settings, setup_logger

logger = setup_logger(__name__)


class ModelRegistry:
    """Thread-safe MLflow model registry loader with in-memory caching."""

    def __init__(self) -> None:
        self._model_cache: Dict[str, Pipeline] = {}
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

        logger.info("ModelRegistry initialized")

    def load_by_stage(
            self, 
            model_name: Optional[str] = None, 
            stage: str = "Production", 
            force_reload: bool = False
    ) -> Pipeline:
        """
        Load model from MLflow Registry by stage.
        
        :param model_name: Registered model name
        :type model_name: Optional[str]
        :param stage: Stage (Production, Staging, Local)
        :type stage: str
        :param force_reload: Bypass cache and reload model
        :type force_reload: bool
        :return: sklearn Pipeline
        :rtype: Pipeline
        """
        model_name = model_name or settings.mlflow_model_name
        cache_key = f"{model_name}:{stage}"

        with self._lock:
            if not force_reload and cache_key in self._model_cache:
                logger.info("Using cached model", extra={"cache_key": cache_key})
                return self._model_cache[cache_key]
            
            logger.info(
                "Loading model form MLflow registry", 
                extra={
                    "model": model_name, 
                    "stage": stage, 
                }, 
            )

            client = mlflow.MlflowClient()
            versions = client.get_latest_versions(model_name, stages=[stage])

            if not versions:
                raise RuntimeError(
                    f"No model found in registry for {model_name} at stage '{stage}'"
                )
            
            model_version = versions[0]
            model_uri = f"models:/{model_name}/{stage}"

            loaded = ms.load_model(model_uri=model_uri)

            if not isinstance(loaded, Pipeline):
                raise TypeError(
                    f"Loaded model is not a sklearn Pipeline. Got {type(loaded)}"
                )
            
            model: Pipeline = loaded
            
            # Fetch run details for metrics
            metrics = {}
            params = {}
            try:
                if model_version.run_id is not None:
                    run = client.get_run(model_version.run_id)
                    metrics = run.data.metrics
                    params = run.data.params
            except Exception:
                logger.warning(f"Failed to fetch run details for {model_version.run_id}")

            self._model_cache[cache_key] = model
            self._metadata_cache[cache_key] = {
                "model_name": model_name, 
                "stage": stage, 
                "version": model_version.version, 
                "run_id": model_version.run_id, 
                "creation_timestamp": model_version.creation_timestamp,
                "metrics": metrics,
                "params": params, 
            }

            logger.info(
                "Model loaded successfully", 
                extra=self._metadata_cache[cache_key], 
            )

            return model
        
    def load_by_run_id(self, run_id: str, force_reload: bool = False) -> Pipeline:
        """
        Load model from a specific MLflow run.
        
        :param run_id: MLflow run ID
        :type run_id: str
        :param force_reload: Bypass cache
        :type force_reload: bool
        :return: sklearn Pipeline
        :rtype: Pipeline
        """
        cache_key = f"run:{run_id}"

        with self._lock:
            if not force_reload and cache_key in self._model_cache:
                logger.info("Using cached model", extra={"cache_key": cache_key})
                return self._model_cache[cache_key]
            
            logger.info("Loading model from run", extra={"run_id": run_id})

            model_uri = f"runs:/{run_id}/model"
            loaded = ms.load_model(model_uri)

            if not isinstance(loaded, Pipeline):
                raise TypeError(
                    f"Loaded model from run {run_id} is not a sklearn Pipeline."
                    f"Got: {type(loaded)}"
                )
            
            model: Pipeline = loaded
            client = mlflow.MlflowClient()
            run = client.get_run(run_id)

            self._model_cache[cache_key] = model
            self._metadata_cache[cache_key] = {
                "run_id": run_id, 
                "experiment_id": run.info.experiment_id, 
                "start_time": run.info.start_time, 
                "params": run.data.params, 
                "metrics": run.data.metrics, 
            }

            return model      

    def get_metadata(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Return cached metadata for a model
        
        :param cache_key: Model cache key
        :type cache_key: str
        :return: Metadata dictionary or None
        :rtype: Dict[str, Any] | None
        """
        return self._metadata_cache.get(cache_key)
    
    def clear_cache(self) -> None:
        """Clear all cached models and metadata."""
        self._model_cache.clear()
        self._metadata_cache.clear()
        logger.warning("Model cache cleared")

    def get_latest_production_model(self) -> Pipeline:
        """
        Convenience method to get latest production model.
        
        Returns:
            Production model pipeline
        """
        return self.load_by_stage(stage="Production")


# Singleton registry instance
_registry: Optional[ModelRegistry] = None
_registry_lock = threading.Lock()


def get_model_registry() -> ModelRegistry:
    """Get global ModelRegistry instance"""
    global _registry

    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ModelRegistry()
    
    return _registry