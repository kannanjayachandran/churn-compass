import io
import platform as pf
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

import pandas as pd
import psutil
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

from churn_compass import __version__, get_settings, setup_logger
from churn_compass.api import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerInput,
    ExplanationResponse,
    HealthResponse,
    PredictionResponse,
    SystemStatusResponse,
    TopKRequest,
    TopKResponse,
    VersionResponse,
)
from churn_compass.config.settings import Settings
from churn_compass.serving import ChurnPredictor, get_model_registry
from churn_compass.utils.normalization import normalize_dataframe

logger = setup_logger(__name__)

settings = get_settings()


# Request ID Middleware for Observability
class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request for tracing."""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Add to response headers for client-side tracing
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        return response


# Lifespan with startup validation
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management with health checks"""
    logger.info(
        "Starting Churn Compass API",
        extra={
            "version": __version__,
            "environment": settings.environment,
            "mlflow_url": settings.mlflow_tracking_uri,
        },
    )

    # Startup validation
    startup_errors = []

    # 1. Check MLflow connectivity
    try:
        loader = get_model_registry()
        model = loader.get_latest_production_model()
        app.state.predictor = ChurnPredictor(model=model)

        logger.info(
            "Model loaded successfully ",
            extra={"model_name": settings.mlflow_model_name},
        )
    except Exception as e:
        logger.exception("Failed to load model during startup")
        startup_errors.append(f"Model loading: {str(e)}")
        app.state.predictor = None

    # 2. Check database connectivity (if enabled)
    if settings.db_type == "postgres":
        try:
            from sqlalchemy import create_engine, text

            engine = create_engine(settings.get_postgres_uri())
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection successful")
        except Exception as e:
            logger.warning("Database connection failed", extra={"error": str(e)})
            startup_errors.append(f"Database: {str(e)}")

    # 3. Validate critical directories exist
    try:
        assert settings.data_raw_dir.exists(), f"Missing: {settings.data_raw_dir}"
        assert settings.data_processed_dir.exists(), (
            f"Missing: {settings.data_processed_dir}"
        )
        logger.info("Data directories validated")
    except AssertionError as e:
        logger.warning(f"Directory validation failed: {e}")
        startup_errors.append(str(e))

    # Log startup summary
    if startup_errors:
        logger.warning(
            "API started with errors",
            extra={"errors": startup_errors, "count": len(startup_errors)},
        )
    else:
        logger.info("All Startup checks passed - API ready")

    # Store startup health for /health endpoint
    app.state.startup_error = startup_errors

    yield

    logger.info("Shutting down Churn Compass API")
    app.state.predictor = None
    if hasattr(app.state, "startup_errors"):
        delattr(app.state, "startup_errors")


# FastAPI app initialization
app = FastAPI(
    title="Churn Compass API",
    description="Customer Churn Prediction API for Retail Banking",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs" if settings.environment != "prod" else None,
    redoc_url="/redoc" if settings.environment != "prod" else None,
)

# CORS
allowed_origins = (
    ["*"]
    if settings.environment == "local"
    else [
        "https://cc.example.com",  # frontend production url
        "https://app.churn.e.com",
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request ID middleware
app.add_middleware(RequestIDMiddleware)


# Dependencies: Get predictor with request context
def get_predictor(request: Request) -> ChurnPredictor:
    """Get predictor instance with request tracing."""
    predictor = request.app.state.predictor
    if predictor is None:
        logger.error(
            "Model not loaded - service unavailable",
            extra={"request_id": getattr(request.state, "request_id", None)},
        )
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable. Check /health for details.",
        )
    return predictor


# Dependency: Get settings (for routes that need it)
def get_app_settings() -> Settings:
    """Get application settings (cached singleton)"""
    return settings


# Routes
@app.get("/", tags=["Root"])
async def root():
    """API root with nav links."""
    return {
        "message": "Churn Compass API",
        "version": __version__,
        "environment": settings.environment,
        "endpoints": {
            "docs": "/docs" if settings.environment != "prod" else "disabled",
            "health": "/health",
            "system_status": "/system/status",
            "version": "/version",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(request: Request):
    """
    Liveness probe for orchestrators (k8s, docker).

    Returns 200 if API is running, regardless of model state. For detailed health, use /system/status.
    """
    predictor_loaded = request.app.state.predictor is not None

    status = "healthy" if predictor_loaded else "unhealthy"
    return HealthResponse(
        status=status,
        model_loaded=predictor_loaded,
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=__version__,
    )


@app.get("/version", response_model=VersionResponse, tags=["Health"])
async def get_version(settings: Settings = Depends(get_app_settings)):
    loader = get_model_registry()
    cache_key = f"{settings.mlflow_model_name}:Production"
    metadata = loader.get_metadata(cache_key)

    return VersionResponse(
        api_version=__version__,
        model_name=settings.mlflow_model_name,
        model_version=metadata.get("version") if metadata else None,
        model_stage=metadata.get("stage") if metadata else None,
        mlflow_tracking_uri=settings.mlflow_tracking_uri,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(
    request: Request,
    customer: CustomerInput,
    predictor: ChurnPredictor = Depends(get_predictor),
):
    """
    Predict churn for a single customer.

    Returns probability, binary prediction, and risk level.
    """
    request_id = getattr(request.state, "request_id", None)
    try:
        result = predictor.predict_single(customer.model_dump())
        logger.info(
            "Single prediction completed",
            extra={
                "request_id": request_id,
                "probability": result.get("probability"),
                "risk_level": result.get("risk_level"),
            },
        )
        return PredictionResponse(**result)
    except Exception as e:
        logger.exception(
            "Single prediction failed",
            extra={"request_id": request_id, "error": str(e)},
        )
        raise HTTPException(status_code=500, detail="Prediction failed: {str(e)}")


@app.post("/explain", response_model=ExplanationResponse, tags=["Prediction"])
async def explain_prediction(
    request: Request,
    customer: CustomerInput,
    top_n: int = 10,
    predictor: ChurnPredictor = Depends(get_predictor),
    settings: Settings = Depends(get_app_settings),
):
    """
    Get SHAP explanations for a prediction.

    Return top N feature contribution to the churn prediction.
    """
    if not settings.enable_shap_explanations:
        raise HTTPException(
            status_code=503, detail="SHAP explanations are disabled in configuration"
        )

    request_id = getattr(request.state, "request_id", None)

    try:
        result = predictor.explain_prediction(customer.model_dump(), top_n=top_n)

        logger.info(
            "Explanation generated", extra={"request_id": request_id, "top_n": top_n}
        )
        return ExplanationResponse(**result)
    except Exception as e:
        logger.exception(
            "Explanation failed", extra={"requst_id": request_id, "error": str(e)}
        )
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    request: Request,
    batch_request: BatchPredictionRequest,
    predictor: ChurnPredictor = Depends(get_predictor),
):
    """
    Predict churn for multiple customers in one request.

    Optimized for batch processing with vectorized operations.
    """
    request_id = getattr(request.state, "request_id", None)

    try:
        df = pd.DataFrame([c.model_dump() for c in batch_request.customers])

        logger.info(
            "Batch prediction started",
            extra={"request_id": request_id, "batch_size": len(df)},
        )
        results_df = predictor.predict_batch(
            df, include_features=batch_request.include_features
        )

        # Compute summary statistics
        mean_probabilities = float(results_df["probability"].astype(float).mean())
        summary = {
            "total": len(results_df),
            "high_risk": int((results_df["risk_level"] == "high").sum()),
            "medium_risk": int((results_df["risk_level"] == "medium").sum()),
            "low_risk": int((results_df["risk_level"] == "low").sum()),
            "mean_probability": mean_probabilities,
        }

        predictions = [
            PredictionResponse(
                prediction=int(row["prediction"]),
                probability=float(row["probability"]),
                risk_level=row["risk_level"],
            )
            for row in results_df.to_dict("records")
        ]

        logger.info(
            "Batch Prediction completed",
            extra={"request_id": request_id, "summary": summary},
        )

        return BatchPredictionResponse(predictions=predictions, summary=summary)
    except Exception as e:
        logger.exception(
            "Batch prediction failed", extra={"request_id": request_id, "error": str(e)}
        )
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


@app.post("/predict_csv", tags=["Prediction"])
async def predict_csv(
    request: Request,
    file: UploadFile = File(...),
    predictor: ChurnPredictor = Depends(get_predictor),
    settings: Settings = Depends(get_app_settings),
):
    """
    Predict churn for customers from CSV file.

    Streams results back as csv for memory efficiency with large files.
    process in configurable chunks to handle files larger than memory.
    """
    request_id = getattr(request.state, "request_id", None)

    # Validate file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        contents = await file.read()
        buffer = io.BytesIO(contents)

        required_cols = list(CustomerInput.model_fields.keys())

        # Chunk size from settings
        chunk_size = getattr(settings, "csv_chunk_size", 2000)

        logger.info(
            "CSV prediction started",
            extra={
                "request_id": request_id,
                "filename": file.filename,
                "chunk_size": chunk_size,
            },
        )

        def generate_predictions():
            """Generator for streaming CSV predictions."""
            header_written = False
            total_processed = 0

            try:
                for chunk in pd.read_csv(buffer, chunksize=chunk_size):
                    # Normalize column names
                    # DUCK TAPE: Preprocessing should have normalized every column
                    # We need to normalize "card type" here because this column was earlier discarded for prediction, but later included.
                    # Fix this
                    chunk = normalize_dataframe(
                        chunk, column_mapping={"Card Type": "CardType"}
                    )

                    missing = set(required_cols) - set(chunk.columns)
                    if missing:
                        error_msg = f"Missing required columns: {sorted(missing)}"
                        logger.error(
                            "CSV validation failed",
                            extra={
                                "request_id": request_id,
                                "error": error_msg,
                            },
                        )
                        yield f"Error: {error_msg}\n"
                        return

                    # Predict
                    results_df = predictor.predict_batch(
                        chunk[required_cols], include_features=True
                    )

                    # Stream chunk to output
                    output = io.StringIO()
                    results_df.to_csv(output, index=False, header=not header_written)
                    header_written = True
                    total_processed += len(chunk)

                    yield output.getvalue()

                logger.info(
                    "CSV prediction completed",
                    extra={
                        "request_id": request_id,
                        "total_processed": total_processed,
                    },
                )

            except Exception as e:
                logger.exception(
                    "Error during CSV processing",
                    extra={"request_id": request_id, "error": str(e)},
                )
                yield f"Error during processing: {str(e)}\n"

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return StreamingResponse(
            generate_predictions(),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=predictions_{timestamp}.csv",
                "X-Request-ID": request_id or "unknown",
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            "CSV prediction setup failed",
            extra={"request_id": request_id, "error": str(e)},
        )
        raise HTTPException(status_code=500, detail=f"CSV prediction failed: {str(e)}")


@app.post("/top_k", response_model=TopKResponse, tags=["Analytics"])
async def top_k(
    request: Request,
    batch_request: BatchPredictionRequest,
    params: TopKRequest,
    predictor: ChurnPredictor = Depends(get_predictor),
):
    """
    Rank customers by churn risk and return top K.

    Useful for targeting retention campaigns at highest-risk customers.
    """
    request_id = getattr(request.state, "request_id", None)

    try:
        df = pd.DataFrame([c.model_dump() for c in batch_request.customers])

        top_k_df = predictor.get_top_k_customers(
            df,
            k=params.k,
            k_percent=params.k_percent,
        )

        customers = [
            {str(k): v for k, v in row.items()} for row in top_k_df.to_dict("records")
        ]

        logger.info(
            "Top-K ranking completed",
            extra={
                "request_id": request_id,
                "total_customers": len(df),
                "top_k": len(top_k_df),
            },
        )

        return TopKResponse(
            customers=customers,
            k=len(top_k_df),
            k_percent=len(top_k_df) / len(df) if len(df) > 0 else 0,
        )
    except Exception as e:
        logger.exception(
            "Top-k computation failed",
            extra={"request_id": request_id, "error": str(e)},
        )
        raise HTTPException(
            status_code=500, detail=f"Top-k computation failed: {str(e)}"
        )


@app.get("/system/status", response_model=SystemStatusResponse, tags=["Health"])
async def system_status(
    request: Request, settings: Settings = Depends(get_app_settings)
):
    """
    Detailed system health and performance metrics.

    Include model metadata, system resources, and configuration.
    For basic health check, use /health instead.

    Status levels:
    - healthy: All systems operational
    - degraded: Operational with warnings (e.g., startup errors, resource constraints)
    - unhealthy: Critical failures (model not loaded, database down)
    """
    predictor_loaded = request.app.state.predictor is not None
    startup_errors = getattr(request.app.state, "startup_errors", [])

    # Detailed status logic
    if not predictor_loaded:
        status = "unhealthy"
    elif startup_errors:
        status = "degraded"
    else:
        status = "healthy"

    # Model status
    predictor_loaded = request.app.state.predictor is not None
    startup_errors = getattr(request.app.state, "startup_errors", [])

    # Model metadata
    loader = get_model_registry()
    cache_key = f"{settings.mlflow_model_name}:Production"
    metadata = loader.get_metadata(cache_key) or {}

    # System resources
    sys_info = {
        "platform": pf.system(),
        "platform_version": pf.version(),
        "python_version": pf.python_version(),
        "cpu_count": psutil.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "memory_percent": psutil.virtual_memory().percent,
    }

    # Overall status
    status: Literal["healthy", "degraded", "unhealthy", "unknown"] = "unknown"
    if not predictor_loaded:
        status = "unhealthy"
    elif startup_errors:
        status = "degraded"
    else:
        status = "healthy"

    return SystemStatusResponse(
        status=status,
        model_loaded=predictor_loaded,
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=__version__,
        metrics=metadata.get("metrics", {}),
        params=metadata.get("params", {}),
        system_info=sys_info,
        startup_errors=startup_errors,
    )
