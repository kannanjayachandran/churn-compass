import io
import platform as pf
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pandas as pd
import psutil
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from churn_compass import __version__, settings, setup_logger
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
from churn_compass.serving import ChurnPredictor, get_model_registry
from churn_compass.utils.normalization import normalize_dataframe

logger = setup_logger(__name__)


# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Churn Compass API")
    settings.setup()

    try:
        loader = get_model_registry()
        model = loader.get_latest_production_model()
        app.state.predictor = ChurnPredictor(model=model)

        logger.info("Model loaded successfully at startup")
    except Exception:
        logger.exception("Failed to load model during startup")
        app.state.predictor = None

    yield

    logger.info("Shutting down Churn Compass API")
    app.state.predictor = None


app = FastAPI(
    title="Churn Compass API",
    description="customer Churn Prediction API for Retail Banking",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependencies
def get_predictor(request: Request) -> ChurnPredictor:
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable",
        )
    return predictor


# Routes
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Churn Compass API",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(request: Request):
    predictor_loaded = request.app.state.predictor is not None

    return HealthResponse(
        status="healthy" if predictor_loaded else "unhealthy",
        model_loaded=predictor_loaded,
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=__version__,
    )


@app.get("/version", response_model=VersionResponse, tags=["Health"])
async def get_version():
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
    customer: CustomerInput,
    predictor: ChurnPredictor = Depends(get_predictor),
):
    try:
        result = predictor.predict_single(customer.model_dump())
        return PredictionResponse(**result)
    except Exception:
        logger.exception("Single prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/explain", response_model=ExplanationResponse, tags=["Prediction"])
async def explain_prediction(
    customer: CustomerInput,
    top_n: int = 10,
    predictor: ChurnPredictor = Depends(get_predictor),
):
    try:
        result = predictor.explain_prediction(customer.model_dump(), top_n=top_n)
        return ExplanationResponse(**result)
    except Exception:
        logger.exception("Explanation failed")
        raise HTTPException(status_code=500, detail="Explanation failed")


@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    request: BatchPredictionRequest, predictor: ChurnPredictor = Depends(get_predictor)
):
    try:
        df = pd.DataFrame([c.model_dump() for c in request.customers])
        results_df = predictor.predict_batch(
            df, include_features=request.include_features
        )

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

        return BatchPredictionResponse(predictions=predictions, summary=summary)
    except Exception:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail="Batch prediction failed")


@app.post("/predict_csv", tags=["Prediction"])
async def predict_csv(
    file: UploadFile = File(...), predictor: ChurnPredictor = Depends(get_predictor)
):
    try:
        contents = await file.read()
        buffer = io.BytesIO(contents)

        required_cols = list(CustomerInput.model_fields.keys())

        def generate_predictions():
            header_written = False
            try:
                # Process in chunks of 2000 for a balance between overhead and memory
                for chunk in pd.read_csv(buffer, chunksize=2000):
                    # Centralized normalization
                    chunk = normalize_dataframe(
                        chunk, column_mapping={"Card Type": "CardType"}
                    )

                    missing = set(required_cols) - set(chunk.columns)
                    if missing:
                        logger.error(f"Missing columns in CSV chunk: {missing}")
                        yield f"Error: Missing required columns: {sorted(missing)}\n"
                        return

                    results_df = predictor.predict_batch(
                        chunk[required_cols], include_features=True
                    )

                    output = io.StringIO()
                    results_df.to_csv(output, index=False, header=not header_written)
                    header_written = True
                    yield output.getvalue()
            except Exception as e:
                logger.exception("Error during chunked CSV processing")
                yield f"Error during processing: {str(e)}\n"

        return StreamingResponse(
            generate_predictions(),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=predictions_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
            },
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("CSV prediction failed")
        raise HTTPException(status_code=500, detail="CSV prediction failed")


@app.post("/top_k", response_model=TopKResponse, tags=["Analytics"])
async def top_k(
    request: BatchPredictionRequest,
    params: TopKRequest,
    predictor: ChurnPredictor = Depends(get_predictor),
):
    try:
        df = pd.DataFrame([c.model_dump() for c in request.customers])
        top_k_df = predictor.get_top_k_customers(
            df,
            k=params.k,
            k_percent=params.k_percent,
        )

        customers = [
            {str(k): v for k, v in row.items()} for row in top_k_df.to_dict("records")
        ]

        return TopKResponse(
            customers=customers,
            k=len(top_k_df),
            k_percent=len(top_k_df) / len(df),
        )
    except Exception:
        logger.exception("Top-k computation failed")
        raise HTTPException(status_code=500, detail="Top-k computation failed")


@app.get("/system/status", response_model=SystemStatusResponse, tags=["Health"])
async def system_status(request: Request):
    # 1. Base Health
    predictor_loaded = request.app.state.predictor is not None

    # 2. Metadata & Metrics
    loader = get_model_registry()
    cache_key = f"{settings.mlflow_model_name}:Production"
    metadata = loader.get_metadata(cache_key) or {}

    # 3. System Info (Basic)
    sys_info = {
        "platform": pf.system(),
        "cpu_percent": psutil.cpu_percent(),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
    }

    return SystemStatusResponse(
        status="healthy" if predictor_loaded else "unhealthy",
        model_loaded=predictor_loaded,
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=__version__,
        metrics=metadata.get("metrics", {}),
        params=metadata.get("params", {}),
        system_info=sys_info,
    )
