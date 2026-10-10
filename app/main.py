"""
app/main.py
FastAPI production application entrypoint with lifespan lifecycle, middleware, error handling, and routing.
"""

import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1.endpoints import router as api_v1_router
from app.core.config import settings
from app.core.logging import setup_logging
from app.services.fraud_service import fraud_service

# Initialize structured logging
logger = setup_logging(
    log_level=settings.log_level,
    structured=settings.structured_logging
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifecycle management: Load ML model on startup, clean up on shutdown.
    """
    logger.info(
        f"Starting {settings.app_name} v{settings.app_version} "
        f"[Environment: {settings.app_env}]"
    )
    logger.info(f"Target MLflow Tracking URI: {settings.mlflow_tracking_uri}")
    logger.info(f"Target MLflow Model: {settings.model_name} (version: {settings.model_version})")

    # Load ML Model into singleton service
    model_loaded = fraud_service.load_model()
    if model_loaded:
        logger.info(f"Model successfully loaded via: {fraud_service.model_source}")
    else:
        logger.warning(
            "Model failed to load during startup. Will attempt fallback on demand."
        )

    yield

    logger.info("Application shutting down. Releasing resources...")


# Instantiate FastAPI Application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Production-grade Fraud Detection API serving MLflow registered models "
        "with real-time risk scoring, latency tracking, and robust offline fallback."
    ),
    docs_url=settings.docs_url,
    redoc_url=settings.redoc_url,
    openapi_url=settings.openapi_url,
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request Logging & Latency Middleware
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.perf_counter()
    client_ip = request.client.host if request.client else "unknown"

    # Process request
    try:
        response = await call_next(request)
        process_time = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = str(process_time)

        # Log request summary
        logger.info(
            f"{request.method} {request.url.path} -> {response.status_code} ({process_time}ms)",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": process_time,
                "client_ip": client_ip,
            }
        )
        return response
    except Exception as exc:
        process_time = round((time.perf_counter() - start_time) * 1000, 2)
        logger.error(
            f"Unhandled exception on {request.method} {request.url.path}: {exc}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "duration_ms": process_time,
                "client_ip": client_ip,
            },
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "detail": str(exc),
                "request_id": request_id,
            },
            headers={"X-Request-ID": request_id, "X-Process-Time-Ms": str(process_time)},
        )


# Global Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Format Pydantic validation errors consistently."""
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error.get("loc", []))
        errors.append({
            "field": field,
            "message": error.get("msg"),
            "type": error.get("type"),
        })

    logger.warning(
        f"Validation error on {request.method} {request.url.path}: {errors}",
        extra={"path": request.url.path, "method": request.method}
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Unprocessable Entity",
            "message": "Input validation failed. Please check the supplied parameters.",
            "details": errors,
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Format HTTP exceptions into structured JSON."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "status_code": exc.status_code,
            "detail": exc.detail,
        },
    )


# Root landing endpoint
@app.get("/", tags=["General"])
async def root():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.app_env,
        "docs": f"{settings.docs_url}",
        "health": f"{settings.api_v1_str}/health",
        "predict": f"{settings.api_v1_str}/predict",
    }


# Include Versioned API Routes
app.include_router(
    api_v1_router,
    prefix=settings.api_v1_str,
    tags=["Fraud Prediction"],
)
