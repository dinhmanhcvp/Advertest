"""Prometheus Metrics Middleware for AdverTest.

Instruments the FastAPI application to expose ML and business-level metrics
for Enterprise Observability.

Requirements:
    pip install prometheus-client
"""

import time
from typing import Callable

from fastapi import FastAPI, Request, Response
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
except ImportError:
    raise ImportError("prometheus_client is required: pip install prometheus-client")

# ─────────────────────────────────────────────────────────
# Define Prometheus Metrics
# ─────────────────────────────────────────────────────────

# Histogram: Track latency of the Attack Engine (frame processing time)
ATTACK_ENGINE_LATENCY = Histogram(
    "advertest_attack_engine_latency_seconds",
    "Processing time per frame in the Physics Attack Engine",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)

# Counter: Track successful vs rejected frames by the Constraint Gate
GENERATION_STATUS_COUNTER = Counter(
    "advertest_generation_status_total",
    "Total count of adversarial frames generated vs rejected",
    ["status"] # e.g., 'success', 'rejected_bbox_out_of_bounds'
)

# Counter: Standard HTTP request metrics
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests handled by the API Gateway",
    ["method", "endpoint", "http_status"]
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"]
)

def setup_metrics(app: FastAPI) -> None:
    """Attach Prometheus middleware and /metrics endpoint to the FastAPI app."""
    
    @app.middleware("http")
    async def prometheus_middleware(request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        process_time = time.time() - start_time
        
        # Record HTTP metrics
        path = request.url.path
        if path != "/metrics":
            HTTP_REQUESTS_TOTAL.labels(
                method=request.method,
                endpoint=path,
                http_status=response.status_code
            ).inc()
            
            HTTP_REQUEST_DURATION.labels(
                method=request.method,
                endpoint=path
            ).observe(process_time)
            
        return response

    @app.get("/metrics", tags=["Observability"])
    async def metrics():
        """Expose metrics for Prometheus scraping."""
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ─────────────────────────────────────────────────────────
# Helper decorators for ML components
# ─────────────────────────────────────────────────────────

def track_attack_latency():
    """Decorator to automatically wrap Attack Engine functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with ATTACK_ENGINE_LATENCY.time():
                return func(*args, **kwargs)
        return wrapper
    return decorator
