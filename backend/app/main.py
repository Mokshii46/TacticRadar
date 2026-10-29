import logging
import sys
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from backend.app.routers import predict, metrics, explain
from backend.app.services.model_service import ModelService

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("backend.app")

app = FastAPI(
    title="TacticRadar API",
    description="ML-Based Multi-Label Chess Tactical Opportunity Detection API",
    version="1.0.0",
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(predict.router)
app.include_router(metrics.router)
app.include_router(explain.router)


@app.on_event("startup")
async def startup_event():
    logger.info("Initializing TacticRadar Backend Services...")
    try:
        # Load models once at startup
        ModelService.get_instance()
        logger.info("TacticRadar models and services initialized successfully.")
    except Exception as e:
        logger.warning(f"Startup model load warning: {e}")


@app.get("/api/health", tags=["Health"])
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "TacticRadar",
        "version": "1.0.0",
    }


# Custom exception handler for validation errors (returns 422)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc.errors()), "body": exc.body},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
