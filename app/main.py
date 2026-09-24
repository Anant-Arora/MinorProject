from fastapi import FastAPI

from app.routes.nudge import router as nudge_router
from app.routes.upload import router as upload_router


app = FastAPI(
    title="FinSight Backend API",
    description=(
        "Privacy-first AI financial assistant "
        "for the Indian market"
    ),
    version="1.0.0",
)


# Statement parsing, ML analysis, and transaction upload routes.
app.include_router(upload_router)

# Adaptive Nudge Engine routes:
# POST /api/nudge/decide
# POST /api/nudge/events
# GET  /api/nudge/health
app.include_router(nudge_router)


@app.get("/")
def health_check():
    """Basic application health-check endpoint."""

    return {
        "status": "healthy",
        "message": (
            "FinSight ML backend is running successfully!"
        ),
    }