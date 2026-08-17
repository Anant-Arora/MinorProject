from fastapi import FastAPI
from app.database import engine, Base
from app.models import transaction

Base.metadata.create_all(bind=engine)

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}