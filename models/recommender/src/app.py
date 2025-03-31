from fastapi import FastAPI
from config import settings

app = FastAPI(title=settings.APP_NAME, debug=settings.DEBUG)

@app.get("/")
def root():
    return {"message": "Welcome to the Recipe API"}
