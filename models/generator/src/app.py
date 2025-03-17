from fastapi import FastAPI
from app.config import settings
from app.routes import recipe

app = FastAPI(title=settings.APP_NAME, debug=settings.DEBUG)

# Include Routes
app.include_router(recipe.router)

@app.get("/")
def root():
    return {"message": "Welcome to the Recipe API"}
