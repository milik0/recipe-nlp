from pydantic import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "FastAPI Recipe API"
    DEBUG: bool = False
    DATABASE_URL: str

    class Config:
        env_file = ".env"


settings = Settings()
