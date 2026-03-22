import os

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    MODEL_NAME = "llama-3.3-70b-versatile"
    MAX_TOKENS = 1024
    APP_NAME = "CareerAI"
    VERSION = "2.0.0"
    DEBUG = False
    PORT = int(os.environ.get("PORT", 10000))

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": ProductionConfig
}
