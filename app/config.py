import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

class Settings:
    """Configuration de l'application"""
    
    # MongoDB
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    DATABASE_NAME: str = os.getenv("DATABASE_NAME", "mon_api_db")
    
    # API
    API_TITLE: str = "API FastAPI avec MongoDB"
    API_DESCRIPTION: str = "API REST avec base de données MongoDB"
    API_VERSION: str = "1.0.0"
    
    # Serveur
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", 8000))
    
    # Pagination
    DEFAULT_SKIP: int = 0
    DEFAULT_LIMIT: int = 100
    MAX_LIMIT: int = 500
    
    # Debug
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"

# Instance globale des paramètres
settings = Settings()