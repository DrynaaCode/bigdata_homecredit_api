from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings

class MongoDB:
    """Gestionnaire de connexion MongoDB"""
    
    client: AsyncIOMotorClient = None
    database = None

# Instance globale
mongodb = MongoDB()

async def connect_to_mongo():
    """Établir la connexion à MongoDB"""
    try:
        mongodb.client = AsyncIOMotorClient(settings.MONGODB_URL)
        mongodb.database = mongodb.client[settings.DATABASE_NAME]
        
        # Test de connexion
        await mongodb.client.admin.command('ismaster')
        print(f"✅ Connecté à MongoDB: {settings.DATABASE_NAME}")
        
    except Exception as e:
        print(f"❌ Erreur de connexion MongoDB: {e}")
        raise e

async def close_mongo_connection():
    """Fermer la connexion MongoDB"""
    if mongodb.client:
        mongodb.client.close()
        print("🔌 Connexion MongoDB fermée")

async def get_database():
    """Récupérer la base de données"""
    return mongodb.database