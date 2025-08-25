
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from app.middleware.error_handlers import global_exception_handler, custom_http_exception_handler
from pydantic import BaseModel, Field
from bson import ObjectId
from typing import List, Optional
import uvicorn

from app.services.database import connect_to_mongo, close_mongo_connection, get_database

# Créer l'instance FastAPI

app = FastAPI(title="Mon API Test")


# Enregistre les gestionnaires d'erreurs du dossier middleware
app.add_exception_handler(Exception, global_exception_handler)
from starlette.exceptions import HTTPException as StarletteHTTPException
app.add_exception_handler(StarletteHTTPException, custom_http_exception_handler)

# Import et inclusion du router test
from app.routers import test, predict, clients_joblib
app.include_router(test.router)
app.include_router(predict.router)
app.include_router(clients_joblib.router)

# Événement de démarrage
@app.on_event("startup")
async def startup_event():
    await connect_to_mongo()

# Événement d'arrêt
@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo_connection()

# Routes basiques
@app.get("/")
async def accueil():
    """Page d'accueil"""
    return {"message": "Hello ! Mon API fonctionne !"}

# Pour lancer le serveur
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)