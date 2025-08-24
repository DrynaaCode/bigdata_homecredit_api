# Home Credit Scoring API

## Présentation
Cette API permet de scorer les clients de la banque Home Credit à partir de leurs données, via un modèle de machine learning déployé en production. Elle utilise FastAPI, MongoDB et un modèle sérialisé au format pickle.

## Fonctionnalités principales
- Récupération de la liste des clients avec filtres et pagination
- Récupération des détails d'un client par son identifiant (`/clients/{id}`)
- Prédiction du score de crédit pour un client (`/predict/{id}`)
- Gestion des erreurs et validation des données

## Endpoints

### 1. Liste des clients
`GET /clients`
- Paramètres de filtre : genre, type de contrat, revenus, etc.
- Pagination : `skip`, `limit`
- Réponse : liste paginée des clients

### 2. Détail d'un client
`GET /clients/{id}`
- Paramètre : identifiant client (SK_ID_CURR)
- Réponse : informations détaillées + score de crédit

### 3. Prédiction du score
`GET /predict/{id}`
- Paramètre : identifiant client
- Réponse : score de crédit calculé par le modèle ML

## Lancement de l'API

1. Installer les dépendances :
```bash
pip install -r requierements.txt
```
2. Configurer le fichier `.env` (MongoDB, chemin du modèle, etc.)
3. Lancer le serveur :
```bash
uvicorn app.main:app --reload
```

## Structure du projet
```
api/
  app/
    main.py
    routers/
    services/
    schemas/
    utils/
  models_artifacts/
    model.pkl
  data/
    application_train.csv
    application_test.csv
  .env
```

## Exemple d'appel
```bash
curl http://127.0.0.1:8000/clients/100002
curl http://127.0.0.1:8000/predict/100002
```

## Dépendances principales
- fastapi
- uvicorn
- motor
- scikit-learn
- pandas
- numpy
- python-dotenv

## Notes
- Le modèle doit être entraîné et sauvegardé dans `models_artifacts/model.pkl`.
- Les features du client doivent correspondre à celles utilisées lors de l'entraînement.
- Pour toute question, voir le code ou contacter l'équipe projet.
