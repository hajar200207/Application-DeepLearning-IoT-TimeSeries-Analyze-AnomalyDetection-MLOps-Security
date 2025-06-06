
Objectif du projet

Ce projet vise à détecter les anomalies sécuritaires (attaques, menaces, virus...) dans des données IoT en utilisant des modèles de Deep Learning appliqués à des séries temporelles, avec intégration des bonnes pratiques MLOps.

Dataset utilisé : IoT_Modbus.csv du corpus ToN_IoT.

Démarrage rapide

Prérequis

Crée un environnement virtuel puis installe les dépendances :

python -m venv venv
source venv/bin/activate  # (Linux/macOS)
venv\Scripts\activate     # (Windows)

 Structure du projet

 .
├── data/                  # Données brutes et prétraitées
├── scripts/               # Scripts d’automatisation, prétraitement, visualisation
├── src/                   # Code modulaire : data, models, features, evaluation, visualisation
├── app/                   # API FastAPI pour l’exposition du modèle
├── models/                # Modèles entraînés
├── notebooks/             # Analyses exploratoires (optionnel)
├── logs/                  # Journaux d’exécution
├── tests/                 # Tests unitaires
├── configs/               # Fichiers de configuration YAML
├── .github/workflows/     # CI/CD GitHub Actions
├── README.md              # Documentation du projet
├── requirements.txt       # Bibliothèques Python


# Installer les dépendances
pip install -r requirements.txt

# Lancer tous les scripts d’un coup
cd scripts
python run_all_steps.py
