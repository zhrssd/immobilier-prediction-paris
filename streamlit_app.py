import sys
from pathlib import Path

# Ajouter le dossier src au Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Importer l'application depuis src/app.py
from app import *
