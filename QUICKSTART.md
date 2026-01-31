# 🚀 Quick Start

## 📦 Installation rapide

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Entraîner le modèle
python src/model.py

# 3. Lancer l'application
streamlit run src/app.py
```

## 📁 Structure du projet

```
immobilier-prediction-project/
│
├── 📊 data/
│   └── immobilier_paris.csv        # Dataset (5000 annonces)
│
├── 📓 notebooks/
│   └── 01_exploration.ipynb        # Analyse exploratoire
│
├── 💻 src/
│   ├── data_processing.py          # Traitement des données
│   ├── model.py                    # Entraînement du modèle
│   └── app.py                      # Application Streamlit
│
├── 🤖 models/
│   └── best_model.pkl              # Modèle entraîné (après python src/model.py)
│
├── 📈 visualizations/               # Graphiques générés
│
├── 📄 README.md                     # Documentation principale
├── 📋 requirements.txt              # Dépendances Python
├── 📚 GUIDE_GITHUB.md              # Guide pour mettre sur GitHub
└── 🚀 QUICKSTART.md                # Ce fichier
```

## ✨ Ce que tu peux faire

### 1. Explorer les données
```bash
jupyter notebook notebooks/01_exploration.ipynb
```

### 2. Entraîner le modèle
```bash
python src/model.py
```
Cela va :
- Charger les données
- Faire du feature engineering
- Entraîner Random Forest
- Générer des visualisations
- Sauvegarder le meilleur modèle

### 3. Utiliser l'app interactive
```bash
streamlit run src/app.py
```
Interface web pour prédire les prix en temps réel !

### 4. Tester le module de traitement
```bash
python src/data_processing.py
```

## 📊 Résultats attendus

Après entraînement, tu devrais obtenir :
- **R² Score**: ~0.90-0.95 (très bon !)
- **RMSE**: ~100,000€
- **MAE**: ~70,000€

## 🎯 Mettre sur GitHub

Suis le guide détaillé dans `GUIDE_GITHUB.md` ou rapidement :

```bash
git init
git add .
git commit -m "🎉 Initial commit: Projet prédiction immobilier"
git remote add origin https://github.com/TON-USERNAME/immobilier-prediction-paris.git
git push -u origin main
```

## 🔧 Personnalisation

1. **README.md** : Change les infos de contact
2. **app.py** : Ajuste les couleurs/style
3. **model.py** : Teste d'autres algorithmes
4. **data** : Ajoute tes propres données

## ❓ Problèmes ?

- **Erreur de packages** : `pip install -r requirements.txt`
- **Données manquantes** : Vérifie que `data/immobilier_paris.csv` existe
- **Modèle non trouvé** : Lance `python src/model.py` d'abord

## 🎓 Pour améliorer le projet

- [ ] Ajouter d'autres algorithmes (XGBoost, LightGBM)
- [ ] Optimiser les hyperparamètres (GridSearch)
- [ ] Déployer l'app sur Streamlit Cloud
- [ ] Ajouter une API REST avec FastAPI
- [ ] Scraper des données réelles

---

**Bon coding ! 🚀**
