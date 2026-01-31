"""
Module d'entraînement et d'évaluation des modèles de prédiction
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

from data_processing import (
    load_data, 
    create_features, 
    prepare_train_test,
    get_feature_importance_df
)


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Évalue un modèle sur le jeu de test
    
    Args:
        model: Modèle entraîné
        X_test: Features de test
        y_test: Target de test
        model_name (str): Nom du modèle pour l'affichage
        
    Returns:
        dict: Métriques de performance
    """
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"📊 {model_name}")
    print(f"{'='*50}")
    print(f"R² Score:  {r2:.4f}")
    print(f"RMSE:      {rmse:,.0f} €")
    print(f"MAE:       {mae:,.0f} €")
    
    return {
        'model_name': model_name,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'predictions': y_pred
    }


def train_models(X_train, y_train, X_test, y_test):
    """
    Entraîne plusieurs modèles et compare leurs performances
    
    Args:
        X_train, y_train: Données d'entraînement
        X_test, y_test: Données de test
        
    Returns:
        dict: Résultats de tous les modèles
    """
    print("🚀 Entraînement des modèles...\n")
    
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        ),
        'Ridge Regression': Ridge(
            alpha=10.0,
            random_state=42
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"⏳ Entraînement de {name}...")
        model.fit(X_train, y_train)
        results[name] = evaluate_model(model, X_test, y_test, name)
        results[name]['model'] = model
    
    return results


def plot_results(results, y_test, save_path='visualizations/'):
    """
    Crée des visualisations des résultats
    
    Args:
        results (dict): Résultats des modèles
        y_test: Vraies valeurs
        save_path (str): Dossier de sauvegarde
    """
    # Comparaison des performances
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    models = list(results.keys())
    metrics = {
        'R² Score': [results[m]['r2'] for m in models],
        'RMSE': [results[m]['rmse'] for m in models],
        'MAE': [results[m]['mae'] for m in models]
    }
    
    for idx, (metric_name, values) in enumerate(metrics.items()):
        axes[idx].bar(models, values, color='steelblue', alpha=0.7)
        axes[idx].set_title(f'{metric_name} par modèle', fontsize=14, fontweight='bold')
        axes[idx].set_ylabel(metric_name)
        axes[idx].tick_params(axis='x', rotation=45)
        
        # Ajouter les valeurs sur les barres
        for i, v in enumerate(values):
            if metric_name == 'R² Score':
                axes[idx].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
            else:
                axes[idx].text(i, v + 2000, f'{v:,.0f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Graphique sauvegardé: {save_path}model_comparison.png")
    plt.close()
    
    # Prédictions vs Réel pour le meilleur modèle
    best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
    best_predictions = results[best_model_name]['predictions']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(y_test, best_predictions, alpha=0.5, s=30)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
            'r--', lw=2, label='Prédiction parfaite')
    
    ax.set_xlabel('Prix Réel (€)', fontsize=12)
    ax.set_ylabel('Prix Prédit (€)', fontsize=12)
    ax.set_title(f'Prédictions vs Réel - {best_model_name}\nR² = {results[best_model_name]["r2"]:.3f}',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    print(f"✅ Graphique sauvegardé: {save_path}predictions_vs_actual.png")
    plt.close()


def plot_feature_importance(model, feature_names, save_path='visualizations/', top_n=15):
    """
    Affiche l'importance des features
    
    Args:
        model: Modèle entraîné
        feature_names (list): Noms des features
        save_path (str): Dossier de sauvegarde
        top_n (int): Nombre de top features à afficher
    """
    importance_df = get_feature_importance_df(model, feature_names)
    
    plt.figure(figsize=(10, 8))
    
    top_features = importance_df.head(top_n)
    
    plt.barh(range(len(top_features)), top_features['importance'], color='coral')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top {top_n} Features les plus importantes', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"✅ Graphique sauvegardé: {save_path}feature_importance.png")
    plt.close()
    
    return importance_df


def save_best_model(results, feature_names, save_path='models/'):
    """
    Sauvegarde le meilleur modèle
    
    Args:
        results (dict): Résultats des modèles
        feature_names (list): Noms des features
        save_path (str): Dossier de sauvegarde
        
    Returns:
        str: Nom du meilleur modèle
    """
    best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
    best_model = results[best_model_name]['model']
    
    # Sauvegarder le modèle et les métadonnées
    model_data = {
        'model': best_model,
        'feature_names': feature_names,
        'model_name': best_model_name,
        'r2_score': results[best_model_name]['r2'],
        'rmse': results[best_model_name]['rmse'],
        'mae': results[best_model_name]['mae']
    }
    
    joblib.dump(model_data, f'{save_path}best_model.pkl')
    print(f"\n✅ Meilleur modèle sauvegardé: {best_model_name}")
    print(f"   R² = {results[best_model_name]['r2']:.4f}")
    print(f"   Fichier: {save_path}best_model.pkl")
    
    return best_model_name


if __name__ == "__main__":
    print("="*60)
    print("🏠 ENTRAÎNEMENT DU MODÈLE DE PRÉDICTION IMMOBILIÈRE")
    print("="*60)
    
    # 1. Charger et préparer les données
    print("\n📂 Étape 1: Chargement des données")
    df = load_data()
    
    print("\n🔧 Étape 2: Feature Engineering")
    df = create_features(df)
    
    print("\n✂️ Étape 3: Préparation train/test")
    X_train, X_test, y_train, y_test, feature_names, scaler = prepare_train_test(df)
    
    # 2. Entraîner les modèles
    print("\n" + "="*60)
    print("🤖 Étape 4: Entraînement des modèles")
    print("="*60)
    results = train_models(X_train, y_train, X_test, y_test)
    
    # 3. Visualisations
    print("\n📊 Étape 5: Génération des visualisations")
    plot_results(results, y_test)
    
    # 4. Feature importance (meilleur modèle)
    best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
    best_model = results[best_model_name]['model']
    
    if hasattr(best_model, 'feature_importances_'):
        print("\n🎯 Étape 6: Analyse de l'importance des features")
        importance_df = plot_feature_importance(best_model, feature_names)
        print("\nTop 10 features:")
        print(importance_df.head(10))
    
    # 5. Sauvegarder le meilleur modèle
    print("\n💾 Étape 7: Sauvegarde du modèle")
    save_best_model(results, feature_names)
    
    print("\n" + "="*60)
    print("✨ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print("="*60)
