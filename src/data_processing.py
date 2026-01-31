"""
Module de traitement des données immobilières
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filepath='data/immobilier_paris.csv'):
    """
    Charge les données immobilières depuis un fichier CSV
    
    Args:
        filepath (str): Chemin vers le fichier CSV
        
    Returns:
        pd.DataFrame: DataFrame contenant les données
    """
    df = pd.read_csv(filepath)
    print(f"✅ Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


def create_features(df):
    """
    Crée des features supplémentaires pour améliorer la prédiction
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame avec features supplémentaires
    """
    df = df.copy()
    
    # Prix au m² (pour analyse, pas utilisé dans le modèle)
    df['prix_m2'] = df['prix'] / df['surface_m2']
    
    # Ratio surface/pièces
    df['surface_par_piece'] = df['surface_m2'] / df['nb_pieces']
    
    # Dernier étage (boolean)
    df['dernier_etage'] = (df['etage'] == df['nb_etages_immeuble']).astype(int)
    
    # Ancien vs moderne
    df['est_ancien'] = (df['annee_construction'] < 1950).astype(int)
    df['est_recent'] = (df['annee_construction'] > 2000).astype(int)
    
    # Catégorie d'arrondissement par prix
    arrond_cher = [1, 6, 7, 8, 16]
    arrond_milieu = [2, 3, 4, 5, 9, 10, 11, 12, 14, 15, 17]
    
    df['arrond_prestige'] = df['arrondissement'].isin(arrond_cher).astype(int)
    df['arrond_populaire'] = df['arrondissement'].isin([13, 18, 19, 20]).astype(int)
    
    # Proximité métro (catégories)
    df['tres_proche_metro'] = (df['distance_metro_m'] < 200).astype(int)
    df['loin_metro'] = (df['distance_metro_m'] > 500).astype(int)
    
    # Score confort (combinaison des équipements)
    df['score_confort'] = (
        df['balcon'] + 
        df['terrasse'] * 2 + 
        df['parking'] * 1.5 + 
        df['cave'] * 0.5 + 
        df['ascenseur']
    )
    
    print(f"✅ Features créées: {len(df.columns)} colonnes totales")
    
    return df


def prepare_train_test(df, test_size=0.2, random_state=42):
    """
    Prépare les données pour l'entraînement et le test
    
    Args:
        df (pd.DataFrame): DataFrame avec toutes les features
        test_size (float): Proportion du jeu de test
        random_state (int): Graine aléatoire pour reproductibilité
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, feature_names, scaler
    """
    # Séparer features et target
    # On retire 'prix' et 'prix_m2' (qui est calculé à partir du prix)
    features_to_drop = ['prix', 'prix_m2']
    
    X = df.drop(columns=features_to_drop)
    y = df['prix']
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Standardisation (pour certains modèles)
    scaler = StandardScaler()
    
    print(f"✅ Train set: {X_train.shape[0]} échantillons")
    print(f"✅ Test set: {X_test.shape[0]} échantillons")
    print(f"✅ Nombre de features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test, X.columns.tolist(), scaler


def get_feature_importance_df(model, feature_names):
    """
    Extrait l'importance des features d'un modèle
    
    Args:
        model: Modèle entraîné (doit avoir feature_importances_)
        feature_names (list): Liste des noms de features
        
    Returns:
        pd.DataFrame: DataFrame trié par importance
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance_df


if __name__ == "__main__":
    # Test du module
    print("🧪 Test du module de traitement des données\n")
    
    df = load_data()
    print(f"\n📊 Aperçu des données:")
    print(df.head())
    
    df_features = create_features(df)
    print(f"\n📊 Nouvelles features:")
    print(df_features[['surface_par_piece', 'dernier_etage', 'score_confort']].head())
    
    X_train, X_test, y_train, y_test, features, scaler = prepare_train_test(df_features)
    print(f"\n✨ Données prêtes pour l'entraînement!")
