"""
Application Streamlit pour la prédiction de prix immobiliers
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Prix Immobilier Paris",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Charge le modèle entraîné"""
    try:
        model_path = Path('models/best_model.pkl')
        if model_path.exists():
            model_data = joblib.load(model_path)
            return model_data
        else:
            st.error("❌ Modèle non trouvé. Veuillez d'abord entraîner le modèle avec `python src/model.py`")
            return None
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {str(e)}")
        return None


def create_features_from_input(input_data):
    """
    Crée toutes les features nécessaires à partir des inputs utilisateur
    
    Args:
        input_data (dict): Dictionnaire des inputs utilisateur
        
    Returns:
        pd.DataFrame: DataFrame avec toutes les features
    """
    df = pd.DataFrame([input_data])
    
    # Features de base (déjà dans input_data)
    # ...
    
    # Features dérivées (même logique que dans data_processing.py)
    df['surface_par_piece'] = df['surface_m2'] / df['nb_pieces']
    df['dernier_etage'] = (df['etage'] == df['nb_etages_immeuble']).astype(int)
    df['est_ancien'] = (df['annee_construction'] < 1950).astype(int)
    df['est_recent'] = (df['annee_construction'] > 2000).astype(int)
    
    arrond_cher = [1, 6, 7, 8, 16]
    df['arrond_prestige'] = df['arrondissement'].isin(arrond_cher).astype(int)
    df['arrond_populaire'] = df['arrondissement'].isin([13, 18, 19, 20]).astype(int)
    
    df['tres_proche_metro'] = (df['distance_metro_m'] < 200).astype(int)
    df['loin_metro'] = (df['distance_metro_m'] > 500).astype(int)
    
    df['score_confort'] = (
        df['balcon'] + 
        df['terrasse'] * 2 + 
        df['parking'] * 1.5 + 
        df['cave'] * 0.5 + 
        df['ascenseur']
    )
    
    return df


def main():
    # Header
    st.title("🏠 Prédiction de Prix Immobilier à Paris")
    st.markdown("### Estimez le prix de votre bien immobilier grâce à l'IA")
    
    # Charger le modèle
    model_data = load_model()
    
    if model_data is None:
        st.stop()
    
    model = model_data['model']
    feature_names = model_data['feature_names']
    
    # Afficher les infos du modèle
    with st.expander("ℹ️ Informations sur le modèle"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Modèle", model_data['model_name'])
        col2.metric("R² Score", f"{model_data['r2_score']:.3f}")
        col3.metric("Erreur moyenne", f"{model_data['mae']:,.0f} €")
    
    # Sidebar pour les inputs
    st.sidebar.header("🔧 Caractéristiques du bien")
    
    # Section 1: Localisation
    st.sidebar.subheader("📍 Localisation")
    arrondissement = st.sidebar.selectbox(
        "Arrondissement",
        options=list(range(1, 21)),
        index=6,
        help="L'arrondissement influence fortement le prix"
    )
    
    distance_metro = st.sidebar.slider(
        "Distance au métro (m)",
        min_value=50,
        max_value=1000,
        value=250,
        step=50
    )
    
    # Section 2: Caractéristiques principales
    st.sidebar.subheader("🏗️ Caractéristiques")
    surface = st.sidebar.number_input(
        "Surface (m²)",
        min_value=15,
        max_value=300,
        value=60,
        step=5
    )
    
    nb_pieces = st.sidebar.slider(
        "Nombre de pièces",
        min_value=1,
        max_value=6,
        value=3
    )
    
    nb_chambres = st.sidebar.slider(
        "Nombre de chambres",
        min_value=0,
        max_value=5,
        value=max(0, nb_pieces - 1)
    )
    
    etage = st.sidebar.slider(
        "Étage",
        min_value=0,
        max_value=10,
        value=2,
        help="0 = Rez-de-chaussée"
    )
    
    nb_etages_immeuble = st.sidebar.slider(
        "Étages dans l'immeuble",
        min_value=max(1, etage + 1),
        max_value=15,
        value=max(6, etage + 2)
    )
    
    annee_construction = st.sidebar.slider(
        "Année de construction",
        min_value=1850,
        max_value=2024,
        value=1970
    )
    
    # Section 3: Équipements
    st.sidebar.subheader("✨ Équipements")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        balcon = st.checkbox("Balcon", value=True)
        parking = st.checkbox("Parking", value=False)
        cave = st.checkbox("Cave", value=True)
    
    with col2:
        terrasse = st.checkbox("Terrasse", value=False)
        ascenseur = st.checkbox("Ascenseur", value=True)
        renovation = st.checkbox("Rénové", value=False)
    
    # Bouton de prédiction
    predict_button = st.sidebar.button("🔮 Prédire le prix", type="primary")
    
    # Main content
    if predict_button:
        # Préparer les données
        input_data = {
            'surface_m2': surface,
            'nb_pieces': nb_pieces,
            'nb_chambres': nb_chambres,
            'arrondissement': arrondissement,
            'etage': etage,
            'nb_etages_immeuble': nb_etages_immeuble,
            'annee_construction': annee_construction,
            'balcon': int(balcon),
            'terrasse': int(terrasse),
            'parking': int(parking),
            'cave': int(cave),
            'ascenseur': int(ascenseur),
            'renovation_recente': int(renovation),
            'distance_metro_m': distance_metro
        }
        
        # Créer toutes les features
        df_features = create_features_from_input(input_data)
        
        # Prédire
        prediction = model.predict(df_features)[0]
        prix_m2 = prediction / surface
        
        # Afficher la prédiction
        st.markdown(f"""
            <div class="prediction-box">
                <h1>💰 Prix estimé</h1>
                <h1 style="font-size: 3rem; margin: 1rem 0;">{prediction:,.0f} €</h1>
                <p style="font-size: 1.2rem;">{prix_m2:,.0f} € / m²</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Métriques détaillées
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <h4>📐 Surface</h4>
                    <h2>{surface} m²</h2>
                    <p>{nb_pieces} pièces</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <h4>📍 Localisation</h4>
                    <h2>{arrondissement}ème</h2>
                    <p>{distance_metro}m du métro</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            equipements = sum([balcon, terrasse, parking, cave, ascenseur, renovation])
            st.markdown(f"""
                <div class="metric-card">
                    <h4>✨ Confort</h4>
                    <h2>{equipements}/6</h2>
                    <p>équipements</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Graphique de comparaison avec le marché
        st.subheader("📊 Comparaison avec le marché")
        
        # Prix moyens par arrondissement (approximatif)
        prix_moyens_arrond = {
            1: 13000, 2: 11000, 3: 10500, 4: 11500, 5: 12000, 6: 14000,
            7: 14500, 8: 15000, 9: 11500, 10: 10000, 11: 10500, 12: 11000,
            13: 9000, 14: 10500, 15: 11500, 16: 14000, 17: 11000, 18: 9500,
            19: 8500, 20: 8000
        }
        
        prix_moyen_arrond = prix_moyens_arrond.get(arrondissement, 10000)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['Prix moyen du marché', 'Votre estimation'],
            y=[prix_moyen_arrond, prix_m2],
            marker_color=['lightblue', 'coral'],
            text=[f"{prix_moyen_arrond:,.0f} €/m²", f"{prix_m2:,.0f} €/m²"],
            textposition='auto',
        ))
        
        fig.update_layout(
            title=f"Prix au m² - {arrondissement}ème arrondissement",
            yaxis_title="Prix au m² (€)",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Fourchette de prix
        st.subheader("📈 Fourchette de prix")
        st.info(f"""
            **Estimation basse:** {prediction * 0.90:,.0f} €  
            **Estimation moyenne:** {prediction:,.0f} €  
            **Estimation haute:** {prediction * 1.10:,.0f} €
            
            *Note: Ces estimations sont basées sur un modèle statistique et peuvent varier selon l'état exact du bien.*
        """)
        
    else:
        # Page d'accueil
        st.info("👈 Renseignez les caractéristiques de votre bien dans la barre latérale puis cliquez sur 'Prédire le prix'")
        
        # Statistiques du modèle
        st.subheader("📊 Performance du modèle")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Précision (R²)", f"{model_data['r2_score']:.1%}", 
                   help="Pourcentage de variance expliquée par le modèle")
        col2.metric("Erreur moyenne", f"{model_data['mae']:,.0f} €",
                   help="Erreur moyenne absolue sur les prédictions")
        col3.metric("RMSE", f"{model_data['rmse']:,.0f} €",
                   help="Racine de l'erreur quadratique moyenne")
        
        st.markdown("---")
        
        # Guide d'utilisation
        with st.expander("📖 Comment utiliser cette application ?"):
            st.markdown("""
                ### Instructions
                
                1. **Renseignez la localisation** : Choisissez l'arrondissement et la distance au métro
                2. **Caractéristiques du bien** : Surface, nombre de pièces, étage, année de construction
                3. **Équipements** : Cochez les équipements présents (balcon, parking, etc.)
                4. **Cliquez sur "Prédire le prix"** pour obtenir l'estimation
                
                ### Facteurs influençant le prix
                
                - 🏘️ **Arrondissement** : Impact majeur (variations de 7000€ à 15000€/m²)
                - 📏 **Surface** : Plus c'est grand, plus le prix au m² peut diminuer
                - 🚇 **Proximité métro** : Bonus de +8% si < 200m
                - ✨ **Équipements** : Balcon (+8%), Parking (+10%), Terrasse (+12%)
                - 🏗️ **État** : Rénovation récente (+12%), Construction récente (+8%)
            """)


if __name__ == "__main__":
    main()
