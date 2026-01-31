"""
Application Streamlit pour la prédiction de prix immobiliers (version démo)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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


def predict_price(input_data):
    """
    Fonction de prédiction simplifiée basée sur des règles
    (En attendant d'avoir un vrai modèle entraîné)
    """
    # Prix de base par arrondissement (€/m²)
    prix_base_arrond = {
        1: 13000, 2: 11000, 3: 10500, 4: 11500, 5: 12000, 6: 14000,
        7: 14500, 8: 15000, 9: 11500, 10: 10000, 11: 10500, 12: 11000,
        13: 9000, 14: 10500, 15: 11500, 16: 14000, 17: 11000, 18: 9500,
        19: 8500, 20: 8000
    }
    
    prix_m2_base = prix_base_arrond.get(input_data['arrondissement'], 10000)
    
    # Ajustements
    # Proximité métro
    if input_data['distance_metro_m'] < 200:
        prix_m2_base *= 1.08
    elif input_data['distance_metro_m'] > 500:
        prix_m2_base *= 0.95
    
    # Equipements
    if input_data['balcon']:
        prix_m2_base *= 1.05
    if input_data['terrasse']:
        prix_m2_base *= 1.08
    if input_data['parking']:
        prix_m2_base *= 1.10
    if input_data['ascenseur']:
        prix_m2_base *= 1.03
    
    # Renovation
    if input_data['renovation_recente']:
        prix_m2_base *= 1.12
    
    # Age du bien
    if input_data['annee_construction'] > 2000:
        prix_m2_base *= 1.08
    elif input_data['annee_construction'] < 1950:
        prix_m2_base *= 0.92
    
    # Etage (bonus pour étages élevés sauf RDC)
    if input_data['etage'] > 2:
        prix_m2_base *= 1.05
    elif input_data['etage'] == 0:
        prix_m2_base *= 0.95
    
    # Dernier étage
    if input_data['etage'] == input_data['nb_etages_immeuble']:
        prix_m2_base *= 1.03
    
    # Prix total
    prix_total = prix_m2_base * input_data['surface_m2']
    
    # Ajustement selon la surface (grandes surfaces = prix/m² plus bas)
    if input_data['surface_m2'] > 100:
        prix_total *= 0.95
    elif input_data['surface_m2'] < 30:
        prix_total *= 1.05
    
    return prix_total


def main():
    # Header
    st.title("🏠 Prédiction de Prix Immobilier à Paris")
    st.markdown("### Estimez le prix de votre bien immobilier grâce à l'IA")
    
    st.info("ℹ️ **Version démo** : Cette application utilise un modèle de prédiction simplifié. Pour de meilleures prédictions, entraînez le modèle complet avec vos données.")
    
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
    st.sidebar.subheader("🗝️ Caractéristiques")
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
            'balcon': balcon,
            'terrasse': terrasse,
            'parking': parking,
            'cave': cave,
            'ascenseur': ascenseur,
            'renovation_recente': renovation,
            'distance_metro_m': distance_metro
        }
        
        # Prédire
        prediction = predict_price(input_data)
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
        
        # Prix moyens par arrondissement
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
            
            *Note: Ces estimations sont basées sur un modèle simplifié. Pour des prédictions plus précises, entraînez le modèle complet avec vos données.*
        """)
        
    else:
        # Page d'accueil
        st.info("👈 Renseignez les caractéristiques de votre bien dans la barre latérale puis cliquez sur 'Prédire le prix'")
        
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
                
                - 🏙️ **Arrondissement** : Impact majeur (variations de 7000€ à 15000€/m²)
                - 📐 **Surface** : Plus c'est grand, plus le prix au m² peut diminuer
                - 🚇 **Proximité métro** : Bonus de +8% si < 200m
                - ✨ **Équipements** : Balcon (+5%), Parking (+10%), Terrasse (+8%)
                - 🗝️ **État** : Rénovation récente (+12%), Construction récente (+8%)
            """)


if __name__ == "__main__":
    main()
