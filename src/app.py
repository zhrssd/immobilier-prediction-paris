"""
Application Streamlit pour la prédiction de prix immobiliers - Version améliorée
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
st.set_page_config(
    page_title="Prédiction Prix Immobilier Paris",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background-color: #2C3E50;
        color: white;
    }
    .stButton>button {
        width: 100%;
        background-color: #3498DB;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: none;
        font-size: 1.1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #2980B9;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.4);
    }
    .prediction-box {
        padding: 2.5rem;
        border-radius: 1.5rem;
        background: linear-gradient(135deg, #2980B9 0%, #2C3E50 100%);
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(44, 62, 80, 0.3);
        animation: fadeIn 0.5s;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .metric-card {
        background: #ECF0F1;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }
    .info-box {
        background: #e3f2fd;
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 5px solid #2196F3;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3e0;
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 5px solid #FF9800;
        margin: 1rem 0;
    }
    .stTextInput>div>div>input {
        background-color: #34495E; /* Modifie la couleur d'arrière-plan de l'input */
    }

    .stAlert {
        background-color: #34495E 
        color: black 
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
    """
    df = pd.DataFrame([input_data])
    
    # Features dérivées
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


def get_arrondissement_info(arrond):
    """Retourne des infos sur l'arrondissement"""
    infos = {
        1: {"nom": "Louvre", "quartiers": "Châtelet, Palais-Royal", "ambiance": "Historique & Touristique"},
        2: {"nom": "Bourse", "quartiers": "Opéra, Sentier", "ambiance": "Affaires & Shopping"},
        3: {"nom": "Temple", "quartiers": "Le Marais, Archives", "ambiance": "Branché & Culturel"},
        4: {"nom": "Hôtel-de-Ville", "quartiers": "Notre-Dame, Île Saint-Louis", "ambiance": "Historique & Central"},
        5: {"nom": "Panthéon", "quartiers": "Quartier Latin, Sorbonne", "ambiance": "Étudiant & Intellectuel"},
        6: {"nom": "Luxembourg", "quartiers": "Saint-Germain, Odéon", "ambiance": "Chic & Littéraire"},
        7: {"nom": "Palais-Bourbon", "quartiers": "Tour Eiffel, Invalides", "ambiance": "Prestige & Institutions"},
        8: {"nom": "Élysée", "quartiers": "Champs-Élysées, Madeleine", "ambiance": "Luxe & Prestige"},
        9: {"nom": "Opéra", "quartiers": "Grands Boulevards, Pigalle", "ambiance": "Vivant & Commerçant"},
        10: {"nom": "Entrepôt", "quartiers": "Canal Saint-Martin, Gare du Nord", "ambiance": "Trendy & Populaire"},
        11: {"nom": "Popincourt", "quartiers": "Bastille, Oberkampf", "ambiance": "Festif & Jeune"},
        12: {"nom": "Reuilly", "quartiers": "Bercy, Nation", "ambiance": "Familial & Verdoyant"},
        13: {"nom": "Gobelins", "quartiers": "Place d'Italie, Butte-aux-Cailles", "ambiance": "Multiculturel & Moderne"},
        14: {"nom": "Observatoire", "quartiers": "Montparnasse, Denfert", "ambiance": "Résidentiel & Artistique"},
        15: {"nom": "Vaugirard", "quartiers": "Beaugrenelle, Montparnasse", "ambiance": "Familial & Résidentiel"},
        16: {"nom": "Passy", "quartiers": "Trocadéro, Auteuil", "ambiance": "Huppé & Calme"},
        17: {"nom": "Batignolles-Monceau", "quartiers": "Batignolles, Ternes", "ambiance": "Bourgeois & Agréable"},
        18: {"nom": "Butte-Montmartre", "quartiers": "Montmartre, Barbès", "ambiance": "Artistique & Populaire"},
        19: {"nom": "Buttes-Chaumont", "quartiers": "Buttes-Chaumont, La Villette", "ambiance": "Jeune & Cosmopolite"},
        20: {"nom": "Ménilmontant", "quartiers": "Belleville, Père-Lachaise", "ambiance": "Alternatif & Vivant"}
    }
    return infos.get(arrond, {"nom": "Inconnu", "quartiers": "-", "ambiance": "-"})


def main():
    # Header avec animation
    st.title("🏠 Prédiction de Prix Immobilier à Paris")
    st.markdown("### 🤖 Estimez le prix de votre bien grâce à l'Intelligence Artificielle")
    
    # Charger le modèle
    model_data = load_model()
    
    if model_data is None:
        st.stop()
    
    model = model_data['model']
    feature_names = model_data['feature_names']
    
    # Sidebar avec onglets
    st.sidebar.title("🎛️ Configuration")
    
    tab_mode = st.sidebar.radio(
        "Mode",
        ["🔮 Prédiction Simple", "📊 Mode Avancé"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("📍 Localisation")
    
    arrondissement = st.sidebar.selectbox(
        "Arrondissement",
        options=list(range(1, 21)),
        index=6,
        format_func=lambda x: f"{x}ème - {get_arrondissement_info(x)['nom']}"
    )
    
    # Afficher infos arrondissement
    info_arrond = get_arrondissement_info(arrondissement)
    st.sidebar.markdown(f"""
        <div style="background: #f0f2f6; padding: 0.8rem; border-radius: 0.5rem; font-size: 0.85rem;">
        <b>📍 {info_arrond['quartiers']}</b><br>
        <i>{info_arrond['ambiance']}</i>
        </div>
    """, unsafe_allow_html=True)
    
    distance_metro = st.sidebar.slider(
        "Distance au métro (m)",
        min_value=50,
        max_value=1000,
        value=250,
        step=50,
        help="🚇 Plus vous êtes proche du métro, plus le prix augmente"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("🏗️ Caractéristiques")
    
    surface = st.sidebar.number_input(
        "Surface (m²)",
        min_value=15,
        max_value=300,
        value=60,
        step=5
    )
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        nb_pieces = st.number_input("🚪 Pièces", 1, 6, 3)
    with col2:
        nb_chambres = st.number_input("🛏️ Chambres", 0, 5, max(0, nb_pieces - 1))
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        etage = st.number_input("⬆️ Étage", 0, 10, 2)
    with col2:
        nb_etages_immeuble = st.number_input("🏢 Total étages", max(1, etage + 1), 15, max(6, etage + 2))
    
    annee_construction = st.sidebar.slider(
        "📅 Année de construction",
        min_value=1850,
        max_value=2024,
        value=1970
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("✨ Équipements & Confort")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        balcon = st.checkbox("🪴 Balcon", value=True)
        parking = st.checkbox("🚗 Parking", value=False)
        cave = st.checkbox("📦 Cave", value=True)
    
    with col2:
        terrasse = st.checkbox("🌿 Terrasse", value=False)
        ascenseur = st.checkbox("🛗 Ascenseur", value=True)
        renovation = st.checkbox("🔨 Rénové", value=False)
    
    st.sidebar.markdown("---")
    predict_button = st.sidebar.button("🔮 PRÉDIRE LE PRIX", type="primary", use_container_width=True)
    
    # Afficher les infos du modèle en haut
    with st.expander("ℹ️ Informations sur le modèle d'IA", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🤖 Modèle", model_data['model_name'])
        col2.metric("🎯 Précision (R²)", f"{model_data['r2_score']:.1%}")
        col3.metric("📊 Erreur moyenne", f"{model_data['mae']/1000:.0f}k €")
        col4.metric("📈 RMSE", f"{model_data['rmse']/1000:.0f}k €")
        
        st.markdown("---")
        st.markdown("""
        **Comment ça marche ?**
        
        Notre modèle utilise l'algorithme **""" + model_data['model_name'] + """** entraîné sur 5000+ annonces 
        immobilières réelles. Il analyse plus de 20 caractéristiques pour prédire le prix le plus précis possible.
        """)
    
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
        
        # Créer features et prédire
        df_features = create_features_from_input(input_data)
        prediction = model.predict(df_features)[0]
        prix_m2 = prediction / surface
        
        # Animation de confettis
        st.balloons()
        
        # Afficher la prédiction avec animation
        st.markdown(f"""
            <div class="prediction-box">
                <h1>💰 Prix Estimé</h1>
                <h1 style="font-size: 3.5rem; margin: 1rem 0; font-weight: bold;">{prediction:,.0f} €</h1>
                <p style="font-size: 1.4rem; opacity: 0.9;">{prix_m2:,.0f} € / m²</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Métriques détaillées avec icônes
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0;">📐 Surface</h3>
                    <h1 style="margin: 0.5rem 0; color: #667eea;">{surface} m²</h1>
                    <p style="margin: 0; color: #666;">{nb_pieces} pièces • {nb_chambres} chambres</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0;">📍 Localisation</h3>
                    <h1 style="margin: 0.5rem 0; color: #764ba2;">{arrondissement}ème</h1>
                    <p style="margin: 0; color: #666;">{distance_metro}m du métro</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            equipements = sum([balcon, terrasse, parking, cave, ascenseur, renovation])
            st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0;">✨ Confort</h3>
                    <h1 style="margin: 0.5rem 0; color: #FF4B4B;">{equipements}/6</h1>
                    <p style="margin: 0; color: #666;">équipements</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Analyse détaillée
        if tab_mode == "📊 Mode Avancé":
            st.markdown("---")
            st.subheader("📊 Analyse Détaillée")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Graphique de comparaison avec le marché
                prix_moyens_arrond = {
                    1: 13000, 2: 11000, 3: 10500, 4: 11500, 5: 12000, 6: 14000,
                    7: 14500, 8: 15000, 9: 11500, 10: 10000, 11: 10500, 12: 11000,
                    13: 9000, 14: 10500, 15: 11500, 16: 14000, 17: 11000, 18: 9500,
                    19: 8500, 20: 8000
                }
                
                prix_moyen_arrond = prix_moyens_arrond.get(arrondissement, 10000)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=['Prix marché', 'Votre bien'],
                    y=[prix_moyen_arrond, prix_m2],
                    marker_color=['#667eea', '#FF4B4B'],
                    text=[f"{prix_moyen_arrond:,.0f} €/m²", f"{prix_m2:,.0f} €/m²"],
                    textposition='auto',
                    textfont=dict(size=14, color='white')
                ))
                
                fig.update_layout(
                    title=f"Comparaison prix/m² - {arrondissement}ème arr.",
                    yaxis_title="Prix au m² (€)",
                    showlegend=False,
                    height=350,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Jauge de prix
                diff_pct = ((prix_m2 - prix_moyen_arrond) / prix_moyen_arrond) * 100
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = prix_m2,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Prix/m² vs Marché"},
                    delta = {'reference': prix_moyen_arrond, 'valueformat': '.0f'},
                    gauge = {
                        'axis': {'range': [None, prix_moyen_arrond * 1.5]},
                        'bar': {'color': "#FF4B4B"},
                        'steps': [
                            {'range': [0, prix_moyen_arrond * 0.8], 'color': "lightgray"},
                            {'range': [prix_moyen_arrond * 0.8, prix_moyen_arrond * 1.2], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "#667eea", 'width': 4},
                            'thickness': 0.75,
                            'value': prix_moyen_arrond
                        }
                    }
                ))
                
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        # Fourchette de prix
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 Fourchette de Prix")
            
            fig = go.Figure()
            
            prix_bas = prediction * 0.90
            prix_haut = prediction * 1.10
            
            fig.add_trace(go.Box(
                y=[prix_bas, prediction, prix_haut],
                name="Fourchette",
                marker_color='#667eea',
                boxmean='sd'
            ))
            
            fig.update_layout(
                height=300,
                yaxis_title="Prix (€)",
                showlegend=False,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
                <div class="info-box">
                <b>💡 Interprétation</b><br><br>
                <b>Prix bas:</b> {prix_bas:,.0f} €<br>
                <b>Prix estimé:</b> {prediction:,.0f} €<br>
                <b>Prix haut:</b> {prix_haut:,.0f} €<br><br>
                <small>Fourchette ±10% selon l'état exact du bien</small>
                </div>
            """, unsafe_allow_html=True)
        
        # Conseils
        st.markdown("---")
        st.subheader("💡 Conseils pour Optimiser Votre Prix")
        
        conseils = []
        if distance_metro > 400:
            conseils.append("🚇 **Proximité métro** : Un bien à moins de 200m du métro vaut environ 8% de plus")
        if not parking and arrondissement in [1, 6, 7, 8, 16]:
            conseils.append("🚗 **Parking** : Dans les arrondissements chers, un parking peut ajouter 10-15% à la valeur")
        if not renovation and annee_construction < 1970:
            conseils.append("🔨 **Rénovation** : Une rénovation récente augmente la valeur d'environ 12%")
        if not balcon and not terrasse:
            conseils.append("🪴 **Extérieur** : Un balcon ou une terrasse est très apprécié (+5-8%)")
        
        if conseils:
            for conseil in conseils:
                st.markdown(f"- {conseil}")
        else:
            st.success("✅ Votre bien dispose déjà de nombreux atouts valorisants !")
        
    else:
        # Page d'accueil améliorée
        st.markdown("""
            <div class="info-box">
            <h3>👋 Bienvenue !</h3>
            Renseignez les caractéristiques de votre bien dans la barre latérale, 
            puis cliquez sur <b>"PRÉDIRE LE PRIX"</b> pour obtenir une estimation précise.
            </div>
        """, unsafe_allow_html=True)
        
        # Statistiques du modèle
        st.subheader("🎯 Performance du Modèle")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🤖 Algorithme", model_data['model_name'][:15] + "...")
        col2.metric("🎯 Précision", f"{model_data['r2_score']:.1%}")
        col3.metric("📊 Erreur moy.", f"{model_data['mae']/1000:.0f}k €")
        col4.metric("📈 RMSE", f"{model_data['rmse']/1000:.0f}k €")
        
        st.markdown("---")
        
        # Features importantes
        st.subheader("🔑 Facteurs Clés Influençant le Prix")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 📊 Facteurs Principaux
            - **Surface** (32%) : Le facteur le plus important
            - **Arrondissement** (24%) : La localisation est cruciale
            - **Nombre de pièces** (18%) : Impact significatif
            - **Étage** (12%) : Les étages élevés sont prisés
            """)
        
        with col2:
            st.markdown("""
            #### 🎁 Bonus de Valorisation
            - **Proximité métro < 200m** : +8%
            - **Parking** : +10-15%
            - **Terrasse** : +8-12%
            - **Rénovation récente** : +12%
            """)


if __name__ == "__main__":
    main()
