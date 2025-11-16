import streamlit as st
import pandas as pd
import numpy as np

# --- Configuration de la Page ---
st.set_page_config(
    page_title="Recherche Multimodale (MMS) - Démo",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔎 Démo de l'Application de Recherche Multimodale")

# --- Barre Latérale (Simule les Options de Filtre) ---
with st.sidebar:
    st.header("Paramètres de Recherche")
    
    # Simule le choix d'un modèle ou d'une source
    modele_choisi = st.selectbox(
        "Sélectionner le Modèle",
        ["Modèle A (Vision)", "Modèle B (Texte)", "Modèle C (Combiné)"]
    )
    
    # Simule un paramètre de seuil de confiance
    seuil_confiance = st.slider(
        "Seuil de Confiance Minimum",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.01
    )
    
    st.markdown("---")
    if st.button("Lancer la Recherche"):
        st.session_state['recherche_lancee'] = True

# --- Contenu Principal ---

# 1. Zone de Requête Utilisateur
st.subheader("Entrez votre Requête")
col1, col2 = st.columns([3, 1])

with col1:
    requete_texte = st.text_input("Requête Textuelle", "Chaton mignon")

with col2:
    requete_image = st.file_uploader("Requête par Image", type=["png", "jpg", "jpeg"])

# 2. Affichage des Résultats
if st.session_state.get('recherche_lancee', False):
    st.header("Résultats de la Recherche")

    st.info(f"Recherche lancée avec le modèle: **{modele_choisi}** et seuil: **{seuil_confiance:.2f}**")

    # Simulation des données de résultat
    data = {
        'ID': range(1, 6),
        'Titre': [f"Résultat {i}" for i in range(1, 6)],
        'Pertinence': np.round(np.random.uniform(seuil_confiance, 1.0, 5), 2)
    }
    df = pd.DataFrame(data).sort_values(by='Pertinence', ascending=False)
    
    # Affichage en colonnes
    cols_results = st.columns(5)
    
    for i, row in df.iterrows():
        with cols_results[i-1]:
            st.metric(label=f"Résultat {row['ID']} ({row['Titre']})", value=f"{row['Pertinence']:.2f}", delta=f"{modele_choisi}")
            st.image("https://via.placeholder.com/150", caption=row['Titre'], use_column_width=True)
            
    st.markdown("---")
    st.dataframe(df, use_container_width=True)

# 3. Message de Bienvenue Initial
else:
    st.warning("Veuillez configurer les paramètres dans la barre latérale et cliquer sur 'Lancer la Recherche'.")
