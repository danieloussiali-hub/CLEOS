import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

# Configuration de la page
st.set_page_config(page_title="Scanner 3D ESP8266 Control Panel", layout="wide")

# --- STYLING ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR : CONNEXION ---
st.sidebar.title("🔌 Connexion")
ip_address = st.sidebar.text_input("Adresse IP / Port", value="192.168.1.100")
if st.sidebar.button("Connecter"):
    st.sidebar.success(f"Connecté à {ip_address}")
    st.session_state['connected'] = True

# --- ÉTAT DE L'APPLICATION ---
if 'connected' not in st.session_state:
    st.session_state['connected'] = False
if 'scan_data' not in st.session_state:
    st.session_state['scan_data'] = None

# --- LAYOUT PRINCIPAL ---
t1, t2 = st.tabs(["🎮 Contrôle & Scan", "📊 Analyse des données"])

with t1:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Paramètres de Scan")
        start_x = st.slider("Start X (°)", 0, 180, 20)
        end_x = st.slider("End X (°)", 0, 180, 110)
        start_y = st.slider("Start Y (°)", 0, 180, 0)
        end_y = st.slider("End Y (°)", 0, 180, 180)
        step = st.select_slider("Pas (°)", options=[1, 2, 5, 10], value=5)
        
        nx = int((end_x - start_x) / step) + 1
        ny = int((end_y - start_y) / step) + 1
        total_points = nx * ny
        st.info(f"Points estimés : {total_points}")

        if st.button("▶ Lancer le Scan", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulation de capture de données
            data_list = []
            for i, y in enumerate(range(start_y, end_y + 1, step)):
                for x in range(start_x, end_x + 1, step):
                    # Simulation d'une mesure (distance)
                    dist = 400 + 300 * np.sin(x*np.pi/180) * np.cos(y*np.pi/180) + np.random.randint(0, 50)
                    data_list.append({"x": x, "y": y, "distance": dist})
                    
                    # Mise à jour progression
                    prog = len(data_list) / total_points
                    progress_bar.progress(prog)
                status_text.text(f"Scan en cours... {int(prog*100)}%")
                time.sleep(0.05) # Simulation délai matériel
            
            st.session_state['scan_data'] = pd.DataFrame(data_list)
            st.success("Scan terminé !")

    with col2:
        st.subheader("Contrôle Manuel")
        c1, c2, c3 = st.columns(3)
        with c2: st.button("↑")
        with c1: st.button("←")
        with c2: st.button("⌂")
        with c3: st.button("→")
        with c2: st.button("↓")
        
        st.divider()
        st.subheader("Aperçu Temps Réel")
        if st.session_state['scan_data'] is not None:
            df = st.session_state['scan_data']
            # Heatmap 2D (Vue polaire/angulaire)
            fig_hm = go.Figure(data=go.Heatmap(
                z=df['distance'], x=df['x'], y=df['y'],
                colorscale='Viridis'
            ))
            fig_hm.update_layout(title="Amplitude du Capteur (Vue 2D)", height=400)
            st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.info("Lancez un scan ou importez un CSV pour voir l'aperçu.")


        st.warning("Aucune donnée disponible. Veuillez scanner ou importer un fichier.")

with t2:
    st.subheader("Traitement et Export")
    
    uploaded_file = st.file_uploader("Importer un fichier CSV (x,y,distance)", type="csv")
    if uploaded_file is not None:
        st.session_state['scan_data'] = pd.read_csv(uploaded_file)

    if st.session_state['scan_data'] is not None:
        df = st.session_state['scan_data']
        
        # Statistiques
        m1, m2, m3 = st.columns(3)
        m1.metric("Points total", len(df))
        m2.metric("Distance Min", f"{df['distance'].min():.1f}")
        m3.metric("Distance Max", f"{df['distance'].max():.1f}")

        # Visualisation avec Matplotlib (Déjà présent dans tes logs)
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        # Conversion simple pour le rendu
        theta = np.radians(df['x'])
        phi = np.radians(df['y'])
        r = df['distance']
        
        xs = r * np.sin(phi) * np.cos(theta)
        ys = r * np.sin(phi) * np.sin(theta)
        zs = r * np.cos(phi)

        img = ax.scatter(xs, ys, zs, c=df['distance'], cmap='viridis')
        fig.colorbar(img, ax=ax, label='Distance')
        ax.set_title("Nuage de points 3D (Matplotlib)")
        
        st.pyplot(fig)

        # Bouton d'export
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Télécharger CSV", csv, "scan_3d.csv", "text/csv")
