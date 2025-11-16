import streamlit as st
import numpy as np
import random
import time
import math
import pandas as pd

# --- Constantes de Simulation ---
GRID_SIZE = 10
CELL_SIZE_M = 1.0  # Taille d'une cellule en mètres
TIME_PER_STEP_S = 1.0  # Temps pour traverser une cellule
TURN_PENALTY_S = 0.5  # Temps additionnel pour un virage

# --- Fonctions Utilitaires ---

def create_random_maze(size, density=0.3):
    """Crée un labyrinthe aléatoire simple (0=libre, 1=mur)."""
    maze = np.zeros((size, size), dtype=int)
    for r in range(size):
        for c in range(size):
            if (r > 0 or c > 0) and (r < size - 1 or c < size - 1): # Évite les coins et bords extrêmes
                if random.random() < density:
                    maze[r, c] = 1
    # Assurez-vous que le point de départ (1,1) et d'arrivée (8,8) sont libres
    maze[1, 1] = 0
    maze[size - 2, size - 2] = 0
    return maze

def find_shortest_path(maze, start, end):
    """Trouve le chemin le plus court en utilisant BFS (Breadth-First Search)."""
    rows, cols = maze.shape
    queue = [([start], start)] # (path, current_position)
    visited = {start}
    
    while queue:
        path, (r, c) = queue.pop(0)
        
        if (r, c) == end:
            return path
        
        # Directions: (dr, dc)
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr, nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                new_path = path + [(nr, nc)]
                queue.append((new_path, (nr, nc)))
                
    return None # Aucun chemin trouvé

def calculate_path_metrics(path):
    """Calcule le temps total et la distance parcourue."""
    if not path:
        return 0, 0
    
    time_s = 0
    distance_m = 0
    
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i+1]
        
        # Distance (une étape = 1 cellule = CELL_SIZE_M)
        distance_m += CELL_SIZE_M
        
        # Temps: STEP_TIME + pénalité de virage
        time_s += TIME_PER_STEP_S
        
        # Simple détection de virage: si le mouvement change d'axe
        if i > 0:
            (r0, c0) = path[i-1]
            prev_dir = (r1 - r0, c1 - c0)
            curr_dir = (r2 - r1, c2 - c1)
            
            if prev_dir != curr_dir:
                time_s += TURN_PENALTY_S
                
    return round(time_s, 2), round(distance_m, 2)

def get_distance_to_nearest_wall(maze, r, c):
    """Calcule la distance minimale (en cellules) au mur le plus proche."""
    rows, cols = maze.shape
    min_dist = float('inf')
    
    # Vérifie dans les 4 directions cardinales
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        dist = 0
        nr, nc = r + dr, c + dc
        while 0 <= nr < rows and 0 <= nc < cols and maze[nr, nc] == 0:
            dist += 1
            nr += dr
            nc += dc
        
        # Si on est sorti par un mur ou la bordure
        if 0 <= nr < rows and 0 <= nc < cols and maze[nr, nc] == 1:
            min_dist = min(min_dist, dist + 0.5) # Ajout de 0.5 pour le mur adjacent
        elif nr < 0 or nr >= rows or nc < 0 or nc >= cols:
            min_dist = min(min_dist, dist + 0.5) # Bordure = Mur
            
    return min_dist if min_dist != float('inf') else 0

def get_sensor_limits_and_units(config_key):
    """Définit les limites et les unités pour chaque capteur."""
    if config_key == 'Télémétrie':
        return [
            (0, 500, 'cm'),    # Distance Proche (ex: Ultrason)
            (-100, 100, 'unités'), # Capteur 2 générique
            (-100, 100, 'unités')  # Capteur 3 générique
        ]
    elif config_key == 'Centrale inertielle':
        return [
            (-20, 20, '$m/s^2$'), # Force X (Accéléromètre)
            (-500, 500, 'deg/s'), # Vitesse Angulaire Z (Gyroscope)
            (0, 360, 'deg')      # Orientation (Magnétomètre/Angle Intégré)
        ]
    elif config_key == 'Coordonnées GPS (triangulation)':
        return [
            (0, GRID_SIZE * CELL_SIZE_M * 100, 'cm (X)'), 
            (0, GRID_SIZE * CELL_SIZE_M * 100, 'cm (Y)'),
            (0, 10, 'DOP') # Dilution of Precision
        ]
    # Autres configurations par défaut
    else: 
        return [
            (0, 100, 'unités'), 
            (0, 100, 'unités'), 
            (0, 100, 'unités')
        ]

# --- MODÈLE DE BRUIT AFFINÉ ET CORRIGÉ ---

def generate_sensor_data(maze, config_key, path_data, frame):
    """Génère des données de capteurs plus réalistes pour l'analyse graphique."""
    limits = get_sensor_limits_and_units(config_key)
    data = []
    
    path = path_data['path']
    if not path:
        return 0, 0, 0 # Retourne zéro si pas de chemin
        
    # S'assurer que l'index ne dépasse pas la taille du chemin
    current_frame = min(frame, len(path)-1)
    
    # Déterminer si le robot est en mouvement ou à un tournant
    is_moving = current_frame > 0 
    
    rx, ry = path[current_frame]
    
    if is_moving:
        px, py = path[current_frame-1]
        dx, dy = rx - px, ry - py
        # Un virage se produit si la direction précédente était différente
        is_turning = current_frame > 1 and (path[current_frame-2] != path[current_frame-1]) and (dx != 0 and dy != 0)
    else:
        dx, dy = 0, 0
        is_turning = False

    # --- Télémétrie (Ultrason/Laser) ---
    if config_key == 'Télémétrie':
        # Capteur 1: Distance Proche (Ultrason/Lidar)
        min_dist_cells = get_distance_to_nearest_wall(maze, rx, ry)
        # Valeur de base en cm
        base_val_1 = np.clip(min_dist_cells * CELL_SIZE_M * 100 - (CELL_SIZE_M * 100 / 2), 0, limits[0][1])
        
        # Bruit dépendant de la distance (bruit plus grand si plus loin)
        noise_factor = 0.05 * base_val_1 / limits[0][1] + 0.02 # Min 2% de l'échelle max
        noise = random.uniform(-noise_factor, noise_factor) * limits[0][1]
        
        val_1 = np.clip(base_val_1 + noise, limits[0][0], limits[0][1])
        data.append(val_1)
        
        # Capteurs 2 & 3: Aléatoire pour cet exemple
        for i in range(1, 3):
            min_val, max_val, _ = limits[i]
            base_val = random.uniform(min_val, max_val)
            noise = random.uniform(-0.05, 0.05) * (max_val - min_val)
            data.append(np.clip(base_val + noise, min_val, max_val))

    # --- Centrale Inertielle (IMU) ---
    elif config_key == 'Centrale inertielle':
        # Capteur 1: Force X (Accéléromètre)
        # Forte accélération/décélération lors des virages
        base_val_1 = 0.0
        if is_turning:
            # Pic d'accélération
            base_val_1 = random.uniform(8, 15) * random.choice([-1, 1])
        elif is_moving:
            # Bruit de vibration en ligne droite
            base_val_1 = random.uniform(-0.5, 0.5) 
        
        accel_noise = random.uniform(-1, 1) * 0.5 # Bruit d'accéléromètre
        data.append(np.clip(base_val_1 + accel_noise, limits[0][0], limits[0][1]))

        # Capteur 2: Vitesse Angulaire Z (Gyroscope)
        base_val_2 = 0.0
        if is_turning:
            # Pic de rotation lors d'un virage (ex: 90 deg/0.25s -> 360 deg/s)
            rotation_rate = 360 / (TURN_PENALTY_S + TIME_PER_STEP_S) 
            base_val_2 = random.uniform(rotation_rate * 0.8, rotation_rate * 1.2) * random.choice([-1, 1])
        
        gyro_noise = random.uniform(-5, 5) # Bruit de gyroscope
        data.append(np.clip(base_val_2 + gyro_noise, limits[1][0], limits[1][1]))
        
        # Capteur 3: Orientation (Magnétomètre/Angle Intégré)
        # Accumulation de l'erreur (drift)
        if is_moving:
            # L'intégration est basée sur le taux de rotation bruité
            rotation_step = data[1] * (TIME_PER_STEP_S + TURN_PENALTY_S) 
            st.session_state.heading_drift += rotation_step * 0.01 # Intégration simplifiée de l'erreur (biais)
        
        # Une valeur d'orientation simple basée sur le temps + drift
        orientation_val = (frame * 10) % 360 + st.session_state.heading_drift
        data.append(np.clip(orientation_val, limits[2][0], limits[2][1]))

    # --- Coordonnées GPS (Triangulation) ---
    elif config_key == 'Coordonnées GPS (triangulation)':
        
        # Coordonnées réelles (en cm)
        real_x = rx * CELL_SIZE_M * 100
        real_y = ry * CELL_SIZE_M * 100
        
        # Bruit de localisation plus important en mouvement
        # Erreur standard en centimètres
        gps_error_cm = 10 + (30 * (is_moving)) # 10cm de base + 30cm en mouvement
        
        # Capteur 1 (X)
        noise_x = random.gauss(0, gps_error_cm)
        data.append(np.clip(real_x + noise_x, limits[0][0], limits[0][1]))
        
        # Capteur 2 (Y)
        noise_y = random.gauss(0, gps_error_cm)
        data.append(np.clip(real_y + noise_y, limits[1][0], limits[1][1]))
        
        # Capteur 3 (DOP: Dilution of Precision) - dépend du mouvement/environnement
        base_dop = 3.0 + random.uniform(0, 3) 
        dop_val = base_dop + (3 * is_moving) # Erreur augmente en mouvement
        data.append(np.clip(dop_val, limits[2][0], limits[2][1]))
        
    # --- Autres Capteurs (Générique) ---
    else:
        # Code générique pour Caméras/Radar
        for min_val, max_val, _ in limits:
            base_val = random.uniform(min_val, max_val)
            noise = random.uniform(-0.05, 0.05) * (max_val - min_val)
            data.append(np.clip(base_val + noise, min_val + noise, max_val - noise))
            
    return data[0], data[1], data[2]

# --- Initialisation de l'état de la session (CORRECTION DE L'ERREUR KEYERROR) ---

# --- Toutes les clés utilisées dans st.session_state doivent être initialisées ici ---
if 'maze' not in st.session_state:
    st.session_state.maze = create_random_maze(GRID_SIZE)

if 'path_data' not in st.session_state:
    # Chemin par défaut (peut être mis à jour plus tard)
    st.session_state.path_data = {'path': [], 'time': 0, 'distance': 0}

if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = pd.DataFrame()

if 'heading_drift' not in st.session_state:
    st.session_state.heading_drift = 0.0 # Clé ajoutée pour la dérive de l'IMU/Gyro

# --- Logique Principale de l'Application Streamlit ---

st.set_page_config(layout="wide", page_title="Simulation Odométrique")

st.title("🗺️ Simulation d'Odométrie et d'Analyse de Capteurs")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("⚙️ Configuration")
    
    # Bouton pour générer un nouveau labyrinthe
    if st.button("🔄 Générer un nouveau Labyrinthe"):
        st.session_state.maze = create_random_maze(GRID_SIZE)
        st.session_state.path_data = {'path': [], 'time': 0, 'distance': 0}
        st.session_state.sensor_data = pd.DataFrame() # Réinitialiser les données des capteurs

    # Paramètres du robot
    start = (1, 1)
    end = (GRID_SIZE - 2, GRID_SIZE - 2)

    # Calcul et affichage du chemin
    path = find_shortest_path(st.session_state.maze, start, end)
    
    if path:
        time_s, distance_m = calculate_path_metrics(path)
        st.session_state.path_data = {'path': path, 'time': time_s, 'distance': distance_m}

        st.success(f"Chemin trouvé en {len(path)} étapes.")
        st.info(f"Temps total estimé: **{time_s} s** | Distance: **{distance_m} m**")
    else:
        st.session_state.path_data = {'path': [], 'time': 0, 'distance': 0}
        st.error("Aucun chemin trouvé pour ce labyrinthe.")

    st.subheader("Modèle de Capteur")
    config_key = st.selectbox(
        "Sélectionnez le type de capteur à analyser :",
        ['Télémétrie', 'Centrale inertielle', 'Coordonnées GPS (triangulation)', 'Caméra (générique)', 'Radar (générique)']
    )

    # Bouton de simulation
    if st.button("▶️ Lancer la Simulation"):
        if path:
            st.session_state.sensor_data = pd.DataFrame()
            st.session_state.heading_drift = 0.0 # Réinitialiser la dérive à chaque simulation
            
            # Création d'une barre de progression
            progress_bar = st.progress(0)
            
            all_sensor_data = []
            
            for i, (r, c) in enumerate(path):
                # Génération des données bruitées
                s1, s2, s3 = generate_sensor_data(st.session_state.maze, config_key, st.session_state.path_data, i)
                
                all_sensor_data.append({
                    'Étape': i,
                    'Temps (s)': i * TIME_PER_STEP_S, # Temps simple pour le graphique
                    'Capteur 1': s1,
                    'Capteur 2': s2,
                    'Capteur 3': s3
                })
                
                # Mise à jour de la barre
                progress_bar.progress((i + 1) / len(path))
                
                # Petit délai pour l'effet de progression
                time.sleep(0.01)

            st.session_state.sensor_data = pd.DataFrame(all_sensor_data)
            progress_bar.empty()
            st.success("Simulation terminée. Analyse des données des capteurs prête.")
        else:
            st.warning("Veuillez générer un labyrinthe avec un chemin valide avant de simuler.")

with col2:
    st.header("🖼️ Labyrinthe et Trajectoire")
    
    # Affichage du labyrinthe
    fig_maze = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8) # Image RGB
    
    # Couleurs
    WALL_COLOR = [50, 50, 50]       # Gris foncé
    PATH_COLOR = [150, 255, 150]    # Vert clair
    START_COLOR = [0, 200, 0]       # Vert
    END_COLOR = [200, 0, 0]         # Rouge
    
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if st.session_state.maze[r, c] == 1:
                fig_maze[r, c] = WALL_COLOR
            else:
                fig_maze[r, c] = [255, 255, 255] # Blanc
    
    # Afficher le chemin
    if path:
        for r, c in path:
            fig_maze[r, c] = PATH_COLOR
        fig_maze[start[0], start[1]] = START_COLOR
        fig_maze[end[0], end[1]] = END_COLOR
        
    st.image(fig_maze, caption="Labyrinthe (Mur Gris, Chemin Vert, Début Vert Clair, Fin Rouge)", use_column_width=True)

st.header(f"📈 Analyse Odométrique du Capteur : **{config_key}**")
st.markdown("Les graphiques ci-dessous représentent les signaux bruts et bruités de vos capteurs. Ils sont les données d'entrée typiques pour un filtre de Kalman ou un algorithme de fusion de données.")

if not st.session_state.sensor_data.empty:
    
    # Récupérer les unités pour les titres des graphiques
    limits_and_units = get_sensor_limits_and_units(config_key)
    
    data_df = st.session_state.sensor_data.set_index('Temps (s)')
    
    col_chart_1, col_chart_2, col_chart_3 = st.columns(3)

    with col_chart_1:
        st.subheader(f"Capteur 1 ({limits_and_units[0][2]})")
        st.line_chart(data_df['Capteur 1'])

    with col_chart_2:
        st.subheader(f"Capteur 2 ({limits_and_units[1][2]})")
        st.line_chart(data_df['Capteur 2'])

    with col_chart_3:
        st.subheader(f"Capteur 3 ({limits_and_units[2][2]})")
        st.line_chart(data_df['Capteur 3'])

    st.subheader("Données Brutes (Extrait)")
    st.dataframe(st.session_state.sensor_data.head())

else:
    st.info("Lancez la simulation pour visualiser l'analyse des capteurs.")
