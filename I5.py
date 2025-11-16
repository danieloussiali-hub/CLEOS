import time
import random
import heapq
from collections import deque
import json
import base64

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-GUI backend suitable for Streamlit
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

# --- Constantes et Configuration Physique ---

# Vitesse du robot: 1 km/h = 1000m / 3600s = 0.2777... m/s
ROBOT_SPEED_MPS = 1.0 / 3.6 
# Espacement des murs: 20 cm = 0.2 m
CELL_SIZE_M = 0.2
# Temps par cellule à vitesse maximale (sans compter les virages)
TIME_PER_CELL_MAX_S = CELL_SIZE_M / ROBOT_SPEED_MPS 
# Pénalité de temps par virage (estimation)
TURN_PENALTY_S = 0.5 

DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W
MAZE_SIZE = 16

# --- Maze, Cartographie, et Algorithmes ---

def create_all_mazes():
    # Codes: 0=Passage, 1=Mur, 2=Départ (Start), 3=Arrivée (End)
    mazes = {}
    mazes['Classique (Facile)'] = np.array([
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], [1,2,0,0,1,0,0,0,0,0,0,0,0,0,0,1], [1,0,1,0,1,0,1,1,1,1,0,1,1,1,0,1], [1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1],
        [1,0,1,1,1,1,1,0,1,1,1,1,0,1,0,1], [1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1], [1,1,1,0,1,1,1,1,1,1,0,1,1,1,0,1], [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1], [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], [1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1], [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1], [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1], [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    ])
    # D'autres labyrinthes ici... (omission pour concision)
    mazes['Complexe (Moyen)'] = np.array([
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], [1,2,1,0,0,0,1,0,0,0,1,0,0,0,0,1], [1,0,1,0,1,0,1,0,1,0,1,0,1,1,0,1], [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1],
        [1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,1], [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1], [1,0,1,1,1,1,1,1,1,0,1,1,1,1,0,1], [1,0,1,0,0,0,0,0,1,0,0,0,0,1,0,1],
        [1,0,1,0,1,1,1,0,1,1,1,1,0,1,0,1], [1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1], [1,1,1,0,1,0,1,1,1,1,1,1,1,1,0,1], [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,1,1,1,1,1,1,1,0,1,1,1,1,0,1], [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1], [1,1,1,1,1,0,1,1,1,1,1,0,1,1,3,1], [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    ])
    
    return mazes

# --- Fonctions d'exploration (Passage 1) ---

def wall_follow(maze, sx, sy, sh, right_hand=False):
    start_time = time.time() # Chrono
    path = [(sx, sy)]; visited = set([(sx, sy)]); x, y, heading = sx, sy, sh
    rows, cols = maze.shape
    
    # Limiter l'exploration à la taille du labyrinthe * 5 pour éviter les boucles infinies
    max_steps = MAZE_SIZE * MAZE_SIZE * 5 
    steps = 0
    
    while steps < max_steps:
        if maze[y][x] == 3: break
        
        turn = 1 if right_hand else -1
        turn_heading = (heading + turn) % 4
        dy, dx = DIRECTIONS[turn_heading]; turn_y, turn_x = y + dy, x + dx
        
        # 1. Vérifier si on peut tourner (la main touche le mur)
        if 0 <= turn_y < rows and 0 <= turn_x < cols and maze[turn_y][turn_x] != 1:
            heading = turn_heading; x, y = turn_x, turn_y
        else:
            # 2. Avancer tout droit
            dy, dx = DIRECTIONS[heading]; next_y, next_x = y + dy, x + dx
            if 0 <= next_y < rows and 0 <= next_x < cols and maze[next_y][next_x] != 1: 
                x, y = next_x, next_y
            else: 
                # 3. Tourner dans l'autre sens
                heading = (heading - turn) % 4; steps += 1; continue
                
        path.append((x, y)); visited.add((x, y)); steps += 1
        
    end_time = time.time()
    return {'path': path, 'visited': visited, 'time': end_time - start_time}

def dfs_search(maze, sx, sy):
    start_time = time.time() # Chrono
    
    # Utilisation d'un labyrinthe "map" pour stocker uniquement les murs rencontrés.
    # On initialise tout à '1' (mur/inconnu) sauf les bords qui restent à '1' (mur réel)
    mapped_maze = np.ones_like(maze) 
    rows, cols = maze.shape
    mapped_maze[sy, sx] = 0 # Le départ est connu
    
    stack = [((sx, sy), [(sx, sy)])]; visited = {(sx, sy)}
    
    # Exploration pour cartographier tout le labyrinthe accessible
    while stack:
        (x, y), path_history = stack.pop()
        
        # Mettre à jour les informations du mur/passage pour la cellule actuelle
        if maze[y][x] != 1:
            mapped_maze[y][x] = maze[y][x] if maze[y][x] in [2, 3] else 0
        
        for dy, dx in DIRECTIONS:
            nx, ny = x + dx, y + dy
            
            if 0 <= ny < rows and 0 <= nx < cols:
                
                # Mise à jour de la carte: si c'est un mur, on le note sur la carte
                if maze[ny][nx] == 1:
                    mapped_maze[ny][nx] = 1 # Le mur est révélé
                
                if maze[ny][nx] != 1 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    # La path_history est pour le chemin d'exploration, pas pour le chemin final
                    stack.append(((nx, ny), path_history + [(nx, ny)]))
    
    end_time = time.time()
    
    # On utilise le labyrinthe "révélé" (mapped_maze) pour le Passage 2
    return {'path': path_history, 'visited': visited, 'time': end_time - start_time, 'mapped_maze': mapped_maze}

# --- Fonction Planification (Passage 2: A* optimisé) ---

def calculate_best_path(mapped_maze, start_pos, end_pos):
    rows, cols = mapped_maze.shape
    sx, sy = start_pos; ex, ey = end_pos

    def get_cost(path):
        """Calcule le coût total (temps) d'un chemin, incluant la pénalité de virage."""
        if not path or len(path) <= 1: return 0.0, 0
        
        # Coût de base: temps passé à avancer
        time_cost = (len(path) - 1) * TIME_PER_CELL_MAX_S
        turn_count = 0
        
        if len(path) > 1:
            for i in range(1, len(path) - 1):
                px, py = path[i-1]; cx, cy = path[i]; nx, ny = path[i+1]
                dx_curr = cx - px; dy_curr = cy - py
                dx_next = nx - cx; dy_next = ny - cy
                
                # Vérification du changement de direction
                if dx_curr != dx_next or dy_curr != dy_next:
                    turn_count += 1
        
        time_cost += turn_count * TURN_PENALTY_S
        return time_cost, turn_count

    def heuristic(x, y):
        """Heuristique: Distance de Manhattan (estimation du temps minimum restant)."""
        return (abs(ex - x) + abs(ey - y)) * TIME_PER_CELL_MAX_S

    # Open set stocke: (f_score, g_score, x, y, path, last_dx, last_dy)
    # g_score = coût réel (temps) depuis le départ
    # f_score = g_score + heuristic
    open_set = [(heuristic(sx, sy), 0, sx, sy, [(sx, sy)], 0, 0)]
    g_score_map = {(sx, sy, 0, 0): 0} # Clé: (x, y, last_dx, last_dy) pour pénaliser les virages

    best_path = None
    min_cost = float('inf')

    while open_set:
        f, g, x, y, path, last_dx, last_dy = heapq.heappop(open_set)

        if x == ex and y == ey:
            current_cost, _ = get_cost(path)
            if current_cost < min_cost:
                min_cost = current_cost
                best_path = path
            continue 

        for dy, dx in DIRECTIONS:
            nx, ny = x + dx, y + dy
            
            # Vérifier que la nouvelle position est dans les limites et n'est pas un mur sur la carte révélée
            if 0 <= ny < rows and 0 <= nx < cols and mapped_maze[ny][nx] != 1:
                
                new_g = g + TIME_PER_CELL_MAX_S
                
                is_turn = False
                # Ne pas pénaliser le premier mouvement (quand last_dx et last_dy sont à 0)
                if len(path) > 1 and (dx != last_dx or dy != last_dy):
                    new_g += TURN_PENALTY_S
                    is_turn = True
                
                # La clé doit inclure le sens d'arrivée pour gérer la pénalité de virage
                key = (nx, ny, dx, dy)
                
                if new_g < g_score_map.get(key, float('inf')):
                    g_score_map[key] = new_g
                    
                    new_path = path + [(nx, ny)]
                    new_f = new_g + heuristic(nx, ny)
                    
                    heapq.heappush(open_set, (new_f, new_g, nx, ny, new_path, dx, dy))
                    
    return best_path, min_cost

# --- Fonctions Utilitaires et Capteurs (restent inchangées) ---

def get_sensor_limits_and_units(config_key):
    if config_key == 'Télémétrie':
        return (0, 35, 'cm'), (0, 1, 'Probabilité'), (0, 500, 'Lux')
    elif config_key == 'Caméras':
        return (0, 255, 'Niveau de gris'), (0, 10, 'Pixels'), (-180, 180, 'Degrés')
    elif config_key == 'Centrale inertielle':
        return (-15, 15, 'm/s²'), (-180, 180, '°/s'), (0, 360, 'Degrés')
    elif config_key == 'Système radar doppler et optique':
        return (-20, 20, 'm/s'), (0, 100, 'dB'), (0, 1000, 'Lux')
    elif config_key == 'Coordonnées GPS (triangulation)':
        return (0, 1600*CELL_SIZE_M, 'm (X)'), (0, 1600*CELL_SIZE_M, 'm (Y)'), (0, 10, 'DOP (m)')
    else:
        return (0, 10, 'Valeur'), (0, 1, 'Valeur'), (0, 100, 'Valeur')

def get_distance_to_nearest_wall(maze, rx, ry):
    distances = []
    for dy, dx in DIRECTIONS:
        d = 0
        curr_x, curr_y = rx, ry
        while True:
            curr_x += dx
            curr_y += dy
            d += 1
            if not (0 <= curr_y < MAZE_SIZE and 0 <= curr_x < MAZE_SIZE) or maze[curr_y][curr_x] == 1:
                distances.append(d)
                break
    return min(distances) if distances else 0

def generate_sensor_data(maze, config_key, path_data, frame):
    limits = get_sensor_limits_and_units(config_key)
    data = []
    
    if path_data and frame < len(path_data['path']):
        rx, ry = path_data['path'][frame]
        
        # Capteur 1 (ancré à la position/environnement)
        min_val, max_val, _ = limits[0]
        if config_key == 'Télémétrie':
            min_dist_cells = get_distance_to_nearest_wall(maze, rx, ry)
            base_val = np.clip(min_dist_cells * 10, min_val + 5, max_val)
            noise = random.uniform(-0.1, 0.1) * (max_val / 2)
        elif config_key == 'Coordonnées GPS (triangulation)':
            base_val = np.clip(rx * (max_val / MAZE_SIZE), min_val, max_val)
            noise = random.uniform(-5, 5)
        else:
            base_val = random.uniform(min_val, max_val)
            noise = random.uniform(-0.05, 0.05) * (max_val - min_val)
            
        val_1 = np.clip(base_val + noise, min_val, max_val)
        data.append(val_1)
            
    else:
        # Données aléatoires par défaut
        min_val, max_val, _ = limits[0]
        data.append(random.uniform(min_val, max_val))
    
    # Capteurs 2 et 3 (aléatoires avec bruit)
    for min_val, max_val, _ in limits[1:]:
        base_val = random.uniform(min_val, max_val)
        noise = random.uniform(-0.05, 0.05) * (max_val - min_val)
        val = base_val + noise
        data.append(np.clip(val, min_val, max_val))
        
    return data[0], data[1], data[2]

def get_total_time_path(path):
    # La fonction existe déjà mais est renommée pour éviter les conflits et clarifier
    # On va utiliser celle implémentée dans A* qui est plus précise. 
    # Cette version simplifiée est conservée pour la compatibilité
    if not path or len(path) <= 1: return 0.0, 0
    total_time = (len(path) - 1) * TIME_PER_CELL_MAX_S 
    turn_count = 0
    if len(path) > 1:
        for i in range(1, len(path) - 1):
            px, py = path[i-1]; cx, cy = path[i]; nx, ny = path[i+1]
            dx_curr = cx - px; dy_curr = cy - py
            dx_next = nx - cx; dy_next = ny - cy
            if dx_curr != dx_next or dy_curr != dy_next:
                turn_count += 1
    total_time += turn_count * TURN_PENALTY_S
    return total_time, turn_count

# --- Plotting ---

def plot_maze_and_path(maze, path_data, current_frame, highlight_path=None, current_stage="Prêt"):
    
    # 🎨 Définition des couleurs
    COLORS = {
        'Passage': '#222831',     # Noir/Gris très foncé pour les passages (connu)
        'Mur': '#00ADB5',         # Cyan pour les murs (connu)
        'Départ': '#EEEEEE',      # Blanc pour le départ
        'Arrivée': '#F05454',     # Rouge pour l'arrivée
        'Inconnu': '#171A1D',     # Noir absolu pour les zones inconnues
        'Visité': '#00ADB5',      # Cyan clair pour la zone visitée
        'Trajet_Planifie': '#F05454', # Rouge pour le chemin optimal
        'Robot_Actuel': '#FFD369'     # Jaune pour le robot et le parcours actuel
    }
    
    fig, ax = plt.subplots(figsize=(6.5, 6.5), facecolor='#393E46') 
    ax.set_facecolor('#393E46') 
    rows, cols = maze.shape

    # Déterminer la visibilité du labyrinthe (Passage 1 ou Passages 2/3)
    is_exploring = current_stage.startswith("Passage 1") 
    
    # 1. Rendu du Labyrinthe (Murs/Passages)
    for yy in range(rows):
        for xx in range(cols):
            cell_val = maze[yy][xx]
            
            # --- Logique de Visibilité ---
            # Si nous sommes en exploration (Passage 1) et que la case n'a pas été visitée/connue: C'est INCONNU
            is_visited_or_known = (xx, yy) in path_data.get('visited', set()) or cell_val in [2, 3]
            
            if is_exploring and not is_visited_or_known:
                color = COLORS['Inconnu']
            elif cell_val == 1:
                color = COLORS['Mur']
            elif cell_val == 2:
                color = COLORS['Départ']
            elif cell_val == 3:
                color = COLORS['Arrivée']
            else: # Passage (0)
                color = COLORS['Passage']
            
            ax.add_patch(Rectangle((xx, yy), 1, 1, facecolor=color, edgecolor='#222831', linewidth=0.2))

    # 2. Mise en évidence de la zone explorée (Passage 1)
    if is_exploring and 'visited' in path_data:
        # On ne met en surbrillance que les passages visités, pas les murs révélés
        for (x, y) in path_data['visited']:
            if maze[y][x] != 1: 
                 ax.add_patch(Rectangle((x, y), 1, 1, facecolor=COLORS['Visité'], alpha=0.15, edgecolor='none'))

    # 3. Rendu du Chemin (Optimal ou Exploration)
    path = highlight_path if highlight_path is not None else path_data.get('path', [])
    
    # Pour le Passage 3, on affiche le chemin optimal planifié en arrière-plan
    if not is_exploring and path:
        path_x_full = [p[0] + 0.5 for p in path]
        path_y_full = [p[1] + 0.5 for p in path]
        ax.plot(path_x_full, path_y_full, color=COLORS['Trajet_Planifie'], linestyle=':', linewidth=1.5, alpha=0.6, label='Chemin Optimal Planifié')

    # Rendu du parcours actuel du robot
    path_segment = path[:current_frame + 1]
    if len(path_segment) > 1:
        path_x = [p[0] + 0.5 for p in path_segment]
        path_y = [p[1] + 0.5 for p in path_segment]
        ax.plot(path_x, path_y, color=COLORS['Robot_Actuel'], linestyle='-', linewidth=3, solid_capstyle='round', label='Parcours Actuel')
    
    # Rendu du robot (position actuelle)
    if path:
        idx = min(current_frame, len(path) - 1)
        rx, ry = path[idx]
        ax.plot(rx + 0.5, ry + 0.5, 'o', color=COLORS['Robot_Actuel'], markersize=14, markeredgecolor=COLORS['Départ'], markeredgewidth=1.5, zorder=5, label='Robot')

    ax.set_xlim(0, MAZE_SIZE); ax.set_ylim(MAZE_SIZE, 0); ax.set_aspect('equal'); ax.axis('off')
    fig.tight_layout()
    return fig

def plot_sensor_graphs():
    # ... (fonction de tracé des capteurs inchangée) ...
    config_key = st.session_state.config
    titles = {
        'Télémétrie': ("Distance Proche (Ultrason)", "Modèle Probabiliste (Pression)", "Scan Laser (Balayage)"),
        'Caméras': ("Luminosité (Simple)", "Disparité (Stéréoscopie)", "Angle Panoramique"),
        'Centrale inertielle': ("Force X (Accéléromètre)", "Vitesse Angulaire Z (Gyroscope)", "Orientation (Magnétomètre)"),
        'Système radar doppler et optique': ("Vitesse Rel. (Doppler)", "Amplitude (Radar)", "Luminosité (Optique)"),
        'Coordonnées GPS (triangulation)': ("X (Localisation)", "Y (Localisation)", "Erreur (Dilution)"),
    }
    limits_and_units = get_sensor_limits_and_units(config_key)
    figs = []
    
    COLORS = ['#F05454', '#00ADB5', '#FFD369']
    LINESTYLE = '-'

    for i in range(3):
        fig, ax = plt.subplots(figsize=(4,1.8), facecolor='#393E46')
        ax.set_facecolor('#222831')
        key = f'capteur{i+1}'
        ax.set_title(titles.get(config_key, ("Capteur 1","Capteur 2","Capteur 3"))[i], fontsize=9, color='#EEEEEE')
        data = st.session_state.sensor_data.get(key, [])
        if data:
            times = [d[0] for d in data]; values = [d[1] for d in data]
            
            # Correction de l'erreur Matplotlib (ValueError: unrecognized format specifier)
            ax.plot(times, values, color=COLORS[i], linestyle=LINESTYLE, linewidth=1) 
            
        min_val, max_val, unit = limits_and_units[i]
        ax.set_ylim(min_val, max_val); ax.set_ylabel(unit, fontsize=8, color='#EEEEEE')
        ax.set_xlabel('Temps (s)', fontsize=8, color='#EEEEEE')
        ax.tick_params(axis='both', which='major', labelsize=7, colors='#EEEEEE')
        ax.grid(True, alpha=0.25, color='#4E545B')
        fig.tight_layout()
        figs.append(fig)
    return figs

# --- Streamlit App UI & State Management ---

def load_current_maze():
    """Charge le labyrinthe en cours (pré-défini ou custom) et met à jour les positions S/E."""
    maze_key = st.session_state.maze_name
    
    if maze_key == 'Construction autonome':
        current_maze = st.session_state.custom_maze_array.copy()
    else:
        # Si un labyrinthe prédéfini est choisi, assurez-vous de repartir d'une copie propre
        current_maze = st.session_state.mazes.get(maze_key, st.session_state.mazes['Classique (Facile)']).copy()
        
    # Mettre à jour les positions de départ et d'arrivée dans le labyrinthe chargé
    # On remet à 0 d'abord pour le cas où S/E ont bougé
    current_maze[current_maze == 2] = 0 
    current_maze[current_maze == 3] = 0 
    
    # S'assurer que S/E sont dans les limites 
    s_x, s_y = st.session_state.start_pos[0], st.session_state.start_pos[1]
    e_x, e_y = st.session_state.end_pos[0], st.session_state.end_pos[1]

    if 0 <= s_y < MAZE_SIZE and 0 <= s_x < MAZE_SIZE:
        current_maze[s_y, s_x] = 2
    if 0 <= e_y < MAZE_SIZE and 0 <= e_x < MAZE_SIZE:
        current_maze[e_y, e_x] = 3
    
    return current_maze

# (Code de style CSS omis ici pour la concision, mais il doit être conservé dans le vrai fichier)

st.set_page_config(page_title="Simulateur Labyrinthe Robotique", layout="wide")

# --- Initialisation des variables de session ---
if 'mazes' not in st.session_state: st.session_state.mazes = create_all_mazes()
if 'custom_maze_array' not in st.session_state: 
    st.session_state.custom_maze_array = np.ones((MAZE_SIZE, MAZE_SIZE), dtype=int)
    st.session_state.custom_maze_array[1][1] = 2
    st.session_state.custom_maze_array[MAZE_SIZE-2][MAZE_SIZE-2] = 3
if 'maze_name' not in st.session_state: st.session_state.maze_name = 'Complexe (Moyen)'
if 'algo' not in st.session_state: st.session_state.algo = 'Deep First Search'
if 'config' not in st.session_state: st.session_state.config = 'Télémétrie'
if 'start_pos' not in st.session_state: st.session_state.start_pos = (1, 1)
if 'end_pos' not in st.session_state: st.session_state.end_pos = (MAZE_SIZE-2, MAZE_SIZE-2)
if 'path_data_map' not in st.session_state: st.session_state.path_data_map = None # Résultat Passage 1
if 'mapped_maze' not in st.session_state: st.session_state.mapped_maze = None # Carte révélée
if 'path_data_optimal' not in st.session_state: st.session_state.path_data_optimal = None # Résultat Passage 2/3
if 'frame' not in st.session_state: st.session_state.frame = 0
if 'sensor_data' not in st.session_state: st.session_state.sensor_data = {'capteur1': [], 'capteur2': [], 'capteur3': []}
if 'is_running' not in st.session_state: st.session_state.is_running = False
if 'delay_ms' not in st.session_state: st.session_state.delay_ms = 100
if 'total_time' not in st.session_state: st.session_state.total_time = 0.0 # Temps calculé (P2/P3)
if 'time_p1' not in st.session_state: st.session_state.time_p1 = 0.0 # Chrono Passage 1
if 'time_p2' not in st.session_state: st.session_state.time_p2 = 0.0 # Chrono Passage 2
if 'history' not in st.session_state: st.session_state.history = []
if 'stage' not in st.session_state: st.session_state.stage = "Prêt"
if 'current_passage' not in st.session_state: st.session_state.current_passage = 0 # 0=Prêt, 1=P1, 2=P2, 3=P3
if 'selection_mode' not in st.session_state: st.session_state.selection_mode = None 
# Temps de la dernière mise à jour pour contrôler la cadence (Passage 3)
if 'last_update_time' not in st.session_state: st.session_state.last_update_time = time.time() 


ALGO_MAP = {
    'Wall following left-hand rule': 'wallFollowLeft',
    'Wall following right-hand rule': 'wallFollowRight',
    'Deep First Search': 'dfs',
    'Broad First Search': 'bfs', # Non implémenté ici pour rester simple
    'Tremaux': 'tremaux', # Non implémenté ici pour rester simple
    'AI-based learning (A*)': 'aiLearning' # Le Passage 2 utilise A*
}

def run_simulation(start_passage=1):
    """Contrôleur de la séquence des 3 passages."""
    
    st.session_state.selection_mode = None
    maze = load_current_maze()
    sx, sy = st.session_state.start_pos
    ex, ey = st.session_state.end_pos
    
    # --- Passage 1: Cartographie / Exploration ---
    if start_passage <= 1:
        st.session_state.current_passage = 1
        st.session_state.stage = "Passage 1/3: Exploration et Cartographie"
        st.session_state.frame = 0 # Initialiser l'animation du P1
        
        # On utilise toujours DFS pour garantir l'exploration (sinon on n'aurait pas la carte complète)
        st.session_state.path_data_map = dfs_search(maze, sx, sy)
        st.session_state.time_p1 = st.session_state.path_data_map['time']
        st.session_state.mapped_maze = st.session_state.path_data_map['mapped_maze']
        
        # Passer directement au Passage 2
        run_simulation(start_passage=2)
        return

    # --- Passage 2: Planification du Chemin Optimal (A*) ---
    if start_passage <= 2:
        st.session_state.current_passage = 2
        st.session_state.stage = "Passage 2/3: Calcul du Chemin Optimal (A*)"
        st.session_state.sensor_data = {'capteur1': [], 'capteur2': [], 'capteur3': []}
        
        calc_start_time = time.time()
        best_path, min_cost = calculate_best_path(st.session_state.mapped_maze, st.session_state.start_pos, st.session_state.end_pos)
        calc_end_time = time.time()
        
        st.session_state.time_p2 = calc_end_time - calc_start_time # Chrono Calcul
        st.session_state.path_data_optimal = {'path': best_path, 'visited': st.session_state.path_data_map['visited']}
        st.session_state.total_time = min_cost # Le temps simulé pour le P3 est le coût optimal calculé
        
        # Passer directement au Passage 3
        run_simulation(start_passage=3)
        return

    # --- Passage 3: Exécution Rapide / Simulation ---
    if start_passage == 3:
        st.session_state.current_passage = 3
        st.session_state.stage = "Passage 3/3: Exécution du Trajet Optimal"
        st.session_state.frame = 0
        st.session_state.is_running = True
        st.session_state.last_update_time = time.time()
        st.rerun() 
        return

def reset():
    st.session_state.frame = 0
    st.session_state.is_running = False
    st.session_state.path_data_map = None
    st.session_state.path_data_optimal = None
    st.session_state.mapped_maze = None
    st.session_state.sensor_data = {'capteur1': [], 'capteur2': [], 'capteur3': []}
    st.session_state.total_time = 0.0
    st.session_state.time_p1 = 0.0
    st.session_state.time_p2 = 0.0
    st.session_state.stage = "Prêt"
    st.session_state.current_passage = 0
    st.session_state.selection_mode = None 
    st.session_state.last_update_time = time.time()

def save_simulation():
    if st.session_state.path_data_optimal and st.session_state.total_time > 0:
        # Assurer que l'état du labyrinthe est sauvegardé s'il est personnalisé
        maze_array_to_save = st.session_state.custom_maze_array.tolist() if st.session_state.maze_name == 'Construction autonome' else None
        
        sim_data = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'maze_name': st.session_state.maze_name,
            'algo': st.session_state.algo,
            'config': st.session_state.config,
            'start_pos': st.session_state.start_pos,
            'end_pos': st.session_state.end_pos,
            'total_time': st.session_state.total_time, # Chrono P3
            'time_p1': st.session_state.time_p1, # Chrono P1
            'path': st.session_state.path_data_optimal['path'],
            'visited': list(st.session_state.path_data_optimal['visited']),
            'maze_array': maze_array_to_save
        }
        st.session_state.history.append(sim_data)
        st.sidebar.success("Simulation sauvegardée !")

# --- UI Sidebar (Contrôles) ---

with st.sidebar:
    st.title("🚀 Contrôles du Robot")
    
    st.subheader("Configuration de la Course")
    maze_options = list(st.session_state.mazes.keys()) + ['Construction autonome']
    st.session_state.maze_name = st.selectbox("Labyrinthe", maze_options, index=maze_options.index(st.session_state.maze_name))
    st.session_state.algo = st.selectbox("Algorithme (Passage 1)", list(ALGO_MAP.keys()), index=list(ALGO_MAP.keys()).index(st.session_state.algo))
    st.session_state.config = st.selectbox("Scénario Capteurs", ('Télémétrie', 'Caméras', 'Centrale inertielle', 'Système radar doppler et optique', 'Coordonnées GPS (triangulation)'), index=0)
    
    # Échelle du curseur ajustée : 10ms (rapide) à 250ms (lent) pour respecter les contraintes de temps.
    st.session_state.delay_ms = st.slider("Vitesse Simulation (ms/step)", min_value=10, max_value=250, value=100, step=10)

    st.subheader("Positions S/E")
    
    s_x, s_y = st.session_state.start_pos
    e_x, e_y = st.session_state.end_pos

    # Affichage des coordonnées actuelles
    st.markdown(f"**Départ (S):** ({s_x}, {s_y})")
    st.markdown(f"**Arrivée (E):** ({e_x}, {e_y})")

    # NOUVEAUX BOUTONS DE SÉLECTION AU CLIC
    col_s, col_e = st.columns(2)
    
    with col_s:
        is_selecting_s = st.session_state.selection_mode == 'start'
        if st.button("🖱️ Choisir Départ", disabled=st.session_state.is_running, type="primary" if is_selecting_s else "secondary", use_container_width=True):
            st.session_state.selection_mode = 'start' if not is_selecting_s else None
            if st.session_state.selection_mode: st.rerun() 
    
    with col_e:
        is_selecting_e = st.session_state.selection_mode == 'end'
        if st.button("🖱️ Choisir Arrivée", disabled=st.session_state.is_running, type="primary" if is_selecting_e else "secondary", use_container_width=True):
            st.session_state.selection_mode = 'end' if not is_selecting_e else None
            if st.session_state.selection_mode: st.rerun() 
            
    if st.session_state.selection_mode:
        st.warning(f"Veuillez cliquer sur une case du labyrinthe pour choisir la position de **{'Départ' if st.session_state.selection_mode == 'start' else 'Arrivée'}**.")
        
    st.markdown("---") # Séparateur visuel
    
    st.subheader("Contrôles")
    if st.button("▶ Lancer la Simulation (3 Passages)"):
        st.session_state.selection_mode = None 
        run_simulation()
    
    if st.session_state.path_data_optimal:
        if st.button("💾 Sauvegarder Passage Final"):
            save_simulation()
    
    if st.button("🔄 Réinitialiser"):
        reset()

    # --- Affichage des Chronomètres ---
    st.markdown("### ⏱️ Chronomètres")
    
    if st.session_state.time_p1 > 0:
        st.text(f"P1 (Exploration): {st.session_state.time_p1:.3f} s")
    else:
        st.text("P1 (Exploration): 0.000 s")
        
    if st.session_state.time_p2 > 0:
        st.text(f"P2 (Calcul A*): {st.session_state.time_p2 * 1000:.3f} ms")
    else:
        st.text("P2 (Calcul A*): 0.000 ms")
        
    if st.session_state.total_time > 0 and st.session_state.current_passage >= 2:
        path_len = len(st.session_state.path_data_optimal['path'])
        visited_len = len(st.session_state.path_data_optimal['visited'])
        _, turn_count = get_total_time_path(st.session_state.path_data_optimal['path'])
        
        st.text(f"P3 (Simulé Optimal): {st.session_state.total_time:.3f} s")
        st.markdown("---")
        st.markdown("### 📊 Statistiques Optimales")
        st.text(f"Distance: {(path_len-1) * CELL_SIZE_M:.2f} m")
        st.text(f"Virages: {turn_count}")
        st.text(f"Exploré (P1): {visited_len} / {MAZE_SIZE*MAZE_SIZE} cases")
        st.text(f"Vitesse Robot: {ROBOT_SPEED_MPS:.3f} m/s")


# --- UI Principale ---

current_maze = load_current_maze() 

# -------------------------------------------------------------
# LOGIQUE DE LA GRILLE INTERACTIVE (SÉLECTION S/E ou CONSTRUCTION)
# -------------------------------------------------------------
is_interactive_grid_active = st.session_state.maze_name == 'Construction autonome' or st.session_state.selection_mode is not None

if is_interactive_grid_active:
    
    # ... (Code de la grille interactive inchangé) ...
    if st.session_state.maze_name == 'Construction autonome':
        st.title("🔨 Construction de Labyrinthe Autonome")
        st.markdown("Cliquez sur les cases pour basculer entre **Passage (0)** et **Mur (1)**. Utilisez les boutons dans la barre latérale pour sélectionner **Départ/Arrivée**.")
        current_grid_array = st.session_state.custom_maze_array.copy()
    else:
        st.title(f"Sélection de Position : {'Départ' if st.session_state.selection_mode == 'start' else 'Arrivée'}")
        st.markdown("Cliquez sur une case libre du labyrinthe pour définir la nouvelle position.")
        current_grid_array = current_maze.copy() # Afficher le labyrinthe prédéfini

    cols = st.columns([1] * MAZE_SIZE)

    for r in range(MAZE_SIZE):
        for c in range(MAZE_SIZE):
            cell_value = current_grid_array[r, c]
            cell_key = f"grid_cell_{r}_{c}"
            
            # Définir le contenu de la cellule
            is_start_pos = (r == st.session_state.start_pos[1] and c == st.session_state.start_pos[0])
            is_end_pos = (r == st.session_state.end_pos[1] and c == st.session_state.end_pos[0])
            is_start_end = is_start_pos or is_end_pos
            
            # Définir l'apparence
            if cell_value == 1: label = "🧱 Mur"; color = "#00ADB5"
            elif is_start_pos: label = "START"; color = "#EEEEEE"
            elif is_end_pos: label = "END"; color = "#F05454"
            else: label = "⬜ Passage"; color = "#222831"
            
            # Ajuster le type de bouton pour la sélection
            button_type = "primary" if (is_start_pos and st.session_state.selection_mode == 'start') or (is_end_pos and st.session_state.selection_mode == 'end') else "secondary"

            with cols[c]:
                # On utilise une taille/couleur cohérente avec le mode construction
                if st.button(label, key=cell_key, help=f"({c}, {r})", use_container_width=True, type=button_type):
                    
                    if st.session_state.selection_mode is None:
                        # Mode Construction Autonome: Bascule Mur/Passage (si ce n'est pas S/E)
                        if st.session_state.maze_name == 'Construction autonome' and not is_start_end:
                            st.session_state.custom_maze_array[r, c] = 1 if cell_value == 0 else 0
                            st.rerun() 
                    
                    elif cell_value != 1:
                        # Mode Sélection S/E: Mettre à jour la position si ce n'est pas un mur
                        if st.session_state.selection_mode == 'start':
                            st.session_state.start_pos = (c, r)
                        elif st.session_state.selection_mode == 'end':
                            st.session_state.end_pos = (c, r)
                            
                        st.session_state.selection_mode = None # Désactiver après la sélection
                        st.rerun()
                    else:
                        st.error("Impossible de choisir un Mur comme point de départ ou d'arrivée.")

    # Mise à jour du custom_maze_array après interaction (si on est en mode construction)
    if st.session_state.maze_name == 'Construction autonome':
        st.session_state.custom_maze_array[st.session_state.custom_maze_array == 2] = 0
        st.session_state.custom_maze_array[st.session_state.custom_maze_array == 3] = 0
        st.session_state.custom_maze_array[st.session_state.start_pos[1], st.session_state.start_pos[0]] = 2
        st.session_state.custom_maze_array[st.session_state.end_pos[1], st.session_state.end_pos[0]] = 3
# -------------------------------------------------------------
# FIN DE LA LOGIQUE DE LA GRILLE INTERACTIVE
# -------------------------------------------------------------


# Onglet Historique
with st.expander("📂 Historique des Simulations", expanded=False):
# ... (Logique d'affichage de l'historique conservée) ...
    if not st.session_state.history:
        st.info("Aucune simulation sauvegardée pour l'instant.")
    else:
        for i, sim in enumerate(reversed(st.session_state.history)):
            st.markdown(f"**{i+1}. {sim['timestamp']}** - **{sim['maze_name']}** / **{sim['algo']}**")
            st.text(f"Temps P1 (Exploration): {sim['time_p1']:.3f} s")
            st.text(f"Temps P3 (Simulé Optimal): {sim['total_time']:.3f} s")
            
            if st.button(f"Visualiser Passage {len(st.session_state.history) - i}", key=f"hist_viz_{i}"):
                st.session_state.temp_path_data = {'path': sim['path'], 'visited': set(sim['visited'])}
                st.session_state.temp_maze = np.array(sim['maze_array']) if sim['maze_array'] is not None else load_current_maze()
                st.session_state.temp_time = sim['total_time']
                st.session_state.temp_name = f"{sim['maze_name']} ({sim['timestamp']})"
            
            if 'temp_path_data' in st.session_state and st.session_state.temp_name == f"{sim['maze_name']} ({sim['timestamp']})":
                st.subheader(f"Visualisation de l'Historique: {st.session_state.temp_name}")
                # Affichage de l'historique toujours en mode "connu" (simulé P3)
                fig_hist = plot_maze_and_path(st.session_state.temp_maze, st.session_state.temp_path_data, len(st.session_state.temp_path_data['path']) - 1, highlight_path=sim['path'], current_stage="Passage 3/3: Exécution du Trajet Optimal")
                st.pyplot(fig_hist)
                st.info(f"Temps Chronométré (Simulé): {st.session_state.temp_time:.3f} s")
            
            st.markdown("---")


# Affichage principal du labyrinthe et des capteurs (UNIQUEMENT si la grille interactive n'est PAS active)
if not is_interactive_grid_active:
    
    st.title("Simulateur Labyrinthe Robotique 🤖")
    
    maze_col, graphs_col = st.columns((2, 1))

    with maze_col:
        st.subheader(f"Labyrinthe & Trajectoire ({st.session_state.stage})")
        
        path_data = {}
        path_to_display = []
        
        if st.session_state.current_passage == 1 and st.session_state.path_data_map:
            # P1: Afficher le chemin d'exploration DFS (chemin le plus long possible)
            path_data = st.session_state.path_data_map
            path_to_display = path_data['path']
            
        elif st.session_state.current_passage >= 2 and st.session_state.path_data_optimal:
            # P2/P3: Afficher le chemin optimal (chemin le plus court/rapide)
            path_data = st.session_state.path_data_optimal
            path_to_display = path_data['path']
        
        if not path_to_display:
            # Par défaut, juste la position de départ
            path_to_display = [(st.session_state.start_pos)]
            path_data = {'path': path_to_display, 'visited': {st.session_state.start_pos}}
        
        # Le labyrinthe affiché dépend du passage
        maze_to_plot = st.session_state.mapped_maze if st.session_state.current_passage >= 2 else current_maze
        
        # Afficher le graphique Matplotlib classique
        fig_maze = plot_maze_and_path(maze_to_plot, path_data, st.session_state.frame, highlight_path=path_to_display, current_stage=st.session_state.stage)
        st.pyplot(fig_maze)

        if st.session_state.current_passage == 3 and not st.session_state.is_running and st.session_state.path_data_optimal:
            st.success(f"🏁 **Passage 3 Terminé !** Temps Optimal Simulé: **{st.session_state.total_time:.3f} s**")
            
    with graphs_col:
        st.subheader("Analyse Odométrique des Capteurs")
        figs = plot_sensor_graphs()
        for f in figs:
            st.pyplot(f)

# --- Auto-play (Passage 3: Exécution Rapide) ---
if st.session_state.is_running:
    # On exécute uniquement le Passage 3
    if st.session_state.current_passage == 3 and st.session_state.path_data_optimal:
        
        path_len = len(st.session_state.path_data_optimal['path'])
        
        current_time = time.time()
        delay_s = st.session_state.delay_ms / 1000.0
        
        # Avancer si le temps écoulé est supérieur au délai demandé
        if current_time - st.session_state.last_update_time >= delay_s:
            
            if st.session_state.frame < path_len - 1:
                
                st.session_state.frame += 1
                st.session_state.last_update_time = current_time # Mettre à jour le temps de la dernière exécution

                # Le temps écoulé dans le graphique des capteurs doit être le temps simulé optimal
                # On utilise la fonction de coût pour obtenir le temps réel simulé à la frame actuelle
                t_current, _ = get_total_time_path(st.session_state.path_data_optimal['path'][:st.session_state.frame + 1])
                
                d1, d2, d3 = generate_sensor_data(current_maze, st.session_state.config, st.session_state.path_data_optimal, st.session_state.frame)
                st.session_state.sensor_data['capteur1'].append((t_current, d1))
                st.session_state.sensor_data['capteur2'].append((t_current, d2))
                st.session_state.sensor_data['capteur3'].append((t_current, d3))
                
                for k in st.session_state.sensor_data:
                    if len(st.session_state.sensor_data[k]) > 200:
                        st.session_state.sensor_data[k] = st.session_state.sensor_data[k][-200:]
                        
                st.rerun() 
                
            else:
                st.session_state.is_running = False
                st.session_state.stage = "Terminé"
                st.rerun()
        
        # Rerun constant pour l'animation Streamlit
        else:
            st.rerun()
