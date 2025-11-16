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
# Temps = Distance / Vitesse = 0.2m / ROBOT_SPEED_MPS
TIME_PER_CELL_MAX_S = CELL_SIZE_M / ROBOT_SPEED_MPS 
# Pénalité de temps par virage (estimation : un robot s'arrête/ralentit, tourne, et réaccélère)
TURN_PENALTY_S = 0.5 # 500 ms par virage

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
    # Correction: Mettre le départ '2' dans les autres labyrinthes pour cohérence
    mazes['Complexe (Moyen)'] = np.array([
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], [1,2,1,0,0,0,1,0,0,0,1,0,0,0,0,1], [1,0,1,0,1,0,1,0,1,0,1,0,1,1,0,1], [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1],
        [1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,1], [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1], [1,0,1,1,1,1,1,1,1,0,1,1,1,1,0,1], [1,0,1,0,0,0,0,0,1,0,0,0,0,1,0,1],
        [1,0,1,0,1,1,1,0,1,1,1,1,0,1,0,1], [1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1], [1,1,1,0,1,0,1,1,1,1,1,1,1,1,0,1], [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,1,1,1,1,1,1,1,0,1,1,1,1,0,1], [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1], [1,1,1,1,1,0,1,1,1,1,1,0,1,1,3,1], [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    ])
    # ... autres labyrinthes (omission pour concision, mais ils seraient mis à jour avec le point de départ '2' aux coordonnées de départ par défaut)
    
    # Trouver le point de départ par défaut (coordonnée du '2')
    start_y, start_x = np.argwhere(mazes['Classique (Facile)'] == 2)[0]
    st.session_state.start_pos = (start_x, start_y)
    st.session_state.end_pos = (14, 14) # Coordonnée de l'arrivée par défaut
    
    return mazes

def get_start_end_pos(maze):
    # Chercher la position de départ (2) et d'arrivée (3)
    starts = np.argwhere(maze == 2)
    ends = np.argwhere(maze == 3)
    
    start_pos = (starts[0][1], starts[0][0]) if starts.size else (1, 1) # (x, y)
    end_pos = (ends[0][1], ends[0][0]) if ends.size else (MAZE_SIZE-2, MAZE_SIZE-2) # (x, y)
    
    return start_pos, end_pos

# --- Fonctions d'Algorithmes (Passage 1: Cartographie) ---
# Ces fonctions restent des versions simples pour le premier passage (exploration).

def wall_follow(maze, sx, sy, sh, right_hand=False):
    # Implémente la logique de "Wall following" pour la cartographie
    path = [(sx, sy)]; visited = set([(sx, sy)]); x, y, heading = sx, sy, sh
    rows, cols = maze.shape
    
    for _ in range(MAZE_SIZE * MAZE_SIZE * 5): # Limite pour éviter les boucles infinies
        if maze[y][x] == 3: break
        
        # Calculer le virage (gauche ou droite)
        turn = 1 if right_hand else -1
        
        # 1. Tenter de tourner vers le mur (main droite ou gauche)
        turn_heading = (heading + turn) % 4
        dy, dx = DIRECTIONS[turn_heading]
        turn_y, turn_x = y + dy, x + dx
        
        if 0 <= turn_y < rows and 0 <= turn_x < cols and maze[turn_y][turn_x] != 1:
            # Succès: on tourne et avance
            heading = turn_heading
            x, y = turn_x, turn_y
        else:
            # 2. Tenter d'avancer tout droit
            dy, dx = DIRECTIONS[heading]
            next_y, next_x = y + dy, x + dx
            
            if 0 <= next_y < rows and 0 <= next_x < cols and maze[next_y][next_x] != 1:
                # Succès: on avance tout droit
                x, y = next_x, next_y
            else:
                # 3. Tenter de tourner de l'autre côté
                heading = (heading - turn) % 4 # Tourner à l'opposé du mur
                continue
                
        path.append((x, y))
        visited.add((x, y))
        
    # Renvoie un chemin de (x, y) et les cellules visitées
    return {'path': path, 'visited': visited}

# Les autres algorithmes (DFS, BFS, Flood Fill, Tremaux, AI-based) sont utilisés comme des algorithmes de cartographie/exploration par défaut.
# La fonction `dfs_search` est un bon exemple d'exploration systématique (similaire au Tremaux pour le 1er passage).

def dfs_search(maze, sx, sy):
    stack = [((sx, sy), [(sx, sy)])]; visited = {(sx, sy)}
    
    # Le DFS est ici utilisé pour simuler l'exploration. Le chemin renvoyé est l'historique des pas.
    
    while stack:
        (x, y), path_history = stack.pop()
        
        if maze[y][x] == 3: 
            return {'path': path_history, 'visited': visited}
        
        # Visiter les voisins non visités
        for dy, dx in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= ny < MAZE_SIZE and 0 <= nx < MAZE_SIZE and maze[ny][nx] != 1 and (nx, ny) not in visited:
                visited.add((nx, ny))
                stack.append(((nx, ny), path_history + [(nx, ny)]))
                
    return {'path': [(sx, sy)], 'visited': visited}


# --- Fonctions d'Algorithmes (Passage 2: Planification du Chemin Optimal) ---

def calculate_best_path(mapped_maze, start_pos, end_pos):
    """
    Calcule le chemin optimal en minimisant la distance et le nombre de virages
    en utilisant l'algorithme A*.
    Le coût est: G = (Distance * TIME_PER_CELL_MAX_S) + (Nombre de Virages * TURN_PENALTY_S)
    H = Heuristique Manhattan (juste la distance en cases)
    """
    rows, cols = mapped_maze.shape
    sx, sy = start_pos
    ex, ey = end_pos

    def get_cost(path):
        """Calculer le coût total du chemin: temps + pénalité de virage."""
        time_cost = (len(path) - 1) * TIME_PER_CELL_MAX_S # -1 car le départ ne compte pas
        turn_cost = 0
        if len(path) > 1:
            # Calcul des virages
            dx_prev, dy_prev = None, None
            for i in range(1, len(path) - 1):
                px, py = path[i-1]
                cx, cy = path[i]
                nx, ny = path[i+1]
                
                # Déplacement actuel (vers le centre)
                dx_curr = cx - px
                dy_curr = cy - py
                
                # Déplacement suivant (depuis le centre)
                dx_next = nx - cx
                dy_next = ny - cy
                
                # S'il y a un changement de direction, c'est un virage
                if dx_curr != dx_next or dy_curr != dy_next:
                    turn_cost += TURN_PENALTY_S
        
        return time_cost + turn_cost

    def heuristic(x, y):
        # Heuristique: Distance Manhattan vers la fin
        return abs(ex - x) + abs(ey - y)

    # (f_score, g_score, x, y, path, last_dx, last_dy)
    # last_dx/dy: direction pour calculer les virages
    open_set = [(heuristic(sx, sy), 0, sx, sy, [(sx, sy)], 0, 0)]
    # Mémoriser les meilleurs g_scores par position pour éviter les cycles inutiles
    # Clé: (x, y), Valeur: g_score (temps total)
    g_score_map = {(sx, sy): 0}
    
    best_path = None
    min_cost = float('inf')

    while open_set:
        f, g, x, y, path, last_dx, last_dy = heapq.heappop(open_set)

        if x == ex and y == ey:
            # On a trouvé un chemin
            current_cost = get_cost(path)
            if current_cost < min_cost:
                min_cost = current_cost
                best_path = path
            continue # Continuer à chercher pour un chemin encore meilleur

        # Tentative d'exploration des voisins
        for dy, dx in DIRECTIONS:
            nx, ny = x + dx, y + dy
            
            if 0 <= ny < rows and 0 <= nx < cols and mapped_maze[ny][nx] != 1:
                
                # Calcul du nouveau coût G
                # Coût de base (temps de déplacement d'une case)
                new_g = g + TIME_PER_CELL_MAX_S
                
                # Pénalité de virage si la direction change
                if len(path) > 1:
                    # Le virage est calculé entre (path[-2]) -> (x, y) -> (nx, ny)
                    # La direction actuelle est (dx, dy)
                    if dx != last_dx or dy != last_dy:
                        new_g += TURN_PENALTY_S
                # Pour le premier pas, on considère que c'est un départ en ligne droite (pas de virage)
                
                if new_g < g_score_map.get((nx, ny), float('inf')):
                    g_score_map[(nx, ny)] = new_g
                    
                    new_path = path + [(nx, ny)]
                    new_f = new_g + heuristic(nx, ny)
                    
                    # Pour le prochain pas, notre direction actuelle (dx, dy) devient le last_dx/dy
                    heapq.heappush(open_set, (new_f, new_g, nx, ny, new_path, dx, dy))
                    
    return best_path, min_cost

# --- Fonctions Utilitaires ---

def generate_sensor_data(maze, config_key, path_data, frame):
    # Logique de génération de données de capteurs (inchangée)
    # ... (code omis pour concision, car la fonction reste la même que dans l'original)
    
    limits = get_sensor_limits_and_units(config_key)
    data = []
    
    if path_data and frame < len(path_data['path']):
        rx, ry = path_data['path'][frame]
        
        # Valeur 1 (souvent distance/position): Tentative d'ancrer une valeur au labyrinthe
        if config_key == 'Télémétrie':
            min_dist_cells = get_distance_to_nearest_wall(maze, rx, ry)
            base_val_1 = np.clip(min_dist_cells * 10, limits[0][0] + 5, limits[0][1])
            noise = random.uniform(-0.1, 0.1) * (limits[0][1] / 2)
            val_1 = np.clip(base_val_1 + noise, limits[0][0], limits[0][1])
            data.append(val_1)
        elif config_key == 'Coordonnées GPS (triangulation)':
            # Simuler la position X
            min_val, max_val, _ = limits[0]
            val_1 = np.clip(rx * (max_val / MAZE_SIZE) + random.uniform(-5, 5), min_val, max_val)
            data.append(val_1)
        else:
            # Génération aléatoire par défaut pour les autres scénarios
            min_val, max_val, _ = limits[0]
            base_val = random.uniform(min_val, max_val)
            noise = random.uniform(-0.05, 0.05) * (max_val - min_val)
            data.append(np.clip(base_val + noise, min_val, max_val))
            
    else:
        # Données de capteur aléatoires si hors chemin
        min_val, max_val, _ = limits[0]
        base_val = random.uniform(min_val, max_val)
        noise = random.uniform(-0.05, 0.05) * (max_val - min_val)
        data.append(np.clip(base_val + noise, min_val, max_val))
    
    # Génération aléatoire pour les valeurs 2 et 3
    for min_val, max_val, _ in limits[1:]:
        base_val = random.uniform(min_val, max_val)
        noise = random.uniform(-0.05, 0.05) * (max_val - min_val)
        val = base_val + noise
        data.append(np.clip(val, min_val, max_val))
        
    return data[0], data[1], data[2]

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

def get_sensor_limits_and_units(config_key):
    # Logique des capteurs (inchangée)
    if config_key == 'Télémétrie':
        return (0, 35, 'cm'), (0, 1, 'Probabilité'), (0, 500, 'Lux')
    elif config_key == 'Caméras':
        return (0, 255, 'Niveau de gris'), (0, 10, 'Pixels'), (-180, 180, 'Degrés')
    elif config_key == 'Centrale inertielle':
        return (-15, 15, 'm/s²'), (-180, 180, '°/s'), (0, 360, 'Degrés')
    elif config_key == 'Système radar doppler et optique':
        return (-20, 20, 'm/s'), (0, 100, 'dB'), (0, 1000, 'Lux')
    elif config_key == 'Coordonnées GPS (triangulation)':
        return (0, 1600*CELL_SIZE_M, 'm (X)'), (0, 1600*CELL_SIZE_M, 'm (Y)'), (0, 10, 'DOP (m)') # Mise à l'échelle
    else:
        return (0, 10, 'Valeur'), (0, 1, 'Valeur'), (0, 100, 'Valeur')

# --- Fonctions de Plotting (pour l'image réaliste) ---

def get_total_time_path(path):
    """Calcule le temps total de parcours du chemin (pour le chronomètre)."""
    if not path or len(path) <= 1:
        return 0.0
    
    total_time = (len(path) - 1) * TIME_PER_CELL_MAX_S # Temps de déplacement
    turn_count = 0
    
    if len(path) > 1:
        # Simuler le calcul du virage comme dans le Passage 2
        for i in range(1, len(path) - 1):
            px, py = path[i-1]
            cx, cy = path[i]
            nx, ny = path[i+1]
            
            dx_curr = cx - px
            dy_curr = cy - py
            dx_next = nx - cx
            dy_next = ny - cy
            
            if dx_curr != dx_next or dy_curr != dy_next:
                turn_count += 1
    
    total_time += turn_count * TURN_PENALTY_S
    return total_time, turn_count

def plot_maze_and_path(maze, path_data, current_frame, highlight_path=None):
    # Création d'une colormap professionnelle (inspirée de Google Material Design/Dark)
    # Couleurs du thème : Gris foncé, Bleu, Jaune, Rouge (comme dans le code original)
    colors = {0: '#222831', 1: '#00ADB5', 2: '#EEEEEE', 3: '#F05454'} # 0: Passage, 1: Mur, 2: Départ, 3: Arrivée
    
    # Créer une colormap pour la zone visitée (transition de gris très foncé à bleu clair)
    visited_color = 'lightblue'
    
    fig, ax = plt.subplots(figsize=(6.5, 6.5), facecolor='#393E46') # Fond de graphique
    ax.set_facecolor('#393E46') # Fond de l'aire de dessin

    for yy in range(MAZE_SIZE):
        for xx in range(MAZE_SIZE):
            color = colors.get(maze[yy][xx], colors[0])
            ax.add_patch(Rectangle((xx, yy), 1, 1, facecolor=color, edgecolor='#222831', linewidth=0.2))

    # Afficher les cellules visitées (Cartographie)
    if path_data and 'visited' in path_data:
        for (x, y) in path_data['visited']:
            # Utiliser une couleur claire et transparente
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor='#00ADB5', alpha=0.15, edgecolor='none'))

    # Dessiner la trajectoire du robot
    path = highlight_path if highlight_path is not None else path_data['path']
    if path:
        # Dessiner le chemin total
        path_x_full = [p[0] + 0.5 for p in path]
        path_y_full = [p[1] + 0.5 for p in path]
        ax.plot(path_x_full, path_y_full, color='#F05454', linestyle=':', linewidth=1.5, alpha=0.5, label='Chemin Optimal Planifié')
        
        # Dessiner le chemin parcouru jusqu'à la frame actuelle (Passage 3: Exécution)
        path_segment = path[:current_frame + 1]
        if len(path_segment) > 1:
            path_x = [p[0] + 0.5 for p in path_segment]
            path_y = [p[1] + 0.5 for p in path_segment]
            # Utiliser une ligne pleine épaisse pour l'exécution
            ax.plot(path_x, path_y, color='#FFD369', linestyle='-', linewidth=3, solid_capstyle='round', label='Parcours Actuel')
        
        # Marqueur Robot
        if path:
            idx = min(current_frame, len(path) - 1)
            rx, ry = path[idx]
            # Utiliser une couleur d'accentuation (Jaune/Or) pour le robot
            ax.plot(rx + 0.5, ry + 0.5, 'o', color='#FFD369', markersize=14, markeredgecolor='#EEEEEE', markeredgewidth=1.5, zorder=5, label='Robot')

    ax.set_xlim(0, MAZE_SIZE)
    ax.set_ylim(MAZE_SIZE, 0)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()
    return fig

# --- Fonctions Streamlit (Gestion de l'UI et de l'État) ---

# Définition des styles CSS pour un look professionnel
def apply_custom_style():
    st.markdown(
        """
        <style>
        /* Fond de page et conteneur principal */
        .stApp {
            background-color: #222831; /* Gris foncé */
            color: #EEEEEE; /* Blanc cassé */
        }
        /* Titres */
        h1, h2, h3, .css-1dp5qg, .css-1v3fvcr {
            color: #00ADB5 !important; /* Bleu turquoise */
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        /* Sidebar */
        .sidebar .sidebar-content {
            background-color: #393E46 !important; /* Gris moyen */
            color: #EEEEEE;
        }
        /* Boutons */
        .stButton>button {
            background-color: #F05454; /* Rouge/Orange */
            color: #EEEEEE;
            border: none;
            border-radius: 5px;
            padding: 10px 15px;
            font-weight: bold;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #CC4747;
        }
        /* Selectbox et autres inputs */
        .stSelectbox, .stNumberInput, .stSlider {
            color: #EEEEEE;
            background-color: #393E46;
            border-radius: 5px;
        }
        /* Statistiques */
        .stSidebar .markdown {
            color: #EEEEEE;
        }
        /* Message de Succès/Fin */
        div[data-testid="stSuccess"] {
            background-color: #17A962; /* Vert */
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

st.set_page_config(page_title="Simulateur Labyrinthe Robotique", layout="wide")
apply_custom_style()

# Initialize session state
if 'mazes' not in st.session_state:
    st.session_state.mazes = create_all_mazes()
if 'custom_maze_array' not in st.session_state:
    st.session_state.custom_maze_array = np.ones((MAZE_SIZE, MAZE_SIZE), dtype=int)
    st.session_state.custom_maze_array[1][1] = 2
    st.session_state.custom_maze_array[MAZE_SIZE-2][MAZE_SIZE-2] = 3
if 'maze_name' not in st.session_state:
    st.session_state.maze_name = 'Complexe (Moyen)'
if 'algo' not in st.session_state:
    st.session_state.algo = 'Deep First Search' # Préférer un algo d'exploration
if 'config' not in st.session_state:
    st.session_state.config = 'Télémétrie'
if 'start_pos' not in st.session_state:
    st.session_state.start_pos = (1, 1) # (x, y)
if 'end_pos' not in st.session_state:
    st.session_state.end_pos = (MAZE_SIZE-2, MAZE_SIZE-2) # (x, y)
if 'path_data_map' not in st.session_state:
    st.session_state.path_data_map = None # Passage 1: Cartographie
if 'path_data_optimal' not in st.session_state:
    st.session_state.path_data_optimal = None # Passage 2: Planification
if 'frame' not in st.session_state:
    st.session_state.frame = 0
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = {'capteur1': [], 'capteur2': [], 'capteur3': []}
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'delay_ms' not in st.session_state:
    st.session_state.delay_ms = 100
if 'total_time' not in st.session_state:
    st.session_state.total_time = 0.0
if 'history' not in st.session_state:
    st.session_state.history = []

# Helper: map UI algo label to function key
ALGO_MAP = {
    'Wall following left-hand rule': 'wallFollowLeft',
    'Wall following right-hand rule': 'wallFollowRight',
    'Deep First Search': 'dfs',
    'Broad First Search': 'bfs',
    'Tremaux': 'tremaux',
    'AI-based learning (A*)': 'aiLearning' 
    # Flood Fill n'est pas utilisé pour la cartographie
}

def load_current_maze():
    """Charge le labyrinthe en cours (pré-défini ou custom) et met à jour les positions S/E."""
    maze_key = st.session_state.maze_name
    if maze_key == 'Construction autonome':
        current_maze = st.session_state.custom_maze_array.copy()
    else:
        current_maze = st.session_state.mazes[maze_key].copy()
    
    # Mettre à jour les positions de départ et d'arrivée dans le labyrinthe chargé
    current_maze[current_maze == 2] = 0 # Nettoyer les anciens points de départ
    current_maze[current_maze == 3] = 0 # Nettoyer les anciens points d'arrivée
    current_maze[st.session_state.start_pos[1], st.session_state.start_pos[0]] = 2
    current_maze[st.session_state.end_pos[1], st.session_state.end_pos[0]] = 3
    
    return current_maze

def run_simulation():
    """Exécute les 3 passages complets de la simulation."""
    maze = load_current_maze()
    sx, sy = st.session_state.start_pos
    
    # ----------------------------------------------------
    # Passage 1: Cartographie / Exploration (Algo sélectionné)
    # ----------------------------------------------------
    st.session_state.stage = "Cartographie (Passage 1/3)"
    algo_key = ALGO_MAP.get(st.session_state.algo, 'dfs')
    default_heading = 1
    
    if algo_key == 'wallFollowLeft':
        st.session_state.path_data_map = wall_follow(maze, sx, sy, default_heading, right_hand=False)
    elif algo_key == 'wallFollowRight':
        st.session_state.path_data_map = wall_follow(maze, sx, sy, default_heading, right_hand=True)
    # On utilise DFS comme base pour l'exploration pour les autres (simule un balayage)
    elif algo_key in ['dfs', 'tremaux', 'bfs', 'aiLearning']: 
        st.session_state.path_data_map = dfs_search(maze, sx, sy) # DFS est une bonne simulation d'exploration
    
    # Labyrinthe cartographié (tout ce qui n'est pas 1 ou 0 est considéré comme passage)
    mapped_maze = maze.copy()
    
    # ----------------------------------------------------
    # Passage 2: Planification du Chemin Optimal
    # ----------------------------------------------------
    st.session_state.stage = "Planification (Passage 2/3)"
    best_path, min_cost = calculate_best_path(mapped_maze, st.session_state.start_pos, st.session_state.end_pos)
    
    # On stocke le chemin optimal pour l'exécution
    st.session_state.path_data_optimal = {'path': best_path, 'visited': st.session_state.path_data_map['visited']}
    st.session_state.total_time = min_cost
    
    # ----------------------------------------------------
    # Passage 3: Exécution Rapide (Initialisation)
    # ----------------------------------------------------
    st.session_state.stage = "Exécution Rapide (Passage 3/3)"
    st.session_state.frame = 0
    st.session_state.sensor_data = {'capteur1': [], 'capteur2': [], 'capteur3': []}
    st.session_state.is_running = True
    st.session_state.start_time = time.time()
    st.rerun() # Commencer l'exécution

def reset():
    """Réinitialise la simulation à l'état initial après un Solve/Apply."""
    st.session_state.frame = 0
    st.session_state.is_running = False
    st.session_state.path_data_map = None
    st.session_state.path_data_optimal = None
    st.session_state.sensor_data = {'capteur1': [], 'capteur2': [], 'capteur3': []}
    st.session_state.total_time = 0.0
    st.session_state.stage = "Prêt"

def save_simulation():
    """Sauvegarde les données de la simulation dans l'historique."""
    if st.session_state.path_data_optimal and st.session_state.total_time > 0:
        sim_data = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'maze_name': st.session_state.maze_name,
            'algo': st.session_state.algo,
            'config': st.session_state.config,
            'start_pos': st.session_state.start_pos,
            'end_pos': st.session_state.end_pos,
            'total_time': st.session_state.total_time,
            'path': st.session_state.path_data_optimal['path'],
            'visited': list(st.session_state.path_data_optimal['visited']),
            'maze_array': st.session_state.custom_maze_array.tolist() if st.session_state.maze_name == 'Construction autonome' else None
        }
        st.session_state.history.append(sim_data)
        st.sidebar.success("Simulation sauvegardée !")

# --- UI Sidebar (Contrôles) ---

with st.sidebar:
    st.title("🚀 Contrôles du Robot")
    
    # Labyrinthe et Algorithme
    st.subheader("Configuration de la Course")
    maze_options = list(st.session_state.mazes.keys()) + ['Construction autonome']
    st.session_state.maze_name = st.selectbox("Labyrinthe", maze_options, index=maze_options.index(st.session_state.maze_name))
    st.session_state.algo = st.selectbox("Algorithme (Passage 1)", list(ALGO_MAP.keys()), index=list(ALGO_MAP.keys()).index(st.session_state.algo))
    st.session_state.config = st.selectbox("Scénario Capteurs", ('Télémétrie', 'Caméras', 'Centrale inertielle', 'Système radar doppler et optique', 'Coordonnées GPS (triangulation)'), index=0)
    st.session_state.delay_ms = st.slider("Vitesse Simulation (ms/step)", min_value=10, max_value=1000, value=st.session_state.delay_ms, step=10)

    # Position de départ
    st.subheader("Positions S/E")
    
    current_maze = load_current_maze()
    s_x, s_y = st.session_state.start_pos
    e_x, e_y = st.session_state.end_pos

    # Nettoyer les murs avant l'input
    s_x = max(1, min(MAZE_SIZE - 2, s_x))
    s_y = max(1, min(MAZE_SIZE - 2, s_y))
    e_x = max(1, min(MAZE_SIZE - 2, e_x))
    e_y = max(1, min(MAZE_SIZE - 2, e_y))
    
    st.session_state.start_pos = (st.number_input("X Départ", min_value=1, max_value=MAZE_SIZE-2, value=int(s_x), step=1), 
                                  st.number_input("Y Départ", min_value=1, max_value=MAZE_SIZE-2, value=int(s_y), step=1))
    
    st.session_state.end_pos = (st.number_input("X Arrivée", min_value=1, max_value=MAZE_SIZE-2, value=int(e_x), step=1),
                                st.number_input("Y Arrivée", min_value=1, max_value=MAZE_SIZE-2, value=int(e_y), step=1))

    # Contrôles de simulation
    st.subheader("Contrôles")
    if st.button("▶ Lancer la Simulation (3 Passages)"):
        run_simulation()
    
    if st.session_state.path_data_optimal:
        if st.button("💾 Sauvegarder Passage Final"):
            save_simulation()
    
    if st.button("🔄 Réinitialiser"):
        reset()

# Affichage du Chronomètre
if st.session_state.total_time > 0 and st.session_state.path_data_optimal:
    # Calcul des statistiques du chemin optimal pour l'affichage
    path_len = len(st.session_state.path_data_optimal['path'])
    visited_len = len(st.session_state.path_data_optimal['visited'])
    _, turn_count = get_total_time_path(st.session_state.path_data_optimal['path'])
    
    st.sidebar.markdown("### 📊 Statistiques Optimales")
    st.sidebar.text(f"Robot Speed: {ROBOT_SPEED_MPS:.3f} m/s")
    st.sidebar.text(f"Distance: {(path_len-1) * CELL_SIZE_M:.2f} m")
    st.sidebar.text(f"Virages: {turn_count}")
    st.sidebar.text(f"Chronomètre: {st.session_state.total_time:.3f} s")
    st.sidebar.text(f"Exploré (Passage 1): {visited_len} / {MAZE_SIZE*MAZE_SIZE}")

# --- UI Principale ---

# Onglet pour la construction autonome
if st.session_state.maze_name == 'Construction autonome':
    st.title("🔨 Construction de Labyrinthe Autonome")
    st.markdown("Cliquez sur les cases pour basculer entre **Passage (0)** et **Mur (1)**. Départ (2) et Arrivée (3) sont contrôlés par les inputs dans la barre latérale.")
    
    # Création du tableau interactif
    cols = st.columns([1] * MAZE_SIZE)
    current_custom_maze = st.session_state.custom_maze_array.copy()

    for r in range(MAZE_SIZE):
        for c in range(MAZE_SIZE):
            cell_value = current_custom_maze[r, c]
            cell_key = f"cell_{r}_{c}"
            
            # Définir le contenu de la cellule
            if r == st.session_state.start_pos[1] and c == st.session_state.start_pos[0]:
                label = "START"
                color = "#EEEEEE"
            elif r == st.session_state.end_pos[1] and c == st.session_state.end_pos[0]:
                label = "END"
                color = "#F05454"
            elif cell_value == 1:
                label = "🧱 Mur"
                color = "#00ADB5"
            else:
                label = "⬜ Passage"
                color = "#222831"
            
            # Le bouton dans le layout du Streamlit
            with cols[c]:
                if st.button(label, key=cell_key, help=f"({c}, {r})", use_container_width=True):
                    # Ne pas modifier START/END
                    if not ((r == st.session_state.start_pos[1] and c == st.session_state.start_pos[0]) or 
                            (r == st.session_state.end_pos[1] and c == st.session_state.end_pos[0])):
                        # Basculer 0 <-> 1
                        st.session_state.custom_maze_array[r, c] = 1 if cell_value == 0 else 0
                        st.rerun() # Rafraîchir l'affichage
                        
    # Forcer la mise à jour des positions S/E dans le custom_maze_array
    st.session_state.custom_maze_array[st.session_state.custom_maze_array == 2] = 0
    st.session_state.custom_maze_array[st.session_state.custom_maze_array == 3] = 0
    st.session_state.custom_maze_array[st.session_state.start_pos[1], st.session_state.start_pos[0]] = 2
    st.session_state.custom_maze_array[st.session_state.end_pos[1], st.session_state.end_pos[0]] = 3


# Onglet Historique
with st.expander("📂 Historique des Simulations", expanded=False):
    if not st.session_state.history:
        st.info("Aucune simulation sauvegardée pour l'instant.")
    else:
        for i, sim in enumerate(reversed(st.session_state.history)):
            st.markdown(f"**{i+1}. {sim['timestamp']}** - **{sim['maze_name']}** / **{sim['algo']}**")
            st.text(f"Temps Final: {sim['total_time']:.3f} s")
            
            # Affichage de l'image du parcours optimal
            temp_maze = load_current_maze() if sim['maze_array'] is None else np.array(sim['maze_array'])
            
            if st.button(f"Visualiser Passage {len(st.session_state.history) - i}", key=f"hist_viz_{i}"):
                
                # Charger les données de l'historique dans l'état de session temporairement
                st.session_state.temp_path_data = {'path': sim['path'], 'visited': set(sim['visited'])}
                st.session_state.temp_maze = temp_maze
                st.session_state.temp_time = sim['total_time']
                st.session_state.temp_name = f"{sim['maze_name']} ({sim['timestamp']})"
            
            # Afficher la visualisation temporaire si sélectionnée
            if 'temp_path_data' in st.session_state and st.session_state.temp_name == f"{sim['maze_name']} ({sim['timestamp']})":
                st.subheader(f"Visualisation de l'Historique: {st.session_state.temp_name}")
                
                fig_hist = plot_maze_and_path(st.session_state.temp_maze, st.session_state.temp_path_data, len(st.session_state.temp_path_data['path']) - 1, highlight_path=sim['path'])
                st.pyplot(fig_hist)
                st.info(f"Temps Chronométré (Simulé): {st.session_state.temp_time:.3f} s")
            
            st.markdown("---")

# Affichage principal du labyrinthe et des capteurs (sauf en mode construction)
if st.session_state.maze_name != 'Construction autonome':
    
    st.title("Simulateur Labyrinthe Robotique 🤖")
    
    maze_col, graphs_col = st.columns((2, 1))

    with maze_col:
        # Affichage du statut
        stage_status = st.session_state.get('stage', 'Prêt')
        st.subheader(f"Labyrinthe & Trajectoire ({stage_status})")
        
        current_maze = load_current_maze()
        
        # Afficher la trajectoire optimale même si l'exécution n'est pas lancée
        path_to_display = st.session_state.path_data_optimal['path'] if st.session_state.path_data_optimal else st.session_state.path_data_map['path'] if st.session_state.path_data_map else None

        if path_to_display is None:
            # Si aucune simulation n'a encore été lancée, initialiser avec un chemin de base
            path_to_display = [(st.session_state.start_pos)]
            path_data = {'path': path_to_display, 'visited': {st.session_state.start_pos}}
        else:
            path_data = st.session_state.path_data_optimal if st.session_state.path_data_optimal else st.session_state.path_data_map
        
        # Plot du labyrinthe (utilise la fonction mise à jour)
        fig_maze = plot_maze_and_path(current_maze, path_data, st.session_state.frame, highlight_path=path_to_display)
        st.pyplot(fig_maze)

        if st.session_state.total_time > 0 and not st.session_state.is_running and st.session_state.path_data_optimal:
            # Chronomètre affiché après la fin de la simulation
            st.success(f"🏁 Course Terminée ! Temps Optimal Calculé: **{st.session_state.total_time:.3f} s**")
            
    with graphs_col:
        st.subheader("Analyse Odométrique des Capteurs")
        figs = plot_sensor_graphs()
        for f in figs:
            st.pyplot(f)

# --- Auto-play (Passage 3: Exécution Rapide) ---
if st.session_state.is_running:
    path_len = len(st.session_state.path_data_optimal['path'])
    
    if st.session_state.frame < path_len - 1:
        
        # Avancer d'un pas
        st.session_state.frame += 1
        
        # Mettre à jour les données de capteur (t est le temps simulé pour l'affichage)
        t_current = (st.session_state.frame - 1) * TIME_PER_CELL_MAX_S # Temps estimé
        
        d1, d2, d3 = generate_sensor_data(current_maze, st.session_state.config, st.session_state.path_data_optimal, st.session_state.frame)
        st.session_state.sensor_data['capteur1'].append((t_current, d1))
        st.session_state.sensor_data['capteur2'].append((t_current, d2))
        st.session_state.sensor_data['capteur3'].append((t_current, d3))
        
        # Nettoyer l'historique
        for k in st.session_state.sensor_data:
            if len(st.session_state.sensor_data[k]) > 200:
                st.session_state.sensor_data[k] = st.session_state.sensor_data[k][-200:]
                
        # Petit délai et relance
        time.sleep(st.session_state.delay_ms / 1000.0)
        st.rerun()
        
    else:
        # Fin de la simulation
        st.session_state.is_running = False
        st.session_state.stage = "Terminé"
        st.rerun()
