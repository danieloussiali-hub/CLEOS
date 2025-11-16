import time
import random
import heapq
from collections import deque
import json
import uuid

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg") # non-GUI backend suitable for Streamlit
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# --- Constantes et Configuration ---

# Dimensions physiques du labyrinthe et du robot
CELL_SIZE_M = 0.20 # 20 cm
ROBOT_SPEED_MS = 1000 / 3600 # 1 km/h en m/s (approx 0.2778 m/s)
TIME_PER_STEP_S = CELL_SIZE_M / ROBOT_SPEED_MS # Temps pour traverser une cellule (approx 0.72 s)
# Pénalité de temps pour un virage (estimée)
TURN_PENALTY_S = 0.25 # 250 ms par changement de direction (virage à 90 degrés)

# Directions de mouvement: (dy, dx, direction_label)
# N=0, E=1, S=2, W=3
DIRECTIONS = [(-1, 0, 0), (0, 1, 1), (1, 0, 2), (0, -1, 3)] 

# --- Maze & Algorithms (Adaptations pour les 3 phases) ---

def create_all_mazes():
    mazes = {}
    # Les labyrinthes existants sont conservés, ils seront écrasés si l'utilisateur choisit la construction autonome
    mazes['Classique (Facile)'] = np.array([
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1], [1,0,1,0,1,0,1,1,1,1,0,1,1,1,0,1], [1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1],
        [1,0,1,1,1,1,1,0,1,1,1,1,0,1,0,1], [1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1], [1,1,1,0,1,1,1,1,1,1,0,1,1,1,0,1], [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1], [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], [1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1], [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1], [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1], [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    ])
    # ... autres labyrinthes ...
    mazes['Complexe (Moyen)'] = np.array([
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], [1,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1], [1,0,1,0,1,0,1,0,1,0,1,0,1,1,0,1], [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1],
        [1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,1], [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1], [1,0,1,1,1,1,1,1,1,0,1,1,1,1,0,1], [1,0,1,0,0,0,0,0,1,0,0,0,0,1,0,1],
        [1,0,1,0,1,1,1,0,1,1,1,1,0,1,0,1], [1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1], [1,1,1,0,1,0,1,1,1,1,1,1,1,1,0,1], [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,1,1,1,1,1,1,1,0,1,1,1,1,0,1], [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1], [1,1,1,1,1,0,1,1,1,1,1,0,1,1,3,1], [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    ])
    mazes['Diabolique (Difficile)'] = np.array([
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], [1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1], [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1], [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1], [1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,1], [1,0,1,1,0,1,1,1,0,1,1,1,0,1,0,1], [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,1,0,1,1,1,1,1,1,1,1,1,1,0,1], [1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1], [1,1,1,1,1,0,1,0,1,0,1,1,0,1,0,1], [1,0,0,0,1,0,1,0,1,0,0,0,0,0,0,1],
        [1,0,1,0,1,0,1,0,1,1,1,1,1,1,0,1], [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1], [1,0,1,1,1,1,1,1,1,1,1,1,1,1,3,1], [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    ])
    mazes['Spirale (Expert)'] = np.array([
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], [1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1], [1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1],
        [1,0,1,0,1,1,1,1,1,1,1,1,0,1,0,1], [1,0,1,0,1,0,0,0,0,0,0,1,0,1,0,1], [1,0,1,0,1,0,1,1,1,1,0,1,0,1,0,1], [1,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1],
        [1,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1], [1,0,1,0,1,0,1,1,1,1,0,1,0,1,0,1], [1,0,1,0,1,0,0,0,0,0,0,1,0,1,0,1], [1,0,1,0,1,1,1,1,1,1,1,1,0,1,0,1],
        [1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1], [1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1], [1,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1], [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    ])
    mazes['Ouvert (Simple)'] = np.array([
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], [1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1], [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1], [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1], [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1], [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1], [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1], [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], [1,0,1,0,1,0,1,0,1,0,1,0,1,0,3,1], [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    ])
    return mazes

def wall_follow_left(maze, sx, sy, sh):
    """Passage 1: Mapping avec Wall Follow Left (nécessite une direction initiale)"""
    # ... (code inchangé pour le mapping) ...
    path = [[sx, sy]]; visited = set([(sx, sy)]); x, y, heading = sx, sy, sh
    for _ in range(5000): # Augmentation du max_steps pour le mapping
        left_heading = (heading - 1) % 4; dy, dx, _ = DIRECTIONS[left_heading]; left_y, left_x = y + dy, x + dx
        if 0 <= left_y < 16 and 0 <= left_x < 16 and maze[left_y][left_x] != 1:
            heading = left_heading; x, y = left_x, left_y
        else:
            dy, dx, _ = DIRECTIONS[heading]; next_y, next_x = y + dy, x + dx
            if 0 <= next_y < 16 and 0 <= next_x < 16 and maze[next_y][next_x] != 1: x, y = next_x, next_y
            else: heading = (heading + 1) % 4; continue
        path.append([x, y]); visited.add((x, y));
        if maze[y][x] == 3: break
    return {'path': path, 'visited': visited}

# Note: Tous les autres algorithmes (bfs, dfs, floodFill, tremaux, ai_based_learning) 
# sont adaptés pour servir de phase de 'Mapping' (passage 1) en renvoyant le chemin trouvé
# et l'ensemble des cases visitées. Pour un véritable mapping, Tremaux ou une version
# de DFS qui couvre tout l'espace ouvert serait plus approprié que BFS/A* qui s'arrêtent au but.

def floodFill(maze, sx, sy):
    """Utilisé pour le Passage 1 (Mapping) ou pour le Passage 2 (Optimisation)"""
    rows, cols = maze.shape
    distance_grid = np.full((rows, cols), -1)
    queue = deque()
    end_points = np.argwhere(maze == 3)
    if not end_points.size: return {'path': [[sx, sy]], 'visited': set([(sx, sy)])}
    for y_end, x_end in end_points:
        distance_grid[y_end, x_end] = 0
        queue.append((x_end, y_end))
    visited_cells = set()
    while queue:
        x, y = queue.popleft()
        visited_cells.add((x, y))
        current_dist = distance_grid[y, x]
        for dy, dx, _ in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= ny < rows and 0 <= nx < cols and maze[ny][nx] != 1 and distance_grid[ny, nx] == -1:
                distance_grid[ny, nx] = current_dist + 1
                queue.append((nx, ny))
                visited_cells.add((nx, ny))
    # Reconstruction du chemin le plus court (non optimisé pour les virages)
    if distance_grid[sy, sx] == -1: return {'path': [[sx, sy]], 'visited': visited_cells}
    path = [[sx, sy]]
    cx, cy = sx, sy
    while distance_grid[cy, cx] > 0:
        current_dist = distance_grid[cy, cx]
        found_next = False
        for dy, dx, _ in DIRECTIONS:
            nx, ny = cx + dx, cy + dy
            if 0 <= ny < rows and 0 <= nx < cols and distance_grid[ny, nx] == current_dist - 1:
                cx, cy = nx, ny
                path.append([cx, cy])
                found_next = True
                break
        if not found_next: break
    return {'path': path, 'visited': visited_cells}

# ... autres algos (dfs, bfs, tremaux, ai_based_learning) inchangés, ils servent de "Passage 1" ...

def optimal_path_search(maze, sx, sy, ex, ey):
    """
    Passage 2: Recherche de Chemin Optimal (A* avec coût de virage)
    
    Le coût de déplacement est (coût_distance + coût_virage).
    Le coût_distance est simple (1 par case).
    Le coût_virage est une pénalité ajoutée à chaque changement de direction.
    """
    rows, cols = maze.shape
    
    def heuristic(x, y):
        # Heuristique de Manhattan
        return abs(ex - x) + abs(ey - y)

    # Priority Queue: (f_score, g_score, x, y, path, heading)
    # g_score = coût réel (distance + pénalité de virage)
    # heading: 0=N, 1=E, 2=S, 3=W. -1 pour le début.
    
    # On force la première étape pour avoir une direction initiale
    
    # 1. Trouver tous les voisins initiaux possibles
    initial_moves = []
    for dir_idx, (dy, dx, heading) in enumerate(DIRECTIONS):
        nx, ny = sx + dx, sy + dy
        if 0 <= ny < rows and 0 <= nx < cols and maze[ny][nx] != 1:
            # Coût initial: 1 (distance) + 0 (pas de virage à l'étape 0)
            g = 1
            h = heuristic(nx, ny)
            f = g + h
            # (f_score, g_score, x, y, path_list, current_heading)
            initial_moves.append((f, g, nx, ny, [(sx, sy), (nx, ny)], heading))

    if not initial_moves:
        return {'path': [[sx, sy]], 'cost': 0}

    # Utilisation d'un dictionnaire pour suivre le meilleur g_score
    # Key: (x, y, heading)
    g_scores = {}
    for f, g, x, y, path, heading in initial_moves:
        g_scores[(x, y, heading)] = g

    # Initialisation de la file de priorité avec les mouvements initiaux
    open_set = initial_moves
    heapq.heapify(open_set)

    best_path = []
    min_cost = float('inf')

    while open_set:
        f, g, x, y, path, heading = heapq.heappop(open_set)

        # Si nous sommes arrivés, vérifions si c'est le meilleur chemin trouvé
        if x == ex and y == ey:
            if g < min_cost:
                min_cost = g
                best_path = path
            continue # Continuer pour voir si un autre chemin mène au même point avec un meilleur coût

        # Si le coût actuel est déjà plus grand que le meilleur coût trouvé, ignorer
        if g > g_scores.get((x, y, heading), float('inf')):
            continue

        for dir_idx, (dy, dx, next_heading) in enumerate(DIRECTIONS):
            nx, ny = x + dx, y + dy

            if 0 <= ny < rows and 0 <= nx < cols and maze[ny][nx] != 1:
                
                # Calcul de la pénalité de virage
                turn_cost_units = 0
                if next_heading != heading:
                    # Ajout d'une pénalité si la direction change
                    turn_cost_units = 1 # On ajoute 1 unité de coût équivalente à la pénalité de temps

                # Le coût de la nouvelle étape est (1 pour la distance + pénalité_virage)
                new_g = g + 1 + turn_cost_units
                new_h = heuristic(nx, ny)
                new_f = new_g + new_h
                
                # Check si le nouveau chemin est meilleur que le chemin existant
                if new_g < g_scores.get((nx, ny, next_heading), float('inf')):
                    g_scores[(nx, ny, next_heading)] = new_g
                    
                    new_path = path + [(nx, ny)]
                    heapq.heappush(open_set, (new_f, new_g, nx, ny, new_path, next_heading))
    
    # Conversion du chemin A* au format de sortie
    final_path = [[p[0], p[1]] for p in best_path]
    return {'path': final_path, 'cost': min_cost}

def calculate_time_and_steps(path_data):
    """Calcule le temps total et le nombre de virages pour le chemin"""
    path = path_data['path']
    if not path or len(path) <= 1:
        return 0.0, 0, 0
    
    time_total = 0.0
    turn_count = 0
    
    # 0=N, 1=E, 2=S, 3=W.
    # DICT_DIRECTIONS = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3} 
    heading = -1
    
    for i in range(len(path)):
        time_total += TIME_PER_STEP_S
        if i > 0:
            prev_x, prev_y = path[i-1]
            curr_x, curr_y = path[i]
            
            dx = curr_x - prev_x
            dy = curr_y - prev_y
            
            # Détermination de la direction actuelle
            current_heading = -1
            for idx, (dir_dy, dir_dx, dir_heading) in enumerate(DIRECTIONS):
                 if dir_dx == dx and dir_dy == dy:
                    current_heading = dir_heading
                    break
            
            if heading != -1 and current_heading != -1 and current_heading != heading:
                turn_count += 1
                time_total += TURN_PENALTY_S
            
            heading = current_heading
            
    # Le premier pas ne compte pas comme un virage, et le temps est déjà inclus dans la boucle
    return time_total, len(path) - 1, turn_count

# --- Sensor / Utilities ---

# ... get_sensor_limits_and_units() inchangé ...
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
        return (0, 1600, 'm (X)'), (0, 1600, 'm (Y)'), (0, 10, 'DOP (m)')
    else:
        return (0, 10, 'Valeur'), (0, 1, 'Valeur'), (0, 100, 'Valeur')

def get_distance_to_nearest_wall(maze, rx, ry):
    distances = []
    for dy, dx, _ in DIRECTIONS:
        d = 0
        curr_x, curr_y = rx, ry
        while True:
            curr_x += dx
            curr_y += dy
            d += 1
            if not (0 <= curr_y < 16 and 0 <= curr_x < 16) or maze[curr_y][curr_x] == 1:
                distances.append(d)
                break
    return min(distances) if distances else 0

def generate_sensor_data(maze, config_key, path_data, frame):
    # Génère des données de capteurs plus réalistes pour la Télémétrie pendant le mouvement
    limits = get_sensor_limits_and_units(config_key)
    data = []
    
    current_time = frame * 0.1 # Temps actuel (utilisé pour le bruit)
    
    if config_key == 'Télémétrie' and path_data and frame < len(path_data['path']):
        rx, ry = path_data['path'][frame]
        min_dist_cells = get_distance_to_nearest_wall(maze, rx, ry)
        # Convertir en cm (chaque cellule fait 20cm, mais le mur est à la limite)
        base_val_1 = np.clip(min_dist_cells * CELL_SIZE_M * 100 - (CELL_SIZE_M*100/2), limits[0][0] + 5, limits[0][1])
        noise = random.uniform(-1, 1) * 0.1 * limits[0][1] # Bruit relatif à l'échelle
        val_1 = np.clip(base_val_1 + noise, limits[0][0], limits[0][1])
        data.append(val_1)
    else:
        # Génération aléatoire si pas de télémétrie ou pas de mouvement
        min_val, max_val, _ = limits[0]
        base_val = (min_val + max_val) / 2 # Valeur centrale si pas de mouvement réel
        noise = random.uniform(-0.05, 0.05) * (max_val - min_val)
        data.append(np.clip(base_val + noise, min_val, max_val))
        
    # Remplir les autres capteurs aléatoirement (pour l'instant)
    for min_val, max_val, _ in limits[1:]:
        base_val = random.uniform(min_val, max_val)
        noise = random.uniform(-0.05, 0.05) * (max_val - min_val)
        val = base_val + noise
        data.append(np.clip(val, min_val, max_val))
        
    return data[0], data[1], data[2]

# --- Streamlit App UI & State Management ---

st.set_page_config(
    page_title="Simulateur de Robot Labyrinthe Pro", 
    layout="wide",
    # Thème sombre par défaut pour un look plus "pro"
    initial_sidebar_state="expanded" 
)

# Initialisation de l'état de la session
if 'mazes' not in st.session_state: st.session_state.mazes = create_all_mazes()
if 'maze_name' not in st.session_state: st.session_state.maze_name = 'Complexe (Moyen)'
if 'algo' not in st.session_state: st.session_state.algo = 'Flood Fill'
if 'config' not in st.session_state: st.session_state.config = 'Télémétrie'
if 'x_start' not in st.session_state: st.session_state.x_start = 1
if 'y_start' not in st.session_state: st.session_state.y_start = 1
if 'x_end' not in st.session_state: st.session_state.x_end = 14 # Nouvelle variable
if 'y_end' not in st.session_state: st.session_state.y_end = 14 # Nouvelle variable
if 'path_data' not in st.session_state: st.session_state.path_data = None
if 'frame' not in st.session_state: st.session_state.frame = 0
if 'sensor_data' not in st.session_state: st.session_state.sensor_data = {'capteur1': [], 'capteur2': [], 'capteur3': []}
if 'is_running' not in st.session_state: st.session_state.is_running = False
if 'delay_ms' not in st.session_state: st.session_state.delay_ms = 100
if 'sim_history' not in st.session_state: st.session_state.sim_history = [] # Historique des simulations
if 'total_time_s' not in st.session_state: st.session_state.total_time_s = 0.0
if 'current_maze_editor' not in st.session_state: 
    # Labyrinthe 16x16 vide (que des 0, murs externes à 1)
    st.session_state.current_maze_editor = np.ones((16, 16), dtype=int)
    st.session_state.current_maze_editor[1:-1, 1:-1] = 0
    st.session_state.current_maze_editor[14, 14] = 3
if 'current_phase' not in st.session_state: st.session_state.current_phase = 'Mapping'

# Helper: map UI algo label to function key
ALGO_MAP = {
    'Wall following left-hand rule': 'wallFollowLeft',
    'Wall following right-hand rule': 'wallFollowRight',
    'Flood Fill': 'floodFill',
    'Deep First Search': 'dfs',
    'Broad First Search': 'bfs',
    'AI-based learning (A*)': 'aiLearning',
    'Tremaux': 'tremaux'
}

# --- Fonctions de simulation des 3 phases ---

def execute_mapping_phase(maze, sx, sy):
    """Passage 1: Mapping du Labyrinthe."""
    st.session_state.current_phase = 'Mapping'
    st.session_state.delay_ms = 100 # Vitesse normale pour le mapping
    
    algo_key = ALGO_MAP.get(st.session_state.algo, 'wallFollowLeft')
    default_heading = 1 # Démarrer vers l'Est
    
    # Exécuter l'algorithme choisi pour le mapping
    if algo_key == 'wallFollowLeft':
        path_data = wall_follow_left(maze, sx, sy, default_heading)
    elif algo_key == 'wallFollowRight':
        path_data = wall_follow_right(maze, sx, sy, default_heading)
    elif algo_key == 'floodFill':
        path_data = floodFill(maze, sx, sy) # FloodFill cartographie tout même s'il est optimisé
    elif algo_key == 'dfs':
        path_data = dfs_search(maze, sx, sy)
    elif algo_key == 'bfs':
        path_data = bfs_search(maze, sx, sy)
    elif algo_key == 'tremaux':
        path_data = tremaux_algorithm(maze, sx, sy, default_heading)
    elif algo_key == 'aiLearning':
        # Pour A*, on utilise la fonction pour le mapping, mais l'objectif est d'explorer
        path_data = ai_based_learning(maze, sx, sy)
        
    return path_data

def execute_optimization_phase(maze, sx, sy, ex, ey):
    """Passage 2: Calcul du Chemin Optimal (minimisant virages)"""
    st.session_state.current_phase = 'Optimisation'
    # Pas de mouvement physique, seul le calcul est fait. Le résultat est le chemin optimal.
    
    # Le robot utilise les données du mapping (cells visitées) pour trouver le meilleur chemin
    # On utilise A* avec coût pondéré pour les virages.
    path_data = optimal_path_search(maze, sx, sy, ex, ey)
    
    return path_data

def execute_race_phase(maze, path_data, sx, sy):
    """Passage 3: Course Rapide (Chemin optimal)"""
    st.session_state.current_phase = 'Course'
    st.session_state.delay_ms = 10 # Vitesse très rapide pour la course
    
    # On utilise le chemin optimisé du passage 2
    # On recalcule le temps et les étapes pour la course finale
    time_total, steps, turns = calculate_time_and_steps(path_data)
    
    return path_data, time_total, steps, turns

def run_3_phases_simulation():
    """Fonction principale pour lancer la simulation complète"""
    st.session_state.is_running = True
    st.session_state.frame = 0
    st.session_state.sensor_data = {'capteur1': [], 'capteur2': [], 'capteur3': []}
    st.session_state.total_time_s = 0.0
    st.session_state.optimal_path = None
    
    maze = st.session_state.mazes[st.session_state.maze_name].copy()
    
    # Assurer que les coordonnées sont valides (et réelles pour le labyrinthe)
    sx = max(1, min(14, st.session_state.x_start))
    sy = max(1, min(14, st.session_state.y_start))
    ex = max(1, min(14, st.session_state.x_end))
    ey = max(1, min(14, st.session_state.y_end))
    
    # Réinitialisation des points de départ/arrivée
    if maze[sy, sx] == 1: sx, sy = 1, 1
    
    # 1. Mise à jour de la case d'arrivée dans le labyrinthe
    # On remet l'ancienne arrivée à 0
    maze_copy = st.session_state.mazes[st.session_state.maze_name].copy()
    end_old = np.argwhere(maze_copy == 3)
    if end_old.size > 0:
        maze[end_old[0][0], end_old[0][1]] = 0
    # On met la nouvelle arrivée à 3
    maze[ey, ex] = 3
    st.session_state.mazes[st.session_state.maze_name] = maze
    
    # --- PHASE 1: MAPPING ---
    with st.spinner(f"Phase 1/3: Mapping du labyrinthe avec {st.session_state.algo}..."):
        mapping_result = execute_mapping_phase(maze, sx, sy)
    
    # --- PHASE 2: OPTIMISATION (Calcul) ---
    with st.spinner(f"Phase 2/3: Calcul du chemin optimal (A* pondéré)..."):
        # On utilise le labyrinthe complet (post-mapping) pour l'optimisation
        # Le robot "connaît" maintenant les murs.
        optimization_result = execute_optimization_phase(maze, sx, sy, ex, ey)
    
    # --- PHASE 3: COURSE RAPIDE ---
    # Le chemin de la course est le chemin optimal trouvé
    st.session_state.optimal_path = optimization_result
    
    # Préparation du chemin pour l'animation et le chronomètre
    race_path_data = {'path': optimization_result['path'], 'visited': mapping_result['visited']} 
    race_result, total_time_s, steps_count, turn_count = execute_race_phase(maze, race_path_data, sx, sy)
    
    st.session_state.path_data = race_result
    st.session_state.total_time_s = total_time_s
    st.session_state.total_steps = steps_count
    st.session_state.total_turns = turn_count
    st.session_state.frame = 0 # Redémarre l'animation au début du chemin de course
    st.session_state.current_phase = 'Course'
    
    st.session_state.is_running = True # Démarre l'animation de la course (Passage 3)

def reset():
    """Réinitialise les paramètres de simulation"""
    st.session_state.frame = 0
    st.session_state.is_running = False
    st.session_state.sensor_data = {'capteur1': [], 'capteur2': [], 'capteur3': []}
    st.session_state.total_time_s = 0.0
    st.session_state.path_data = None
    st.session_state.optimal_path = None
    st.session_state.current_phase = 'Prêt'
    
    # Recharger les labyrinthes pour mettre à jour la case d'arrivée
    st.session_state.mazes = create_all_mazes()
    
# --- Fonctions de l'éditeur de labyrinthe ---
def update_maze_cell(row, col):
    """Fonction de rappel pour mettre à jour une cellule du labyrinthe édité"""
    current_val = st.session_state.current_maze_editor[row, col]
    
    # Gère le placement des murs (1) et des passages (0)
    if current_val == 1:
        st.session_state.current_maze_editor[row, col] = 0 # Clic sur un mur -> Passage
    else:
        # Clic sur un passage/arrivée -> Mur
        st.session_state.current_maze_editor[row, col] = 1

def set_start_cell(row, col):
    """Définit le point de départ"""
    if st.session_state.current_maze_editor[row, col] != 1:
        st.session_state.x_start = col
        st.session_state.y_start = row

def set_end_cell(row, col):
    """Définit la case d'arrivée (3)"""
    if st.session_state.current_maze_editor[row, col] != 1:
        # Remettre l'ancienne arrivée à 0
        old_ends = np.argwhere(st.session_state.current_maze_editor == 3)
        if old_ends.size > 0:
            st.session_state.current_maze_editor[old_ends[0][0], old_ends[0][1]] = 0
            
        st.session_state.current_maze_editor[row, col] = 3
        st.session_state.x_end = col
        st.session_state.y_end = row
        
def apply_custom_maze():
    """Applique le labyrinthe personnalisé à la simulation"""
    st.session_state.mazes['Construction autonome'] = st.session_state.current_maze_editor.copy()
    st.session_state.maze_name = 'Construction autonome'
    # Redéfinir l'état de départ/arrivée dans la session state
    st.session_state.x_start = st.session_state.x_start
    st.session_state.y_start = st.session_state.y_start
    end_cell = np.argwhere(st.session_state.current_maze_editor == 3)
    if end_cell.size > 0:
        st.session_state.x_end = end_cell[0][1]
        st.session_state.y_end = end_cell[0][0]
    st.session_state.path_data = None # Forcer le recalcul

def draw_maze_editor():
    """Dessine la grille pour l'éditeur de labyrinthe"""
    editor_maze = st.session_state.current_maze_editor
    
    st.markdown("### ✏️ Éditeur de Labyrinthe (16x16)")
    st.caption("Cliquez sur une case pour basculer Mur (gris foncé) / Passage (gris clair). Utilisez les boutons pour définir Départ/Arrivée.")
    
    # Création du tableau de boutons pour l'édition
    rows, cols = editor_maze.shape
    
    # On ignore la première et la dernière ligne/colonne (murs fixes)
    for r in range(1, rows - 1):
        cols_ui = st.columns(cols - 2)
        for c in range(1, cols - 1):
            cell_val = editor_maze[r, c]
            
            # Définir le style du bouton
            if cell_val == 1: button_label = "🧱"
            elif cell_val == 3: button_label = "🏁"
            elif c == st.session_state.x_start and r == st.session_state.y_start: button_label = "🟢"
            else: button_label = " "
                
            col_idx = c - 1
            with cols_ui[col_idx]:
                key_base = f"cell_{r}_{c}"
                
                # Le bouton principal bascule Mur/Passage
                if st.button(button_label, key=key_base, use_container_width=True):
                    update_maze_cell(r, c)
                    st.rerun()

                # Menu contextuel pour définir Départ/Arrivée
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    if st.button("Départ 🟢", key=key_base + "_start", use_container_width=True):
                        set_start_cell(r, c)
                        st.rerun()
                with col_c2:
                    if st.button("Arrivée 🏁", key=key_base + "_end", use_container_width=True):
                        set_end_cell(r, c)
                        st.rerun()
    
    st.button("Appliquer ce Labyrinthe", on_click=apply_custom_maze, type="primary")

# --- Fonctions de Plotting & Affichage ---

def plot_maze_and_path(phase='Course'):
    """Dessine le labyrinthe et la trajectoire (avec l'image réaliste)"""
    maze = st.session_state.mazes.get(st.session_state.maze_name)
    if maze is None:
        st.error("Labyrinthe non défini.")
        return plt.figure()

    # Création de la figure, style réaliste
    # Utilisation d'un fond plus boisé/sableux pour le look pro/compétition
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Couleur du fond (simulant la planche de bois)
    fig.set_facecolor('#d2b48c') 
    ax.set_facecolor('#f5deb3') 

    # Couleurs des cases
    colors = {
        0: '#f5deb3', # Passage (bois clair/sable)
        1: '#5a4f4d', # Mur (gris foncé/noir)
        3: '#ef4444'  # Arrivée (Rouge)
    }

    # Dessin de la grille
    for yy in range(16):
        for xx in range(16):
            color = colors.get(maze[yy][xx], '#f5deb3')
            # Dessin de la case (simulant l'espace ouvert)
            ax.add_patch(Rectangle((xx, yy), 1, 1, facecolor=color, edgecolor='#9ca3af', linewidth=0.5))
            
    # Marquage des cases visitées (Phase de Mapping)
    if st.session_state.path_data and phase == 'Course':
        # Le robot a "visité" ces cases durant la phase de mapping
        for (x, y) in st.session_state.path_data.get('visited', set()):
            # Visité (vert très clair)
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor='#6ee7b7', alpha=0.15, edgecolor='none'))

    # Traçage de la trajectoire (inspiré de l'image réelle)
    if st.session_state.path_data:
        path = st.session_state.path_data['path']
        frame = st.session_state.frame
        
        # Le tracé complet de la trajectoire optimale
        path_x_all = [p[0] + 0.5 for p in path]
        path_y_all = [p[1] + 0.5 for p in path]
        
        # Trajet en points/lignes (noir) pour un look réaliste de trace
        if len(path) > 1:
            ax.plot(path_x_all, path_y_all, 'k--', linewidth=1, alpha=0.5) # Tracé complet en pointillé noir
        
        # Tracé du segment parcouru (ligne plus épaisse)
        if frame > 0:
            path_segment = path[:frame + 1]
            if len(path_segment) > 1:
                path_x = [p[0] + 0.5 for p in path_segment]
                path_y = [p[1] + 0.5 for p in path_segment]
                ax.plot(path_x, path_y, '#1e40af', linewidth=3) # Trajet parcouru (bleu foncé)

        # Dessin du marqueur robot (Jaune)
        if path:
            idx = min(st.session_state.frame, len(path) - 1)
            rx, ry = path[idx]
            # Marqueur plus grand pour simuler le robot
            ax.plot(rx + 0.5, ry + 0.5, 'o', color='#fbbf24', markersize=24, markeredgecolor='black', markeredgewidth=2)
    
    # Marquage des points de départ et d'arrivée
    sx, sy = st.session_state.x_start, st.session_state.y_start
    ex, ey = st.session_state.x_end, st.session_state.y_end
    ax.plot(sx + 0.5, sy + 0.5, 'o', color='#22c55e', markersize=14, markeredgecolor='black', markeredgewidth=1) # Départ (Vert)
    ax.plot(ex + 0.5, ey + 0.5, 'X', color='#ef4444', markersize=14, markeredgecolor='black', markeredgewidth=1) # Arrivée (Rouge)

    ax.set_xlim(0, 16)
    ax.set_ylim(16, 0)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()
    return fig

def plot_sensor_graphs():
    """Dessine les 3 graphiques d'analyse odometrique"""
    # ... (code inchangé pour les graphiques de capteurs) ...
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
    
    # Streamlit est dans un thème sombre, donc les plots doivent être adaptés
    plt.rcParams.update({
        'text.color': 'white', 
        'axes.labelcolor': 'white', 
        'xtick.color': 'white', 
        'ytick.color': 'white',
        'axes.edgecolor': '#9ca3af',
        'figure.facecolor': '#0f172a',
        'axes.facecolor': '#1f2937'
    })
    
    for i in range(3):
        fig, ax = plt.subplots(figsize=(4, 1.8))
        key = f'capteur{i+1}'
        ax.set_title(titles.get(config_key, ("Capteur 1","Capteur 2","Capteur 3"))[i], fontsize=9)
        data = st.session_state.sensor_data.get(key, [])
        if data:
            times = [d[0] for d in data]
            values = [d[1] for d in data]
            ax.plot(times, values, ['#3b82f6','#10b981','#f97316'][i]+'-', linewidth=1)
        min_val, max_val, unit = limits_and_units[i]
        ax.set_ylim(min_val, max_val)
        ax.set_ylabel(unit, fontsize=8)
        ax.set_xlabel('Temps (s)', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        figs.append(fig)
    return figs

def save_simulation():
    """Sauvegarde les données de la simulation courante dans l'historique."""
    if st.session_state.path_data and st.session_state.total_time_s > 0.0:
        sim_id = str(uuid.uuid4())
        # Sauvegarde uniquement les données nécessaires et serialisables
        sim_data = {
            'id': sim_id,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'maze_name': st.session_state.maze_name,
            'algo': st.session_state.algo,
            'start': (st.session_state.x_start, st.session_state.y_start),
            'end': (st.session_state.x_end, st.session_state.y_end),
            'total_time_s': st.session_state.total_time_s,
            'steps': st.session_state.total_steps,
            'turns': st.session_state.total_turns,
            # Sauvegarder le labyrinthe pour l'historique
            'maze_data': st.session_state.mazes[st.session_state.maze_name].tolist(),
            # Sauvegarder le chemin optimal pour le re-visionnage
            'path': st.session_state.path_data['path'],
            'visited': list(st.session_state.path_data['visited']),
            'sensor_config': st.session_state.config
        }
        st.session_state.sim_history.insert(0, sim_data)
        st.success(f"Simulation sauvegardée: {sim_data['timestamp']}")
    else:
        st.warning("Aucune simulation complète à sauvegarder.")

def load_simulation(sim_data):
    """Charge une simulation de l'historique pour re-visionnage."""
    # Temporairement, on charge les données dans l'état pour l'affichage
    st.session_state.maze_name = sim_data['maze_name'] + ' (Historique)'
    st.session_state.algo = sim_data['algo']
    st.session_state.config = sim_data['sensor_config']
    st.session_state.x_start, st.session_state.y_start = sim_data['start']
    st.session_state.x_end, st.session_state.y_end = sim_data['end']
    st.session_state.total_time_s = sim_data['total_time_s']
    st.session_state.total_steps = sim_data['steps']
    st.session_state.total_turns = sim_data['turns']
    
    # Créer un labyrinthe temporaire dans la session state
    temp_maze_name = sim_data['maze_name'] + ' (Historique)'
    st.session_state.mazes[temp_maze_name] = np.array(sim_data['maze_data'])
    
    st.session_state.path_data = {
        'path': sim_data['path'], 
        'visited': set(sim_data['visited'])
    }
    st.session_state.frame = 0
    st.session_state.sensor_data = {'capteur1': [], 'capteur2': [], 'capteur3': []}
    st.session_state.is_running = False
    st.session_state.current_phase = 'Historique'
    st.rerun()

# --- Disposition de l'Interface ---

st.markdown("""
<style>
.stApp {
    background-color: #0f172a; /* Bleu foncé pour un look tech */
}
.css-hxt7xp { /* Header Streamlit */
    color: #cbd5e1;
}
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 16px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Simulateur de Robot Labyrinthe Non-Holonome (Pro)")

# Séparateur pour les onglets
tabs = st.tabs(["Contrôles du Robot et Labyrinthe", "Construction Autonome", "Historique des Simulations"])

with tabs[0]: # Onglet Contrôles
    # Disposition des contrôles
    controls_col, stats_col = st.columns([1, 2])
    
    with controls_col:
        st.subheader("⚙️ Configuration")
        
        # Sélection du Labyrinthe (avec l'option de construction autonome)
        maze_options = list(st.session_state.mazes.keys())
        maze_options.append("Construction autonome")
        st.session_state.maze_name = st.selectbox("Maze", maze_options, 
            index=maze_options.index(st.session_state.maze_name) if st.session_state.maze_name in maze_options else 0)
        
        st.session_state.algo = st.selectbox("Algorithme de Mapping (Passage 1)", list(ALGO_MAP.keys()), 
            index=list(ALGO_MAP.keys()).index(st.session_state.algo))
        
        st.session_state.config = st.selectbox("Scénario Capteurs", 
            ('Télémétrie', 'Caméras', 'Centrale inertielle', 'Système radar doppler et optique', 'Coordonnées GPS (triangulation)'), index=0)

        # Coordonnées modifiables
        st.markdown("---")
        st.subheader("📍 Coordonnées")
        col_x_start, col_y_start = st.columns(2)
        with col_x_start:
             st.session_state.x_start = st.number_input("X start", min_value=1, max_value=14, value=int(st.session_state.x_start), step=1)
        with col_y_start:
             st.session_state.y_start = st.number_input("Y start", min_value=1, max_value=14, value=int(st.session_state.y_start), step=1)
             
        col_x_end, col_y_end = st.columns(2)
        with col_x_end:
             st.session_state.x_end = st.number_input("X end", min_value=1, max_value=14, value=int(st.session_state.x_end), step=1)
        with col_y_end:
             st.session_state.y_end = st.number_input("Y end", min_value=1, max_value=14, value=int(st.session_state.y_end), step=1)
             
        st.markdown("---")
        
        # Bouton Lancer la simulation (3 phases)
        if st.button("▶ Lancer la Simulation (3 Phases)", type="primary", use_container_width=True):
            run_3_phases_simulation()
        
        # Bouton Sauvegarde
        if st.session_state.total_time_s > 0.0:
            if st.button("💾 Sauvegarder cette Course", use_container_width=True):
                save_simulation()
        
        st.session_state.delay_ms = st.slider("Vitesse d'Animation (ms/pas)", min_value=10, max_value=1000, value=st.session_state.delay_ms, step=10)
        
    with stats_col:
        st.subheader("📊 Résultats de Course")
        if st.session_state.total_time_s > 0.0:
            st.metric("Chronomètre (Total Course Rapide)", f"{st.session_state.total_time_s:.3f} s")
            st.metric("Nombre de Pas", st.session_state.total_steps)
            st.metric("Nombre de Virages", st.session_state.total_turns)
            
            # Calculer l'efficacité pour le Passage 1 (Mapping) si le chemin existe
            if st.session_state.path_data:
                 path_len = len(st.session_state.path_data['path'])
                 visited_len = len(st.session_state.path_data['visited'])
                 eff = (path_len / visited_len * 100) if visited_len else 0
                 st.markdown("---")
                 st.text(f"Algorithme de Mapping utilisé: {st.session_state.algo}")
                 st.text(f"Exploré (Mapping): {visited_len} / 256 ({visited_len/256*100:.1f}%)")
                 st.text(f"Chemin Optimal (Pas): {path_len}")
                 st.text(f"Efficacité Mapping (Path/Visited): {eff:.1f}%")
        else:
            st.info(f"Cliquez sur 'Lancer la Simulation' pour démarrer les 3 phases. Phase actuelle: **{st.session_state.current_phase}**")

    st.markdown("---")
    
    # Affichage principal du Labyrinthe et des Graphes
    maze_col, graphs_col = st.columns((2, 1))

    with maze_col:
        st.header("🗺️ Labyrinthe (Passage 3: Course Optimale)")
        # Forcer le rechargement de la figure si le labyrinthe a changé
        if st.session_state.path_data is None:
            reset()
        fig_maze = plot_maze_and_path(phase=st.session_state.current_phase)
        st.pyplot(fig_maze)

    with graphs_col:
        st.header("📈 Analyse Odométrique Holistique")
        st.subheader(f"Capteurs: {st.session_state.config}")
        figs = plot_sensor_graphs()
        for f in figs:
            st.pyplot(f)

with tabs[1]: # Onglet Construction Autonome
    if st.session_state.maze_name == 'Construction autonome':
        draw_maze_editor()
    else:
        st.info("Sélectionnez 'Construction autonome' dans l'onglet 'Contrôles du Robot' pour accéder à l'éditeur.")
        if st.button("Passer à la Construction Autonome", type="primary"):
            st.session_state.maze_name = 'Construction autonome'
            st.rerun()

with tabs[2]: # Onglet Historique
    st.header("🕰️ Historique des Simulations Sauvegardées")
    if not st.session_state.sim_history:
        st.info("Aucune simulation n'a été sauvegardée. Lancez une simulation complète et cliquez sur 'Sauvegarder'.")
    else:
        for sim in st.session_state.sim_history:
            expander = st.expander(f"[{sim['timestamp']}] - Maze: {sim['maze_name']} - Algo: {sim['algo']} - Temps: {sim['total_time_s']:.3f} s")
            with expander:
                st.json({
                    'Départ': f"({sim['start'][0]}, {sim['start'][1]})",
                    'Arrivée': f"({sim['end'][0]}, {sim['end'][1]})",
                    'Pas / Virages': f"{sim['steps']} / {sim['turns']}",
                    'Configuration Capteurs': sim['sensor_config']
                })
                if st.button(f"Re-visionner (ID: {sim['id'][-4:]})", key=f"load_{sim['id']}"):
                    load_simulation(sim)
                

# Auto-play handling: avance d'un pas par run (uniquement pour la Phase 3: Course)
if st.session_state.is_running and st.session_state.current_phase == 'Course':
    path = st.session_state.path_data['path']
    if path and st.session_state.frame < len(path) - 1:
        st.session_state.frame += 1
        
        # Le temps de capteur est basé sur le pas (0.1s/pas) pour l'animation
        t = st.session_state.frame * 0.1 
        
        # Génération des données capteurs à chaque pas de l'animation
        d1, d2, d3 = generate_sensor_data(st.session_state.mazes[st.session_state.maze_name], st.session_state.config, st.session_state.path_data, st.session_state.frame)
        st.session_state.sensor_data['capteur1'].append((t, d1))
        st.session_state.sensor_data['capteur2'].append((t, d2))
        st.session_state.sensor_data['capteur3'].append((t, d3))
        
        # Garder l'historique des capteurs limité pour la performance
        for k in st.session_state.sensor_data:
            if len(st.session_state.sensor_data[k]) > 200:
                st.session_state.sensor_data[k] = st.session_state.sensor_data[k][-200:]
                
        # Petit délai basé sur le slider
        time.sleep(st.session_state.delay_ms / 1000.0)
        st.rerun()
    else:
        st.session_state.is_running = False
        if st.session_state.current_phase == 'Course':
            st.success(f"🏁 **COURSE TERMINÉE !** Temps total simulé: **{st.session_state.total_time_s:.3f} secondes**.")
        st.rerun()
