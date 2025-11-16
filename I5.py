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
    mazes['Complexe (Moyen)'] = np.array([
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], [1,2,1,0,0,0,1,0,0,0,1,0,0,0,0,1], [1,0,1,0,1,0,1,0,1,0,1,0,1,1,0,1], [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1],
        [1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,1], [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1], [1,0,1,1,1,1,1,1,1,0,1,1,1,1,0,1], [1,0,1,0,0,0,0,0,1,0,0,0,0,1,0,1],
        [1,0,1,0,1,1,1,0,1,1,1,1,0,1,0,1], [1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1], [1,1,1,0,1,0,1,1,1,1,1,1,1,1,0,1], [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,1,1,1,1,1,1,1,0,1,1,1,1,0,1], [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1], [1,1,1,1,1,0,1,1,1,1,1,0,1,1,3,1], [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    ])
    
    return mazes

# --- Fonctions d'exploration (Passage 1) ---

def dfs_search(maze, sx, sy):
    """Effectue une recherche en profondeur pour cartographier tout le labyrinthe accessible."""
    start_time = time.time() 
    
    mapped_maze = np.ones_like(maze) 
    rows, cols = maze.shape
    mapped_maze[sy, sx] = 0 
    
    stack = [((sx, sy), [(sx, sy)])]; visited = {(sx, sy)}
    path_history = [(sx, sy)] # Pour le tracé du P1
    
    while stack:
        (x, y), current_path = stack.pop()
        
        if maze[y][x] != 1:
            mapped_maze[y][x] = maze[y][x] if maze[y][x] in [2, 3] else 0
        
        # Mélanger les directions pour une exploration plus "réaliste" / moins prévisible
        directions_shuffled = DIRECTIONS.copy()
        random.shuffle(directions_shuffled) 
        
        for dy, dx in directions_shuffled:
            nx, ny = x + dx, y + dy
            
            if 0 <= ny < rows and 0 <= nx < cols:
                
                if maze[ny][nx] == 1:
                    mapped_maze[ny][nx] = 1 
                
                if maze[ny][nx] != 1 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    
                    # Tracer le mouvement d'exploration dans le chemin
                    path_history.append((nx, ny)) 
                    stack.append(((nx, ny), current_path + [(nx, ny)]))
    
    end_time = time.time()
    
    return {'path': path_history, 'visited': visited, 'time': end_time - start_time, 'mapped_maze': mapped_maze}

# --- Fonction Planification (Passage 2: A* optimisé) ---

def calculate_best_path(mapped_maze, start_pos, end_pos):
    """Calcule le chemin le plus rapide (temps minimal), pénalisant les virages via A*."""
    rows, cols = mapped_maze.shape
    sx, sy = start_pos; ex, ey = end_pos
    
    if mapped_maze[ey, ex] == 1: # Si l'arrivée n'est pas accessible/mappée (mur)
        return None, 0.0

    def get_cost(path):
        """Calcule le coût total (temps) d'un chemin, incluant la pénalité de virage."""
        if not path or len(path) <= 1: return 0.0, 0
        
        time_cost = (len(path) - 1) * TIME_PER_CELL_MAX_S
        turn_count = 0
        
        if len(path) > 1:
            for i in range(1, len(path) - 1):
                px, py = path[i-1]; cx, cy = path[i]; nx, ny = path[i+1]
                dx_curr = cx - px; dy_curr = cy - py
                dx_next = nx - cx; dy_next = ny - cy
                
                if dx_curr != dx_next or dy_curr != dy_next:
                    turn_count += 1
        
        time_cost += turn_count * TURN_PENALTY_S
        return time_cost, turn_count

    def heuristic(x, y):
        """Heuristique: Distance de Manhattan (estimation du temps minimum restant)."""
        return (abs(ex - x) + abs(ey - y)) * TIME_PER_CELL_MAX_S

    # Open set stocke: (f_score, g_score, x, y, path, last_dx, last_dy)
    open_set = [(heuristic(sx, sy), 0, sx, sy, [(sx, sy)], 0, 0)]
    # Clé: (x, y, last_dx, last_dy) -> g_score (temps réel)
    g_score_map = {(sx, sy, 0, 0): 0} 

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
            
            if 0 <= ny < rows and 0 <= nx < cols and mapped_maze[ny][nx] != 1:
                
                new_g = g + TIME_PER_CELL_MAX_S
                
                # Pénalisation des virages
                if len(path) > 1 and (dx != last_dx or dy != last_dy):
                    new_g += TURN_PENALTY_S
                
                key = (nx, ny, dx, dy)
                
                if new_g < g_score_map.get(key, float('inf')):
                    g_score_map[key] = new_g
                    
                    new_path = path + [(nx, ny)]
                    new_f = new_g + heuristic(nx, ny)
                    
                    heapq.heappush(open_set, (new_f, new_g, nx, ny, new_path, dx, dy))
                    
    return best_path, min_cost

def get_total_time_path(path):
    """Calcule le temps total et le nombre de virages pour un chemin donné."""
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


# --- Fonctions Utilitaires et Capteurs ---

def get_sensor_limits_and_units(config_key):
    # Les limites sont importantes pour le plotting
    if config_key == 'Télémétrie':
        return (0, 35, 'cm'), (0, 1, 'Probabilité'), (0, 500, 'Lux')
    elif config_key == 'Caméras':
        return (0, 255, 'Niveau de gris'), (0, 10, 'Pixels'), (-180, 180, 'Degrés')
    elif config_key == 'Centrale inertielle':
        return (-15, 15, 'm/s²'), (-180, 180, '°/s'), (0, 360, 'Degrés')
    elif config_key == 'Système radar doppler et optique':
        return (-20, 20, 'm/s'), (0, 100, 'dB'), (0, 1000, 'Lux')
    elif config_key == 'Coordonnées GPS (triangulation)':
        return (0, MAZE_SIZE * CELL_SIZE_M, 'm (X)'), (0, MAZE_SIZE * CELL_SIZE_M, 'm (Y)'), (0, 10, 'DOP (m)')
    else:
        return (0, 10, 'Valeur'), (0, 1, 'Valeur'), (0, 100, 'Valeur')

def get_distance_to_nearest_wall(maze, rx, ry):
    """Calcule la distance minimale à un mur ou à la limite dans les 4 directions."""
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
    """Génère les données des capteurs en fonction de la position du robot et de la configuration."""
    limits = get_sensor_limits_and_units(config_key)
    data = []
    
    path = path_data.get('path', [])
    if path and frame < len(path):
        rx, ry = path[frame]
        
        # Capteur 1 (ancré à la position/environnement)
        min_val, max_val, _ = limits[0]
        if config_key == 'Télémétrie':
            min_dist_cells = get_distance_to_nearest_wall(maze, rx, ry)
            # Normalisation et ajout de bruit
            base_val = np.clip(min_dist_cells * CELL_SIZE_M * 100, min_val + 5, max_val) # distance en cm
            noise = random.uniform(-0.1, 0.1) * (max_val / 4)
        elif config_key == 'Coordonnées GPS (triangulation)':
            base_val = np.clip(rx * CELL_SIZE_M, min_val, max_val)
            noise = random.uniform(-0.1, 0.1) # Bruit de localisation en mètres
        else:
            base_val = random.uniform(min_val, max_val)
            noise = random.uniform(-0.05, 0.05) * (max_val - min_val)
            
        val_1 = np.clip(base_val + noise, min_val, max_val)
        data.append(val_1)
            
    else:
        # Données aléatoires par défaut
        for min_val, max_val, _ in limits:
            data.append(random.uniform(min_val, max_val))
        return data[0], data[1], data[2]
    
    # Capteurs 2 et 3 (aléatoires avec bruit)
    for min_val, max_val, _ in limits[1:]:
        base_val = random.uniform(min_val, max_val)
        noise = random.uniform(-0.05, 0.05) * (max_val - min_val)
        val = base_val + noise
        data.append(np.clip(val, min_val, max_val))
        
    return data[0], data[1], data[2]

# --- Plotting ---

def plot_maze_and_path(maze, path_data, current_frame, highlight_path=None, current_stage="Prêt"):
    
    COLORS = {
        'Passage': '#222831',     
        'Mur': '#00ADB5',         
        'Départ': '#EEEEEE',      
        'Arrivée': '#F05454',     
        'Inconnu': '#171A1D',     
        'Visité': '#00ADB5',      
        'Trajet_Planifie': '#F05454', 
        'Robot_Actuel': '#FFD369'     
    }
    
    # Utilisation d'un fond gris foncé
    fig, ax = plt.subplots(figsize=(6.5, 6.5), facecolor='#393E46') 
    ax.set_facecolor('#393E46') 
    rows, cols = maze.shape

    is_exploring = current_stage.startswith("Passage 1") 
    
    # 1. Rendu du Labyrinthe (Murs/Passages)
    for yy in range(rows):
        for xx in range(cols):
            cell_val = maze[yy][xx]
            
            # Logique de Visibilité pour le Passage 1 (Labyrinthe Noir)
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
        for (x, y) in path_data['visited']:
            if maze[y][x] != 1: 
                 # Utiliser une couleur différente pour le passage visité
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
    """Trace les graphiques des données des capteurs."""
    
    # La fonction était vide ou incomplète dans la trace de l'erreur, la voici complète
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
        
        # S'assurer que les données existent et sont dans le bon format [(temps, valeur), ...]
        data = st.session_state.sensor_data.get(key, [])
        if data:
            times = [d[0] for d in data]; values = [d[1] for d in data]
            
            # Correction de la ligne de plotting : utiliser des formats simples
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
    """Charge le labyrinthe en cours et met à jour les positions S/E."""
    maze_key = st.session_state.maze_name
    
    if maze_key == 'Construction autonome':
        current_maze = st.session_state.custom_maze_array.copy()
    else:
        current_maze = st.session_state.mazes.get(maze_key, st.session_state.mazes['Classique (Facile)']).copy()
        
    # S'assurer que S/E sont mis à jour (le labyrinthe est réinitialisé à chaque chargement)
    current_maze[current_maze == 2] = 0 
    current_maze[current_maze == 3] = 0 
    
    s_x, s_y = st.session_state.start_pos
    e_x, e_y = st.session_state.end_pos

    if 0 <= s_y < MAZE_SIZE and 0 <= s_x < MAZE_SIZE:
        current_maze[s_y, s_x] = 2
    if 0 <= e_y < MAZE_SIZE and 0 <= e_x < MAZE_SIZE:
        current_maze[e_y, e_x] = 3
    
    return current_maze

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
if 'path_data_map' not in st.session_state: st.session_state.path_data_map = None 
if 'mapped_maze' not in st.session_state: st.session_state.mapped_maze = None 
if 'path_data_optimal' not in st.session_state: st.session_state.path_data_optimal = None 
if 'frame' not in st.session_state: st.session_state.frame = 0
if 'sensor_data' not in st.session_state: st.session_state.sensor_data = {'capteur1': [], 'capteur2': [], 'capteur3': []}
if 'is_running' not in st.session_state: st.session_state.is_running = False
if 'delay_ms' not in st.session_state: st.session_state.delay_ms = 100
if 'total_time' not in st.session_state: st.session_state.total_time = 0.0 
if 'time_p1' not in st.session_state: st.session_state.time_p1 = 0.0 
if 'time_p2' not in st.session_state: st.session_state.time_p2 = 0.0 
if 'history' not in st.session_state: st.session_state.history = []
if 'stage' not in st.session_state: st.session_state.stage = "Prêt"
if 'current_passage' not in st.session_state: st.session_state.current_passage = 0 
if 'selection_mode' not in st.session_state: st.session_state.selection_mode = None 
if 'last_update_time' not in st.session_state: st.session_state.last_update_time = time.time() 


ALGO_MAP = {
    'Wall following left-hand rule': 'wallFollowLeft',
    'Wall following right-hand rule': 'wallFollowRight',
    'Deep First Search': 'dfs',
    'Broad First Search': 'bfs', 
    'AI-based learning (A*)': 'aiLearning' 
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
        st.session_state.frame = 0 
        
        # On utilise DFS pour garantir l'exploration
        result = dfs_search(maze, sx, sy)
        st.session_state.path_data_map = result
        st.session_state.time_p1 = result['time']
        st.session_state.mapped_maze = result['mapped_maze']
        
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
        
        st.session_state.time_p2 = calc_end_time - calc_start_time 
        
        if best_path:
            st.session_state.path_data_optimal = {'path': best_path, 'visited': st.session_state.path_data_map.get('visited', set())}
            st.session_state.total_time = min_cost 
            # Passer directement au Passage 3
            run_simulation(start_passage=3)
        else:
            st.error("Le robot n'a pas pu trouver de chemin vers l'arrivée après l'exploration.")
            st.session_state.stage = "Échec du calcul"
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
        maze_array_to_save = st.session_state.custom_maze_array.tolist() if st.session_state.maze_name == 'Construction autonome' else load_current_maze().tolist()
        
        sim_data = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'maze_name': st.session_state.maze_name,
            'algo': st.session_state.algo,
            'config': st.session_state.config,
            'start_pos': st.session_state.start_pos,
            'end_pos': st.session_state.end_pos,
            'total_time': st.session_state.total_time, 
            'time_p1': st.session_state.time_p1, 
            'time_p2': st.session_state.time_p2, # Ajout du temps de calcul P2
            'path': st.session_state.path_data_optimal['path'],
            'visited': list(st.session_state.path_data_optimal['visited']),
            'maze_array': maze_array_to_save
        }
        st.session_state.history.append(sim_data)
        st.sidebar.success("Simulation sauvegardée !")

# --- UI Sidebar (Contrôles) ---

st.set_page_config(page_title="Simulateur Labyrinthe Robotique", layout="wide")

with st.sidebar:
    st.title("🚀 Contrôles du Robot")
    
    st.subheader("Configuration de la Course")
    maze_options = list(st.session_state.mazes.keys()) + ['Construction autonome']
    st.session_state.maze_name = st.selectbox("Labyrinthe", maze_options, index=maze_options.index(st.session_state.maze_name))
    st.session_state.algo = st.selectbox("Algorithme (Passage 1)", list(ALGO_MAP.keys()), index=2) # DFS par défaut pour l'exploration
    st.session_state.config = st.selectbox("Scénario Capteurs", ('Télémétrie', 'Caméras', 'Centrale inertielle', 'Système radar doppler et optique', 'Coordonnées GPS (triangulation)'), index=0)
    
    st.session_state.delay_ms = st.slider("Vitesse Simulation (ms/step)", min_value=10, max_value=250, value=100, step=10)

    st.subheader("Positions S/E")
    
    s_x, s_y = st.session_state.start_pos
    e_x, e_y = st.session_state.end_pos

    st.markdown(f"**Départ (S):** ({s_x}, {s_y})")
    st.markdown(f"**Arrivée (E):** ({e_x}, {e_y})")

    col_s, col_e = st.columns(2)
    
    with col_s:
        is_selecting_s = st.session_state.selection_mode == 'start'
        if st.button("🖱️ Choisir Départ", disabled=st.session_state.is_running, type="primary" if is_selecting_s else "secondary", use_container_width=True):
            st.session_state.selection_mode = 'start' if not is_selecting_s else None
            st.rerun() 
    
    with col_e:
        is_selecting_e = st.session_state.selection_mode == 'end'
        if st.button("🖱️ Choisir Arrivée", disabled=st.session_state.is_running, type="primary" if is_selecting_e else "secondary", use_container_width=True):
            st.session_state.selection_mode = 'end' if not is_selecting_e else None
            st.rerun() 
            
    if st.session_state.selection_mode:
        st.warning(f"Veuillez cliquer sur une case du labyrinthe pour choisir la position de **{'Départ' if st.session_state.selection_mode == 'start' else 'Arrivée'}**.")
        
    st.markdown("---") 
    
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
    
    st.text(f"P1 (Exploration): {st.session_state.time_p1:.3f} s")
    st.text(f"P2 (Calcul A*): {st.session_state.time_p2 * 1000:.3f} ms") # Affichage en ms
        
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
    
    if st.session_state.maze_name == 'Construction autonome':
        st.title("🔨 Construction de Labyrinthe Autonome")
        st.markdown("Cliquez sur les cases pour basculer entre **Passage (0)** et **Mur (1)**. Utilisez les boutons dans la barre latérale pour sélectionner **Départ/Arrivée**.")
        current_grid_array = st.session_state.custom_maze_array.copy()
    else:
        st.title(f"Sélection de Position : {'Départ' if st.session_state.selection_mode == 'start' else 'Arrivée'}")
        st.markdown("Cliquez sur une case libre du labyrinthe pour définir la nouvelle position.")
        current_grid_array = current_maze.copy()

    cols = st.columns([1] * MAZE_SIZE)

    for r in range(MAZE_SIZE):
        for c in range(MAZE_SIZE):
            cell_value = current_grid_array[r, c]
            cell_key = f"grid_cell_{r}_{c}"
            
            is_start_pos = (r == st.session_state.start_pos[1] and c == st.session_state.start_pos[0])
            is_end_pos = (r == st.session_state.end_pos[1] and c == st.session_state.end_pos[0])
            is_start_end = is_start_pos or is_end_pos
            
            if cell_value == 1: label = "🧱 Mur"; color = "#00ADB5"
            elif is_start_pos: label = "START"; color = "#EEEEEE"
            elif is_end_pos: label = "END"; color = "#F05454"
            else: label = "⬜ Passage"; color = "#222831"
            
            button_type = "primary" if (is_start_pos and st.session_state.selection_mode == 'start') or (is_end_pos and st.session_state.selection_mode == 'end') else "secondary"

            with cols[c]:
                if st.button(label, key=cell_key, help=f"({c}, {r})", use_container_width=True, type=button_type):
                    
                    if st.session_state.selection_mode is None:
                        # Mode Construction Autonome
                        if st.session_state.maze_name == 'Construction autonome' and not is_start_end:
                            st.session_state.custom_maze_array[r, c] = 1 if cell_value == 0 else 0
                            st.rerun() 
                    
                    elif cell_value != 1:
                        # Mode Sélection S/E
                        if st.session_state.selection_mode == 'start':
                            st.session_state.start_pos = (c, r)
                        elif st.session_state.selection_mode == 'end':
                            st.session_state.end_pos = (c, r)
                            
                        st.session_state.selection_mode = None 
                        st.rerun()
                    else:
                        st.error("Impossible de choisir un Mur comme point de départ ou d'arrivée.")

    if st.session_state.maze_name == 'Construction autonome':
        temp_maze = st.session_state.custom_maze_array.copy()
        temp_maze[temp_maze == 2] = 0
        temp_maze[temp_maze == 3] = 0
        temp_maze[st.session_state.start_pos[1], st.session_state.start_pos[0]] = 2
        temp_maze[st.session_state.end_pos[1], st.session_state.end_pos[0]] = 3
        st.session_state.custom_maze_array = temp_maze


# Onglet Historique
with st.expander("📂 Historique des Simulations", expanded=False):
    if not st.session_state.history:
        st.info("Aucune simulation sauvegardée pour l'instant.")
    else:
        # Affichage du dernier élément pour l'exemple
        for i, sim in enumerate(reversed(st.session_state.history)):
            st.markdown(f"**{i+1}. {sim['timestamp']}** - **{sim.get('maze_name', 'Inconnu')}** / **{sim.get('algo', 'Inconnu')}**")
            
            # Correction des KeyError potentielles ici
            st.text(f"Temps P1 (Exploration): {sim.get('time_p1', 0.0):.3f} s")
            st.text(f"Temps P3 (Simulé Optimal): {sim.get('total_time', 0.0):.3f} s")
            
            # Utilisation d'une clé d'état temporaire pour la visualisation
            temp_viz_key = f"hist_viz_active_{i}"
            
            if st.button(f"Visualiser Passage {len(st.session_state.history) - i}", key=f"hist_viz_btn_{i}"):
                # Stocker les données dans la session pour la visualisation
                st.session_state.temp_path_data = {'path': sim['path'], 'visited': set(sim['visited'])}
                # S'assurer que le labyrinthe est correctement chargé (utiliser le labyrinthe sauvegardé s'il existe)
                st.session_state.temp_maze = np.array(sim['maze_array']) if sim['maze_array'] is not None else load_current_maze()
                st.session_state.temp_time = sim['total_time']
                st.session_state.temp_name = f"{sim['maze_name']} ({sim['timestamp']})"
                st.session_state[temp_viz_key] = True 
                
            if st.session_state.get(temp_viz_key, False) and st.session_state.get('temp_name') == f"{sim['maze_name']} ({sim['timestamp']})":
                st.subheader(f"Visualisation de l'Historique: {st.session_state.temp_name}")
                fig_hist = plot_maze_and_path(
                    st.session_state.temp_maze, 
                    st.session_state.temp_path_data, 
                    len(st.session_state.temp_path_data['path']) - 1, 
                    highlight_path=sim['path'], 
                    current_stage="Passage 3/3: Exécution du Trajet Optimal" # Toujours affiché comme P3
                )
                st.pyplot(fig_hist)
                st.info(f"Temps Chronométré (Simulé): {st.session_state.temp_time:.3f} s")
                
            st.markdown("---")


# Affichage principal du labyrinthe et des capteurs
if not is_interactive_grid_active:
    
    st.title("Simulateur Labyrinthe Robotique 🤖")
    
    maze_col, graphs_col = st.columns((2, 1))

    with maze_col:
        st.subheader(f"Labyrinthe & Trajectoire ({st.session_state.stage})")
        
        path_data = {}
        path_to_display = []
        
        if st.session_state.current_passage == 1 and st.session_state.path_data_map:
            path_data = st.session_state.path_data_map
            path_to_display = path_data['path']
            
        elif st.session_state.current_passage >= 2 and st.session_state.path_data_optimal:
            path_data = st.session_state.path_data_optimal
            path_to_display = path_data['path']
        
        if not path_to_display:
            path_to_display = [(st.session_state.start_pos)]
            path_data = {'path': path_to_display, 'visited': {st.session_state.start_pos}}
        
        maze_to_plot = st.session_state.mapped_maze if st.session_state.current_passage >= 2 and st.session_state.mapped_maze is not None else current_maze
        
        fig_maze = plot_maze_and_path(maze_to_plot, path_data, st.session_state.frame, highlight_path=path_to_display, current_stage=st.session_state.stage)
        st.pyplot(fig_maze)

        if st.session_state.current_passage == 3 and not st.session_state.is_running and st.session_state.path_data_optimal:
            st.success(f"🏁 **Passage 3 Terminé !** Temps Optimal Simulé: **{st.session_state.total_time:.3f} s**")
            
    with graphs_col:
        st.subheader("Analyse Odométrique des Capteurs")
        # L'erreur de nom (NameError) devrait être corrigée par la présence de la fonction complète
        figs = plot_sensor_graphs()
        for f in figs:
            st.pyplot(f)

# --- Auto-play (Passage 3: Exécution Rapide) ---
if st.session_state.is_running:
    
    if st.session_state.current_passage == 3 and st.session_state.path_data_optimal:
        
        path_len = len(st.session_state.path_data_optimal['path'])
        
        current_time = time.time()
        delay_s = st.session_state.delay_ms / 1000.0
        
        if current_time - st.session_state.last_update_time >= delay_s:
            
            if st.session_state.frame < path_len - 1:
                
                st.session_state.frame += 1
                st.session_state.last_update_time = current_time 

                t_current, _ = get_total_time_path(st.session_state.path_data_optimal['path'][:st.session_state.frame + 1])
                
                d1, d2, d3 = generate_sensor_data(current_maze, st.session_state.config, st.session_state.path_data_optimal, st.session_state.frame)
                
                # Mise à jour des données des capteurs (limitation à 200 points pour la performance)
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
        
        else:
            # Rerun constant pour l'animation Streamlit
            st.rerun()
