import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from collections import deque
import tkinter as tk
from tkinter import ttk
import random
import heapq 

class MazeSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulateur de Labyrinthe et Analyse Odométrique")
        
        # --- Labyrinthe (Base) ---
        self.maze = None 
        self.directions = [(-1, 0), (0, 1), (1, 0), (0, -1)] # N, E, S, O
        
        # --- Variables de l'interface ---
        self.x_start = tk.IntVar(value=1)
        self.y_start = tk.IntVar(value=1)
        self.algo_var = tk.StringVar(value='Wall following left-hand rule')
        self.maze_var = tk.StringVar(value='Complexe (Moyen)')
        self.config_var = tk.StringVar(value='Télémétrie')
        
        self.mazes = self.create_all_mazes()
        
        # --- Variables de Simulation/Animation ---
        self.is_running = False
        self.frame = 0
        self.path_data = None
        self.sensor_data = {'capteur1': [], 'capteur2': [], 'capteur3': []} 
        self.default_delay_ms = 100
        
        # --- Configuration des Graphiques (MIS À JOUR) ---
        self.sensor_titles = {
            'Télémétrie': ("Distance Proche (Ultrason)", "Modèle Probabiliste (Pression)", "Scan Laser (Balayage)"),
            'Caméras': ("Luminosité (Simple)", "Disparité (Stéréoscopie)", "Angle Panoramique"),
            'Centrale inertielle': ("Force X (Accéléromètre)", "Vitesse Angulaire Z (Gyroscope)", "Orientation (Magnétomètre)"),
            'Système radar doppler et optique': ("Vitesse Rel. (Doppler)", "Amplitude (Radar)", "Luminosité (Optique)"),
            'Coordonnées GPS (triangulation)': ("X (Localisation)", "Y (Localisation)", "Erreur (Dilution)"),
        }
        
        self.setup_ui()
        self.load_maze()
        self.solve_maze()

    # --- Méthodes de Labyrinthe et Algorithmes ---
    def create_all_mazes(self):
        mazes = {}
        mazes['Classique (Facile)'] = np.array([
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1], [1,0,1,0,1,0,1,1,1,1,0,1,1,1,0,1], [1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1],
            [1,0,1,1,1,1,1,0,1,1,1,1,0,1,0,1], [1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1], [1,1,1,0,1,1,1,1,1,1,0,1,1,1,0,1], [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1], [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], [1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1], [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1], [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1], [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        ])
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
    
    def load_maze(self):
        maze_name = self.maze_var.get()
        self.maze = self.mazes[maze_name].copy()
        self.solve_maze()

    def get_algo_key(self):
        algo_map = {
            'Wall following left-hand rule': 'wallFollowLeft', 'Wall following right-hand rule': 'wallFollowRight',
            'Flood Fill': 'floodFill', 'Deep First Search': 'dfs', 'Broad First Search': 'bfs',
            'AI-based learning': 'aiLearning', 'Tremaux': 'tremaux'
        }
        return algo_map.get(self.algo_var.get(), 'wallFollowLeft')

    def wall_follow_left(self, sx, sy, sh):
        path = [[sx, sy]]; visited = set([(sx, sy)]); x, y, heading = sx, sy, sh
        for _ in range(500):
            left_heading = (heading - 1) % 4; dy, dx = self.directions[left_heading]; left_y, left_x = y + dy, x + dx
            if 0 <= left_y < 16 and 0 <= left_x < 16 and self.maze[left_y][left_x] != 1:
                heading = left_heading; x, y = left_x, left_y
            else:
                dy, dx = self.directions[heading]; next_y, next_x = y + dy, x + dx
                if 0 <= next_y < 16 and 0 <= next_x < 16 and self.maze[next_y][next_x] != 1: x, y = next_x, next_y
                else: heading = (heading + 1) % 4; continue
            path.append([x, y]); visited.add((x, y));
            if self.maze[y][x] == 3: break
        return {'path': path, 'visited': visited}
    
    def wall_follow_right(self, sx, sy, sh):
        path = [[sx, sy]]; visited = set([(sx, sy)]); x, y, heading = sx, sy, sh
        for _ in range(500):
            right_heading = (heading + 1) % 4; dy, dx = self.directions[right_heading]; right_y, right_x = y + dy, x + dx
            if 0 <= right_y < 16 and 0 <= right_x < 16 and self.maze[right_y][right_x] != 1:
                heading = right_heading; x, y = right_x, right_y
            else:
                dy, dx = self.directions[heading]; next_y, next_x = y + dy, x + dx
                if 0 <= next_y < 16 and 0 <= next_x < 16 and self.maze[next_y][next_x] != 1: x, y = next_x, next_y
                else: heading = (heading - 1) % 4; continue
            path.append([x, y]); visited.add((x, y));
            if self.maze[y][x] == 3: break
        return {'path': path, 'visited': visited}

    def bfs_search(self, sx, sy):
        queue = deque([((sx, sy), [(sx, sy)])]); visited = {(sx, sy)}
        while queue:
            (x, y), path = queue.popleft()
            if self.maze[y][x] == 3: return {'path': [[p[0], p[1]] for p in path], 'visited': visited}
            for dy, dx in self.directions:
                nx, ny = x + dx, y + dy
                if 0 <= ny < 16 and 0 <= nx < 16 and self.maze[ny][nx] != 1 and (nx, ny) not in visited:
                    visited.add((nx, ny)); queue.append(((nx, ny), path + [(nx, ny)]))
        return {'path': [[sx, sy]], 'visited': visited}
    
    def dfs_search(self, sx, sy):
        stack = [((sx, sy), [(sx, sy)])]; visited = {(sx, sy)}
        while stack:
            (x, y), path = stack.pop()
            if self.maze[y][x] == 3: return {'path': [[p[0], p[1]] for p in path], 'visited': visited}
            for dy, dx in self.directions:
                nx, ny = x + dx, y + dy
                if 0 <= ny < 16 and 0 <= nx < 16 and self.maze[ny][nx] != 1 and (nx, ny) not in visited:
                    visited.add((nx, ny)); stack.append(((nx, ny), path + [(nx, ny)]))
        return {'path': [[sx, sy]], 'visited': visited}
    
    def floodFill(self, sx, sy):
        """
        Implémente l'algorithme Flood Fill (Water-Fill / Reverse BFS) pour trouver le chemin le plus court 
        et génère le chemin continu en descendant la grille de coût.
        """
        rows, cols = self.maze.shape
        # Étape 1 : Remplissage de la grille de coût (Backward Pass)
        distance_grid = np.full((rows, cols), -1)
        queue = deque()
        
        # Trouver la ou les sorties (valeur 3) et initialiser la distance à 0
        end_points = np.argwhere(self.maze == 3)
        if not end_points.size:
            return {'path': [[sx, sy]], 'visited': set([(sx, sy)])}

        for y_end, x_end in end_points:
            distance_grid[y_end, x_end] = 0
            queue.append((x_end, y_end))

        visited_cells = set()

        while queue:
            x, y = queue.popleft()
            visited_cells.add((x, y))
            
            # Distance actuelle
            current_dist = distance_grid[y, x]

            # Vérifier les voisins
            for dy, dx in self.directions:
                nx, ny = x + dx, y + dy
                
                # Vérifie si la case est valide (dans les limites, pas un mur) et non visitée
                if 0 <= ny < rows and 0 <= nx < cols and self.maze[ny][nx] != 1 and distance_grid[ny, nx] == -1:
                    distance_grid[ny, nx] = current_dist + 1
                    queue.append((nx, ny))
                    visited_cells.add((nx, ny))
        
        # Étape 2 : Détermination du chemin (Forward Pass)
        if distance_grid[sy, sx] == -1:
             return {'path': [[sx, sy]], 'visited': visited_cells}

        path = [[sx, sy]]
        cx, cy = sx, sy
        
        while distance_grid[cy, cx] > 0:
            current_dist = distance_grid[cy, cx]
            found_next = False
            
            # Recherche du voisin ayant une distance exactement -1
            for dy, dx in self.directions:
                nx, ny = cx + dx, cy + dy
                
                if 0 <= ny < rows and 0 <= nx < cols and distance_grid[ny, nx] == current_dist - 1:
                    # On se déplace vers cette case
                    cx, cy = nx, ny
                    path.append([cx, cy])
                    found_next = True
                    break
            
            if not found_next:
                # Blocage inattendu, arrête la recherche
                break 

        return {'path': path, 'visited': visited_cells}
    
    def tremaux_algorithm(self, sx, sy, sh):
        path = [[sx, sy]]; visited = {}; visited[(sx, sy)] = 1; x, y, heading = sx, sy, sh
        for _ in range(500):
            if self.maze[y][x] == 3: break
            neighbors = [];
            for idx, (dy, dx) in enumerate(self.directions):
                nx, ny = x + dx, y + dy
                if 0 <= ny < 16 and 0 <= nx < 16 and self.maze[ny][nx] != 1:
                    count = visited.get((nx, ny), 0); neighbors.append((nx, ny, idx, count))
            if not neighbors: break
            neighbors.sort(key=lambda n: n[3]); 
            
            x, y, heading = neighbors[0][0], neighbors[0][1], neighbors[0][2] 
            
            visited[(x, y)] = visited.get((x, y), 0) + 1; path.append([x, y])
            
        return {'path': path, 'visited': set(visited.keys())}
    
    def ai_based_learning(self, sx, sy):
        # A* Search (Simple Heuristic)
        def heuristic(x, y): return abs(14 - x) + abs(14 - y)
        open_set = [(heuristic(sx, sy), 0, sx, sy, [(sx, sy)])]; visited = {(sx, sy)}
        while open_set:
            _, g, x, y, path = heapq.heappop(open_set)
            if self.maze[y][x] == 3: return {'path': [[p[0], p[1]] for p in path], 'visited': visited}
            for dy, dx in self.directions:
                nx, ny = x + dx, y + dy
                if 0 <= ny < 16 and 0 <= nx < 16 and self.maze[ny][nx] != 1 and (nx, ny) not in visited:
                    visited.add((nx, ny)); ng = g + 1; nh = heuristic(nx, ny)
                    heapq.heappush(open_set, (ng + nh, ng, nx, ny, path + [(nx, ny)]))
        return {'path': [[sx, sy]], 'visited': visited}

    def solve_maze(self):
        x = max(1, min(14, self.x_start.get()))
        y = max(1, min(14, self.y_start.get()))
        
        if self.maze[y][x] == 1:
            x, y = 1, 1
        
        algo = self.get_algo_key()
        default_heading = 1 
        
        if algo == 'wallFollowLeft':
            self.path_data = self.wall_follow_left(x, y, default_heading)
        elif algo == 'wallFollowRight':
            self.path_data = self.wall_follow_right(x, y, default_heading)
        elif algo == 'floodFill':
            self.path_data = self.floodFill(x, y) # Appel de la nouvelle fonction Flood Fill
        elif algo == 'dfs':
            self.path_data = self.dfs_search(x, y)
        elif algo == 'bfs':
            self.path_data = self.bfs_search(x, y)
        elif algo == 'tremaux':
            self.path_data = self.tremaux_algorithm(x, y, default_heading)
        elif algo == 'aiLearning':
            self.path_data = self.ai_based_learning(x, y)
        
        self.frame = 0
        self.sensor_data = {'capteur1': [], 'capteur2': [], 'capteur3': []}
        self.update_stats()
        self.update_display()


    # --- Méthode des Limites et Unités ---
    def get_sensor_limits_and_units(self, config_key):
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

    # --- Méthode : Distance au Mur ---
    def get_distance_to_nearest_wall(self, rx, ry):
        distances = []
        for dy, dx in self.directions:
            d = 0
            curr_x, curr_y = rx, ry
            while True:
                curr_x += dx
                curr_y += dy
                d += 1
                if not (0 <= curr_y < 16 and 0 <= curr_x < 16) or self.maze[curr_y][curr_x] == 1:
                    distances.append(d)
                    break
        min_distance_cells = min(distances) if distances else 0
        return min_distance_cells

    # --- Méthode de Simulation de données (Réaliste Télémétrie) ---
    def generate_sensor_data(self, config_key):
        limits = self.get_sensor_limits_and_units(config_key)
        data = []
        
        if config_key == 'Télémétrie' and self.path_data and self.frame < len(self.path_data['path']):
            rx, ry = self.path_data['path'][self.frame]
            min_dist_cells = self.get_distance_to_nearest_wall(rx, ry)
            base_val_1 = np.clip(min_dist_cells * 10, limits[0][0] + 5, limits[0][1])
            noise = random.uniform(-0.1, 0.1) * (limits[0][1] / 2)
            val_1 = np.clip(base_val_1 + noise, limits[0][0], limits[0][1])
            data.append(val_1)
        
        else:
            min_val, max_val, _ = limits[0]
            base_val = random.uniform(min_val, max_val)
            noise = random.uniform(-0.05, 0.05) * (max_val - min_val) 
            data.append(np.clip(base_val + noise, min_val, max_val))
        
        for min_val, max_val, _ in limits[1:]:
            base_val = random.uniform(min_val, max_val)
            noise = random.uniform(-0.05, 0.05) * (max_val - min_val) 
            val = base_val + noise
            data.append(np.clip(val, min_val, max_val))
        
        return data[0], data[1], data[2]

    # --- Méthode d'Interface ---
    def setup_ui(self):
        
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        main_frame.grid_rowconfigure(0, weight=1) 
        main_frame.grid_columnconfigure(0, weight=0) 
        main_frame.grid_columnconfigure(1, weight=1) 

        # Conteneur des contrôles et graphiques d'analyse (Colonne 0)
        control_analysis_container = ttk.Frame(main_frame)
        control_analysis_container.grid(row=0, column=0, sticky=(tk.W, tk.N, tk.S), padx=5)

        # Panneau de contrôle (haut gauche)
        control_frame = ttk.LabelFrame(control_analysis_container, text="Contrôles du Robot/Labyrinthe", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        row = 0
        
        # X Start / Y Start
        ttk.Label(control_frame, text="X start:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(control_frame, from_=1, to=14, textvariable=self.x_start, width=10,
                    command=self.solve_maze).grid(row=row, column=1, pady=5, padx=(0,10))
        ttk.Label(control_frame, text="Y start:").grid(row=row, column=2, sticky=tk.W, pady=5)
        ttk.Spinbox(control_frame, from_=1, to=14, textvariable=self.y_start, width=10,
                    command=self.solve_maze).grid(row=row, column=3, pady=5)
        row += 1
        
        # Algorithme (MIS À JOUR : 'Flood Fill')
        ttk.Label(control_frame, text="Algorithm:", font=('Arial', 10, 'bold')).grid(row=row, column=0, sticky=tk.W, pady=(10,2))
        algo_combo = ttk.Combobox(control_frame, textvariable=self.algo_var, width=28, state='readonly')
        algo_combo['values'] = ('Wall following left-hand rule', 'Wall following right-hand rule', 'Flood Fill', 'Deep First Search', 'Broad First Search', 'AI-based learning', 'Tremaux')
        algo_combo.grid(row=row, column=1, columnspan=3, pady=5, sticky=(tk.W, tk.E))
        algo_combo.bind('<<ComboboxSelected>>', lambda e: self.solve_maze())
        row += 1
        
        # Maze Selection
        ttk.Label(control_frame, text="Maze:", font=('Arial', 10, 'bold')).grid(row=row, column=0, sticky=tk.W, pady=(10,2))
        maze_combo = ttk.Combobox(control_frame, textvariable=self.maze_var, width=28, state='readonly')
        maze_combo['values'] = tuple(self.mazes.keys())
        maze_combo.grid(row=row, column=1, columnspan=3, pady=5, sticky=(tk.W, tk.E))
        maze_combo.bind('<<ComboboxSelected>>', lambda e: self.load_maze())
        row += 1

        # Configuration Scénarisée 
        ttk.Label(control_frame, text="Scénario Capteurs:", font=('Arial', 10, 'bold')).grid(row=row, column=0, sticky=tk.W, pady=(10,2))
        config_combo = ttk.Combobox(control_frame, textvariable=self.config_var, width=28, state='readonly')
        config_combo['values'] = tuple(self.sensor_titles.keys())
        config_combo.grid(row=row, column=1, columnspan=3, pady=5, sticky=(tk.W, tk.E))
        config_combo.bind('<<ComboboxSelected>>', lambda e: self.reset()) 
        row += 1
        
        # Boutons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.grid(row=row, column=0, columnspan=4, pady=10)
        self.play_btn = ttk.Button(btn_frame, text="▶ Play", command=self.toggle_play)
        self.play_btn.grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="↻ Reset", command=self.reset).grid(row=0, column=1, padx=5)
        row += 1
        
        # Statistiques
        stats_frame = ttk.LabelFrame(control_frame, text="Statistiques", padding="10")
        stats_frame.grid(row=row, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=10)
        self.stats_label = ttk.Label(stats_frame, text="", justify=tk.LEFT)
        self.stats_label.pack()

        # Canvas matplotlib (Labyrinthe - Colonne 1)
        canvas_frame = ttk.LabelFrame(main_frame, text="Labyrinthe", padding="10")
        canvas_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Graphiques d'Analyse (Bas de la colonne 0)
        sensor_frame = ttk.LabelFrame(control_analysis_container, text="Analyse Odométrique Holistique du Mouvement Non-Holonomique", padding="10")
        sensor_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        self.sensor_figs = []
        self.sensor_axes = []
        self.sensor_canvases = []

        sensor_plot_frame = ttk.Frame(sensor_frame)
        sensor_plot_frame.pack(fill=tk.BOTH, expand=True)

        for i in range(3):
            fig, ax = plt.subplots(figsize=(3, 1.5)) 
            canvas = FigureCanvasTkAgg(fig, master=sensor_plot_frame)
            canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
            
            self.sensor_figs.append(fig)
            self.sensor_axes.append(ax)
            self.sensor_canvases.append(canvas)

    # --- Méthodes d'Affichage et de Contrôle ---
    def update_display(self):
        if self.path_data is None: return
        self.ax.clear()
        colors = {0: '#f3f4f6', 1: '#1f2937', 3: '#ef4444'}
        for y in range(16):
            for x in range(16):
                color = colors.get(self.maze[y][x], '#f3f4f6')
                self.ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor=color, edgecolor='#9ca3af', linewidth=0.5))
        for (x, y) in self.path_data['visited']:
            self.ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor='lightblue', alpha=0.3))
        if self.frame > 0:
            path_segment = self.path_data['path'][:self.frame]
            if len(path_segment) > 1:
                path_x = [p[0] + 0.5 for p in path_segment]
                path_y = [p[1] + 0.5 for p in path_segment]
                self.ax.plot(path_x, path_y, 'b-', linewidth=2)
        if self.frame < len(self.path_data['path']):
            rx, ry = self.path_data['path'][self.frame]
            self.ax.plot(rx + 0.5, ry + 0.5, 'o', color='#fbbf24', markersize=15, 
                         markeredgecolor='black', markeredgewidth=2)
        self.ax.set_xlim(0, 16)
        self.ax.set_ylim(16, 0)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.fig.tight_layout() 
        self.canvas.draw()
        
        config_key = self.config_var.get()
        titles = self.sensor_titles.get(config_key, ("Capteur 1", "Capteur 2", "Capteur 3"))
        sensor_keys = ['capteur1', 'capteur2', 'capteur3']
        colors = ['b-', 'g-', 'r-']
        limits_and_units = self.get_sensor_limits_and_units(config_key)

        for i in range(3):
            ax = self.sensor_axes[i]
            canvas = self.sensor_canvases[i]
            data_key = sensor_keys[i]
            min_val, max_val, unit = limits_and_units[i]
            
            ax.clear()
            ax.set_title(titles[i], fontsize=8)
            
            if self.sensor_data[data_key]:
                times = [d[0] for d in self.sensor_data[data_key]]
                values = [d[1] for d in self.sensor_data[data_key]]
                
                ax.plot(times, values, colors[i], linewidth=1)
                
            ax.set_ylim(min_val, max_val)
            ax.set_ylabel(unit, fontsize=7)
            
            ax.set_xlabel('Temps (s)', fontsize=7)
            ax.tick_params(axis='both', which='major', labelsize=6)
            ax.grid(True, alpha=0.3)
                
            self.sensor_figs[i].tight_layout() 
            canvas.draw()

    def animate(self):
        if not self.is_running: return
        
        if self.frame < len(self.path_data['path']):
            self.frame += 1
            
            config_key = self.config_var.get()
            d1, d2, d3 = self.generate_sensor_data(config_key)
            time_step = self.frame * 0.1
            
            self.sensor_data['capteur1'].append((time_step, d1))
            self.sensor_data['capteur2'].append((time_step, d2))
            self.sensor_data['capteur3'].append((time_step, d3))
            
            for key in self.sensor_data:
                if len(self.sensor_data[key]) > 50:
                    self.sensor_data[key].pop(0)
            
            self.update_display()
            self.root.after(self.default_delay_ms, self.animate)
        else:
            self.is_running = False
            self.play_btn.config(text="▶ Play")
            
    def update_stats(self):
        if self.path_data:
            path_len = len(self.path_data['path'])
            visited_len = len(self.path_data['visited'])
            
            stats = f"Algorithme: {self.algo_var.get()}\n"
            stats += f"Taille: 16x16\n"
            stats += f"Exploré: {visited_len} / 256 ({visited_len/256*100:.1f}%)\n"
            stats += f"Longueur: {path_len}\n"
            stats += f"Efficacité (Path/Visited): {(path_len/visited_len*100) if visited_len else 0:.1f}%"
            self.stats_label.config(text=stats)

    def toggle_play(self):
        self.is_running = not self.is_running
        if self.is_running:
            self.play_btn.config(text="⏸ Pause")
            self.animate()
        else:
            self.play_btn.config(text="▶ Play")
    
    def reset(self):
        self.is_running = False
        self.play_btn.config(text="▶ Play")
        self.frame = 0
        self.sensor_data = {'capteur1': [], 'capteur2': [], 'capteur3': []}
        self.solve_maze()

# Créer l'application
if __name__ == "__main__":
    root = tk.Tk()
    root.minsize(1200, 750) 
    app = MazeSimulator(root)
    root.mainloop()
