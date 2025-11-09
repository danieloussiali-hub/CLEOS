import time
import random
import heapq
from collections import deque

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-GUI backend suitable for Streamlit
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- Maze & Algorithms (kept mostly intact, adapted for Streamlit) ---

def create_all_mazes():
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

DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W

def wall_follow_left(maze, sx, sy, sh):
    path = [[sx, sy]]; visited = set([(sx, sy)]); x, y, heading = sx, sy, sh
    for _ in range(500):
        left_heading = (heading - 1) % 4; dy, dx = DIRECTIONS[left_heading]; left_y, left_x = y + dy, x + dx
        if 0 <= left_y < 16 and 0 <= left_x < 16 and maze[left_y][left_x] != 1:
            heading = left_heading; x, y = left_x, left_y
        else:
            dy, dx = DIRECTIONS[heading]; next_y, next_x = y + dy, x + dx
            if 0 <= next_y < 16 and 0 <= next_x < 16 and maze[next_y][next_x] != 1: x, y = next_x, next_y
            else: heading = (heading + 1) % 4; continue
        path.append([x, y]); visited.add((x, y));
        if maze[y][x] == 3: break
    return {'path': path, 'visited': visited}

def wall_follow_right(maze, sx, sy, sh):
    path = [[sx, sy]]; visited = set([(sx, sy)]); x, y, heading = sx, sy, sh
    for _ in range(500):
        right_heading = (heading + 1) % 4; dy, dx = DIRECTIONS[right_heading]; right_y, right_x = y + dy, x + dx
        if 0 <= right_y < 16 and 0 <= right_x < 16 and maze[right_y][right_x] != 1:
            heading = right_heading; x, y = right_x, right_y
        else:
            dy, dx = DIRECTIONS[heading]; next_y, next_x = y + dy, x + dx
            if 0 <= next_y < 16 and 0 <= next_x < 16 and maze[next_y][next_x] != 1: x, y = next_x, next_y
            else: heading = (heading - 1) % 4; continue
        path.append([x, y]); visited.add((x, y));
        if maze[y][x] == 3: break
    return {'path': path, 'visited': visited}

def bfs_search(maze, sx, sy):
    queue = deque([((sx, sy), [(sx, sy)])]); visited = {(sx, sy)}
    while queue:
        (x, y), path = queue.popleft()
        if maze[y][x] == 3: return {'path': [[p[0], p[1]] for p in path], 'visited': visited}
        for dy, dx in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= ny < 16 and 0 <= nx < 16 and maze[ny][nx] != 1 and (nx, ny) not in visited:
                visited.add((nx, ny)); queue.append(((nx, ny), path + [(nx, ny)]))
    return {'path': [[sx, sy]], 'visited': visited}

def dfs_search(maze, sx, sy):
    stack = [((sx, sy), [(sx, sy)])]; visited = {(sx, sy)}
    while stack:
        (x, y), path = stack.pop()
        if maze[y][x] == 3: return {'path': [[p[0], p[1]] for p in path], 'visited': visited}
        for dy, dx in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= ny < 16 and 0 <= nx < 16 and maze[ny][nx] != 1 and (nx, ny) not in visited:
                visited.add((nx, ny)); stack.append(((nx, ny), path + [(nx, ny)]))
    return {'path': [[sx, sy]], 'visited': visited}

def floodFill(maze, sx, sy):
    rows, cols = maze.shape
    distance_grid = np.full((rows, cols), -1)
    queue = deque()
    end_points = np.argwhere(maze == 3)
    if not end_points.size:
        return {'path': [[sx, sy]], 'visited': set([(sx, sy)])}
    for y_end, x_end in end_points:
        distance_grid[y_end, x_end] = 0
        queue.append((x_end, y_end))
    visited_cells = set()
    while queue:
        x, y = queue.popleft()
        visited_cells.add((x, y))
        current_dist = distance_grid[y, x]
        for dy, dx in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= ny < rows and 0 <= nx < cols and maze[ny][nx] != 1 and distance_grid[ny, nx] == -1:
                distance_grid[ny, nx] = current_dist + 1
                queue.append((nx, ny))
                visited_cells.add((nx, ny))
    if distance_grid[sy, sx] == -1:
         return {'path': [[sx, sy]], 'visited': visited_cells}
    path = [[sx, sy]]
    cx, cy = sx, sy
    while distance_grid[cy, cx] > 0:
        current_dist = distance_grid[cy, cx]
        found_next = False
        for dy, dx in DIRECTIONS:
            nx, ny = cx + dx, cy + dy
            if 0 <= ny < rows and 0 <= nx < cols and distance_grid[ny, nx] == current_dist - 1:
                cx, cy = nx, ny
                path.append([cx, cy])
                found_next = True
                break
        if not found_next:
            break
    return {'path': path, 'visited': visited_cells}

def tremaux_algorithm(maze, sx, sy, sh):
    path = [[sx, sy]]; visited = {}; visited[(sx, sy)] = 1; x, y, heading = sx, sy, sh
    for _ in range(500):
        if maze[y][x] == 3: break
        neighbors = [];
        for idx, (dy, dx) in enumerate(DIRECTIONS):
            nx, ny = x + dx, y + dy
            if 0 <= ny < 16 and 0 <= nx < 16 and maze[ny][nx] != 1:
                count = visited.get((nx, ny), 0); neighbors.append((nx, ny, idx, count))
        if not neighbors: break
        neighbors.sort(key=lambda n: n[3]);
        x, y, heading = neighbors[0][0], neighbors[0][1], neighbors[0][2]
        visited[(x, y)] = visited.get((x, y), 0) + 1; path.append([x, y])
    return {'path': path, 'visited': set(visited.keys())}

def ai_based_learning(maze, sx, sy):
    def heuristic(x, y): return abs(14 - x) + abs(14 - y)
    open_set = [(heuristic(sx, sy), 0, sx, sy, [(sx, sy)])]; visited = {(sx, sy)}
    while open_set:
        _, g, x, y, path = heapq.heappop(open_set)
        if maze[y][x] == 3: return {'path': [[p[0], p[1]] for p in path], 'visited': visited}
        for dy, dx in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= ny < 16 and 0 <= nx < 16 and maze[ny][nx] != 1 and (nx, ny) not in visited:
                visited.add((nx, ny)); ng = g + 1; nh = heuristic(nx, ny)
                heapq.heappush(open_set, (ng + nh, ng, nx, ny, path + [(nx, ny)]))
    return {'path': [[sx, sy]], 'visited': visited}

# --- Sensor / Utilities ---

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
    for dy, dx in DIRECTIONS:
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
    limits = get_sensor_limits_and_units(config_key)
    data = []
    if config_key == 'Télémétrie' and path_data and frame < len(path_data['path']):
        rx, ry = path_data['path'][frame]
        min_dist_cells = get_distance_to_nearest_wall(maze, rx, ry)
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

# --- Streamlit App UI & State Management ---

st.set_page_config(page_title="Simulateur de Labyrinthe (Streamlit)", layout="wide")

# Initialize session state
if 'mazes' not in st.session_state:
    st.session_state.mazes = create_all_mazes()
if 'maze_name' not in st.session_state:
    st.session_state.maze_name = 'Complexe (Moyen)'
if 'algo' not in st.session_state:
    st.session_state.algo = 'Flood Fill'
if 'config' not in st.session_state:
    st.session_state.config = 'Télémétrie'
if 'x_start' not in st.session_state:
    st.session_state.x_start = 1
if 'y_start' not in st.session_state:
    st.session_state.y_start = 1
if 'path_data' not in st.session_state:
    st.session_state.path_data = None
if 'frame' not in st.session_state:
    st.session_state.frame = 0
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = {'capteur1': [], 'capteur2': [], 'capteur3': []}
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'delay_ms' not in st.session_state:
    st.session_state.delay_ms = 100

# Helper: map UI algo label to function key
ALGO_MAP = {
    'Wall following left-hand rule': 'wallFollowLeft',
    'Wall following right-hand rule': 'wallFollowRight',
    'Flood Fill': 'floodFill',
    'Deep First Search': 'dfs',
    'Broad First Search': 'bfs',
    'AI-based learning': 'aiLearning',
    'Tremaux': 'tremaux'
}

def solve_maze():
    maze = st.session_state.mazes[st.session_state.maze_name].copy()
    x = max(1, min(14, st.session_state.x_start))
    y = max(1, min(14, st.session_state.y_start))
    if maze[y][x] == 1:
        x, y = 1, 1
    algo_key = ALGO_MAP.get(st.session_state.algo, 'wallFollowLeft')
    default_heading = 1
    if algo_key == 'wallFollowLeft':
        st.session_state.path_data = wall_follow_left(maze, x, y, default_heading)
    elif algo_key == 'wallFollowRight':
        st.session_state.path_data = wall_follow_right(maze, x, y, default_heading)
    elif algo_key == 'floodFill':
        st.session_state.path_data = floodFill(maze, x, y)
    elif algo_key == 'dfs':
        st.session_state.path_data = dfs_search(maze, x, y)
    elif algo_key == 'bfs':
        st.session_state.path_data = bfs_search(maze, x, y)
    elif algo_key == 'tremaux':
        st.session_state.path_data = tremaux_algorithm(maze, x, y, default_heading)
    elif algo_key == 'aiLearning':
        st.session_state.path_data = ai_based_learning(maze, x, y)
    st.session_state.frame = 0
    st.session_state.sensor_data = {'capteur1': [], 'capteur2': [], 'capteur3': []}

def reset():
    st.session_state.frame = 0
    st.session_state.is_running = False
    st.session_state.sensor_data = {'capteur1': [], 'capteur2': [], 'capteur3': []}
    solve_maze()

# Sidebar controls
with st.sidebar:
    st.title("Contrôles")
    st.session_state.maze_name = st.selectbox("Maze", list(st.session_state.mazes.keys()), index=list(st.session_state.mazes.keys()).index(st.session_state.maze_name))
    st.session_state.algo = st.selectbox("Algorithm", list(ALGO_MAP.keys()), index=list(ALGO_MAP.keys()).index(st.session_state.algo))
    st.session_state.config = st.selectbox("Scénario Capteurs", ('Télémétrie', 'Caméras', 'Centrale inertielle', 'Système radar doppler et optique', 'Coordonnées GPS (triangulation)'), index=0)
    st.session_state.x_start = st.number_input("X start", min_value=1, max_value=14, value=int(st.session_state.x_start), step=1)
    st.session_state.y_start = st.number_input("Y start", min_value=1, max_value=14, value=int(st.session_state.y_start), step=1)
    st.session_state.delay_ms = st.slider("Delay (ms)", min_value=10, max_value=1000, value=st.session_state.delay_ms, step=10)
    if st.button("Solve / Apply"):
        solve_maze()
    if st.button("Reset"):
        reset()
    # Play / Step controls
    col1, col2 = st.columns([1,1])
    with col1:
        if st.session_state.is_running:
            if st.button("⏸ Pause"):
                st.session_state.is_running = False
        else:
            if st.button("▶ Play"):
                st.session_state.is_running = True
    with col2:
        if st.button("Step"):
            # single step
            if st.session_state.path_data:
                if st.session_state.frame < len(st.session_state.path_data['path']) - 1:
                    st.session_state.frame += 1
                    t = st.session_state.frame * 0.1
                    d1, d2, d3 = generate_sensor_data(st.session_state.mazes[st.session_state.maze_name], st.session_state.config, st.session_state.path_data, st.session_state.frame)
                    st.session_state.sensor_data['capteur1'].append((t, d1))
                    st.session_state.sensor_data['capteur2'].append((t, d2))
                    st.session_state.sensor_data['capteur3'].append((t, d3))
                    # keep history trimmed
                    for k in st.session_state.sensor_data:
                        if len(st.session_state.sensor_data[k]) > 200:
                            st.session_state.sensor_data[k] = st.session_state.sensor_data[k][-200:]

# Main layout
maze_col, graphs_col = st.columns((2, 1))

def plot_maze_and_path():
    maze = st.session_state.mazes[st.session_state.maze_name]
    fig, ax = plt.subplots(figsize=(6,6))
    colors = {0: '#f3f4f6', 1: '#1f2937', 3: '#ef4444'}
    for yy in range(16):
        for xx in range(16):
            color = colors.get(maze[yy][xx], '#f3f4f6')
            ax.add_patch(Rectangle((xx, yy), 1, 1, facecolor=color, edgecolor='#9ca3af', linewidth=0.5))
    if st.session_state.path_data:
        for (x, y) in st.session_state.path_data['visited']:
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor='lightblue', alpha=0.25))
        # draw path up to current frame
        frame = st.session_state.frame
        if frame > 0:
            path_segment = st.session_state.path_data['path'][:frame]
            if len(path_segment) > 1:
                path_x = [p[0] + 0.5 for p in path_segment]
                path_y = [p[1] + 0.5 for p in path_segment]
                ax.plot(path_x, path_y, 'b-', linewidth=2)
        # draw robot marker at current frame
        if st.session_state.path_data['path']:
            idx = min(st.session_state.frame, len(st.session_state.path_data['path']) - 1)
            rx, ry = st.session_state.path_data['path'][idx]
            ax.plot(rx + 0.5, ry + 0.5, 'o', color='#fbbf24', markersize=16, markeredgecolor='black', markeredgewidth=2)
    ax.set_xlim(0, 16)
    ax.set_ylim(16, 0)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()
    return fig

def plot_sensor_graphs():
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
    for i in range(3):
        fig, ax = plt.subplots(figsize=(4,1.8))
        key = f'capteur{i+1}'
        ax.set_title(titles.get(config_key, ("Capteur 1","Capteur 2","Capteur 3"))[i], fontsize=9)
        data = st.session_state.sensor_data.get(key, [])
        if data:
            times = [d[0] for d in data]
            values = [d[1] for d in data]
            ax.plot(times, values, ['b','g','r'][i]+'-', linewidth=1)
        min_val, max_val, unit = limits_and_units[i]
        ax.set_ylim(min_val, max_val)
        ax.set_ylabel(unit, fontsize=8)
        ax.set_xlabel('Temps (s)', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        figs.append(fig)
    return figs

with maze_col:
    st.header("Labyrinthe")
    if st.session_state.path_data is None:
        solve_maze()
    fig_maze = plot_maze_and_path()
    st.pyplot(fig_maze)

with graphs_col:
    st.header("Analyse Odométrique")
    figs = plot_sensor_graphs()
    for f in figs:
        st.pyplot(f)

# Statistics
if st.session_state.path_data:
    path_len = len(st.session_state.path_data['path'])
    visited_len = len(st.session_state.path_data['visited'])
    st.sidebar.markdown("### Statistiques")
    st.sidebar.text(f"Algorithme: {st.session_state.algo}")
    st.sidebar.text("Taille: 16x16")
    st.sidebar.text(f"Exploré: {visited_len} / 256 ({visited_len/256*100:.1f}%)")
    st.sidebar.text(f"Longueur: {path_len}")
    eff = (path_len/visited_len*100) if visited_len else 0
    st.sidebar.text(f"Efficacité (Path/Visited): {eff:.1f}%")

# Auto-play handling: advance one step per run when running, then re-run
if st.session_state.is_running:
    if st.session_state.path_data and st.session_state.frame < len(st.session_state.path_data['path']) - 1:
        st.session_state.frame += 1
        t = st.session_state.frame * 0.1
        d1, d2, d3 = generate_sensor_data(st.session_state.mazes[st.session_state.maze_name], st.session_state.config, st.session_state.path_data, st.session_state.frame)
        st.session_state.sensor_data['capteur1'].append((t, d1))
        st.session_state.sensor_data['capteur2'].append((t, d2))
        st.session_state.sensor_data['capteur3'].append((t, d3))
        for k in st.session_state.sensor_data:
            if len(st.session_state.sensor_data[k]) > 200:
                st.session_state.sensor_data[k] = st.session_state.sensor_data[k][-200:]
        # small delay, then rerun the script to get next frame
        time.sleep(st.session_state.delay_ms / 1000.0)
        st.experimental_rerun()
    else:
        st.session_state.is_running = False
        st.success("Simulation terminée.")