# =============================================================
# SESI 3 — BFS & DFS: PENCARIAN PADA RUANG KEADAAN
# Mata Kuliah: Intelegensi Buatan (IF-030)
# =============================================================

from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────────────────────────
# BAGIAN 1: BFS PADA GRAF
# ─────────────────────────────────────────────
print("=" * 55)
print(" BAGIAN 1: BFS — BREADTH-FIRST SEARCH")
print("=" * 55)


def bfs(graf, start, goal):
    """BFS: eksplorasi level per level menggunakan Queue FIFO."""
    if start == goal:
        return [start], [start]
    antrian    = deque([[start]])
    dikunjungi = {start}
    urutan     = []
    while antrian:
        jalur = antrian.popleft()
        node  = jalur[-1]
        urutan.append(node)
        for nbr in graf.get(node, []):
            if nbr not in dikunjungi:
                dikunjungi.add(nbr)
                jalur_baru = jalur + [nbr]
                if nbr == goal:
                    return jalur_baru, urutan
                antrian.append(jalur_baru)
    return None, urutan


graf = {
    'A': ['B', 'E'], 'B': ['A', 'C', 'F'],
    'C': ['B', 'D'], 'D': ['C', 'G'],
    'E': ['A', 'F'], 'F': ['B', 'E'],
    'G': ['D']
}

jalur, kunjungan = bfs(graf, 'A', 'G')
print(f"\n  Jalur BFS  : {' → '.join(jalur)}")
print(f"  Langkah    : {len(jalur)-1}")
print(f"  Dikunjungi : {' → '.join(kunjungan)}")
print(f"  Total node : {len(kunjungan)}")


# ─────────────────────────────────────────────
# BAGIAN 2: DFS ITERATIF & REKURSIF
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print(" BAGIAN 2: DFS — DEPTH-FIRST SEARCH")
print("=" * 55)


def dfs_iteratif(graf, start, goal):
    """DFS iteratif menggunakan Stack LIFO."""
    stack      = [[start]]
    dikunjungi = {start}
    urutan     = []
    while stack:
        jalur = stack.pop()
        node  = jalur[-1]
        urutan.append(node)
        if node == goal:
            return jalur, urutan
        for nbr in reversed(graf.get(node, [])):
            if nbr not in dikunjungi:
                dikunjungi.add(nbr)
                stack.append(jalur + [nbr])
    return None, urutan


def dfs_rekursif(graf, node, goal, vis=None, path=None):
    """DFS rekursif — lebih bersih, awas stack overflow pada graf dalam."""
    if vis  is None: vis  = set()
    if path is None: path = []
    vis.add(node)
    path = path + [node]
    if node == goal:
        return path
    for nbr in graf.get(node, []):
        if nbr not in vis:
            hasil = dfs_rekursif(graf, nbr, goal, vis, path)
            if hasil:
                return hasil
    return None


j_dfs, k_dfs = dfs_iteratif(graf, 'A', 'G')
j_rek         = dfs_rekursif(graf, 'A', 'G')

print(f"\n  DFS iter   : {' → '.join(j_dfs)} ({len(j_dfs)-1} langkah, {len(k_dfs)} node)")
print(f"  DFS rekur  : {' → '.join(j_rek)} ({len(j_rek)-1} langkah)")

# Tabel perbandingan
print(f"\n  {'─'*55}")
print(f"  {'Algo':<16} {'Jalur':<22} {'Steps':<7} {'Visited'}")
print(f"  {'─'*55}")
j_bfs2, k_bfs2 = bfs(graf, 'A', 'G')
print(f"  {'BFS':<16} {'→'.join(j_bfs2):<22} {len(j_bfs2)-1:<7} {len(k_bfs2)}")
print(f"  {'DFS iteratif':<16} {'→'.join(j_dfs):<22} {len(j_dfs)-1:<7} {len(k_dfs)}")
print(f"  {'DFS rekursif':<16} {'→'.join(j_rek):<22} {len(j_rek)-1:<7} -")
print(f"  {'─'*55}")


# ─────────────────────────────────────────────
# BAGIAN 3: PATHFINDING PADA MAZE 2D
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print(" BAGIAN 3: PATHFINDING PADA MAZE 2D")
print("=" * 55)

MAZE = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0],
]
START, END = (0, 0), (4, 4)
ARAH = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def bfs_maze(maze, start, end):
    R, C = len(maze), len(maze[0])
    q    = deque([[start]])
    vis  = {start}
    while q:
        path = q.popleft()
        r, c = path[-1]
        if (r, c) == end:
            return path
        for dr, dc in ARAH:
            nr, nc = r+dr, c+dc
            if 0 <= nr < R and 0 <= nc < C and \
               maze[nr][nc] == 0 and (nr, nc) not in vis:
                vis.add((nr, nc))
                q.append(path + [(nr, nc)])
    return None


def dfs_maze(maze, start, end):
    R, C = len(maze), len(maze[0])
    stk  = [[start]]
    vis  = {start}
    while stk:
        path = stk.pop()
        r, c = path[-1]
        if (r, c) == end:
            return path
        for dr, dc in reversed(ARAH):
            nr, nc = r+dr, c+dc
            if 0 <= nr < R and 0 <= nc < C and \
               maze[nr][nc] == 0 and (nr, nc) not in vis:
                vis.add((nr, nc))
                stk.append(path + [(nr, nc)])
    return None


def cetak_maze(maze, jalur=None, label="Maze"):
    simbol = {START: 'S', END: 'E'}
    if jalur:
        for pos in jalur[1:-1]:
            simbol[pos] = '·'
    print(f"\n  ── {label} ──")
    for r, baris in enumerate(maze):
        baris_str = "  "
        for c, sel in enumerate(baris):
            if (r, c) in simbol:
                baris_str += f"[{simbol[(r,c)]}]"
            elif sel == 1:
                baris_str += "[█]"
            else:
                baris_str += "[ ]"
        print(baris_str)


j_bfs_m = bfs_maze(MAZE, START, END)
j_dfs_m = dfs_maze(MAZE, START, END)

cetak_maze(MAZE, label="Maze Kosong")
cetak_maze(MAZE, j_bfs_m, "BFS Solution")
cetak_maze(MAZE, j_dfs_m, "DFS Solution")

print(f"\n  BFS panjang : {len(j_bfs_m)-1} langkah")
print(f"  DFS panjang : {len(j_dfs_m)-1} langkah")
print(f"  BFS jalur   : {j_bfs_m}")
print(f"  DFS jalur   : {j_dfs_m}")


# ─────────────────────────────────────────────
# BAGIAN 4: IDDFS (Bonus)
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print(" BAGIAN 4: IDDFS — ITERATIVE DEEPENING DFS")
print("=" * 55)


def dls(graf, node, goal, limit, vis=None, path=None):
    """Depth-Limited Search — DFS dengan batas kedalaman."""
    if vis  is None: vis  = set()
    if path is None: path = []
    vis.add(node)
    path = path + [node]
    if node == goal:
        return path
    if limit == 0:
        return None
    for nbr in graf.get(node, []):
        if nbr not in vis:
            hasil = dls(graf, nbr, goal, limit-1, vis.copy(), path)
            if hasil:
                return hasil
    return None


def iddfs(graf, start, goal, max_depth=20):
    """IDDFS: jalankan DLS dengan batas kedalaman 0, 1, 2, …"""
    for depth in range(max_depth + 1):
        hasil = dls(graf, start, goal, depth)
        if hasil:
            print(f"  Ditemukan di kedalaman {depth}: {' → '.join(hasil)}")
            return hasil
        print(f"  Kedalaman {depth}: tidak ditemukan")
    return None


print("\n  IDDFS: A → G")
iddfs(graf, 'A', 'G')


# ─────────────────────────────────────────────
# BAGIAN 5: VISUALISASI MAZE
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print(" BAGIAN 5: VISUALISASI MAZE")
print("=" * 55)


def visualisasi_maze(maze, jalur_bfs, jalur_dfs):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#141414")

    for ax, jalur, judul, warna in zip(
        axes,
        [None, jalur_bfs, jalur_dfs],
        ["Maze Asli", "BFS — Jalur Terpendek", "DFS — Eksplorasi Dalam"],
        ["#555", "#5B9DFF", "#FF6FA5"]
    ):
        R, C = len(maze), len(maze[0])
        vis_grid = np.zeros((R, C, 3))

        for r in range(R):
            for c in range(C):
                if maze[r][c] == 1:
                    vis_grid[r, c] = [0.2, 0.2, 0.2]   # dinding
                else:
                    vis_grid[r, c] = [0.9, 0.95, 0.9]  # jalan

        # Warnai jalur
        if jalur:
            for pos in jalur[1:-1]:
                r, c = pos
                rgb = [int(warna[1:3], 16)/255,
                       int(warna[3:5], 16)/255,
                       int(warna[5:7], 16)/255]
                vis_grid[r, c] = rgb

        ax.imshow(vis_grid, interpolation='nearest')
        # Start dan End
        ax.text(START[1], START[0], 'S', ha='center', va='center',
                color='white', fontweight='bold', fontsize=12)
        ax.text(END[1], END[0], 'E', ha='center', va='center',
                color='white', fontweight='bold', fontsize=12)

        # Grid lines
        for x in range(C + 1):
            ax.axvline(x - 0.5, color='#141414', linewidth=1.5)
        for y in range(R + 1):
            ax.axhline(y - 0.5, color='#141414', linewidth=1.5)

        ax.set_title(judul, color='white', fontweight='bold',
                     fontsize=11, pad=8)
        ax.set_facecolor("#141414")
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')
        ax.set_xticks(range(C))
        ax.set_yticks(range(R))
        ax.set_xticklabels(range(C), color='#666', fontsize=8)
        ax.set_yticklabels(range(R), color='#666', fontsize=8)

    legend_items = [
        mpatches.Patch(color='#5B9DFF', label=f'BFS ({len(j_bfs_m)-1} langkah)'),
        mpatches.Patch(color='#FF6FA5', label=f'DFS ({len(j_dfs_m)-1} langkah)'),
        mpatches.Patch(color='#333', label='Dinding'),
    ]
    fig.legend(handles=legend_items, loc='lower center', ncol=3,
               labelcolor='white', facecolor='#1c1c1c',
               edgecolor='#444', fontsize=9, bbox_to_anchor=(0.5, -0.05))
    plt.suptitle("Sesi 3 — Pathfinding: BFS vs DFS pada Maze 2D",
                 color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("week03_maze_hasil.png", dpi=150,
                bbox_inches='tight', facecolor='#141414')
    plt.show()
    print("  📊 Grafik disimpan ke week03_maze_hasil.png")


visualisasi_maze(MAZE, j_bfs_m, j_dfs_m)

print("\n" + "=" * 55)
print("  ✅ SELESAI — Semua praktik Sesi 3 berhasil!")
print("=" * 55)
