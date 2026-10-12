# =============================================================
# SESI 4 — UCS, GREEDY BEST-FIRST & A*
# Mata Kuliah: Intelegensi Buatan (IF-030)
# =============================================================

import heapq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque

# ─────────────────────────────────────────────
# BAGIAN 1: UNIFORM COST SEARCH (UCS)
# ─────────────────────────────────────────────
print("=" * 55)
print(" BAGIAN 1: UNIFORM COST SEARCH (UCS)")
print("=" * 55)


def ucs(graf, start, goal):
    """UCS: proses node dengan g(n) terkecil. Priority Queue."""
    pq         = [(0, [start])]
    dikunjungi = {}
    while pq:
        biaya, jalur = heapq.heappop(pq)
        node = jalur[-1]
        if node in dikunjungi:
            continue
        dikunjungi[node] = biaya
        if node == goal:
            return jalur, biaya, len(dikunjungi)
        for nbr, bobot in graf.get(node, []):
            if nbr not in dikunjungi:
                heapq.heappush(pq, (biaya + bobot, jalur + [nbr]))
    return None, float('inf'), 0


graf_kota = {
    'A': [('B', 4), ('C', 2)],
    'B': [('A', 4), ('D', 3), ('E', 5)],
    'C': [('A', 2), ('E', 4), ('F', 1)],
    'D': [('B', 3), ('GOAL', 6)],
    'E': [('B', 5), ('C', 4), ('GOAL', 2)],
    'F': [('C', 1), ('GOAL', 3)],
    'GOAL': []
}

jalur, biaya, n_vis = ucs(graf_kota, 'A', 'GOAL')
print(f"\n  Jalur UCS : {' → '.join(jalur)}")
print(f"  Biaya     : {biaya} km")
print(f"  Dikunjungi: {n_vis} node")
print("  Optimal   : ✅ Ya")


# ─────────────────────────────────────────────
# BAGIAN 2: HEURISTIK & GREEDY BEST-FIRST
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print(" BAGIAN 2: HEURISTIK & GREEDY BEST-FIRST SEARCH")
print("=" * 55)


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5


def chebyshev(a, b):
    return max(abs(a[0]-b[0]), abs(a[1]-b[1]))


# Demonstrasi heuristik
p1, p2 = (0, 0), (3, 4)
print(f"\n  Titik A={p1}, B={p2}")
print(f"  Manhattan  h = {manhattan(p1,p2)}")
print(f"  Euclidean  h = {euclidean(p1,p2):.3f}")
print(f"  Chebyshev  h = {chebyshev(p1,p2)}")

GRID = [
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
]
START, GOAL = (0, 0), (4, 4)
ARAH_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def greedy_bfs(grid, start, goal):
    """Greedy: hanya pakai h(n)."""
    pq  = [(manhattan(start, goal), start, [start])]
    vis = {start}
    while pq:
        _, pos, jalur = heapq.heappop(pq)
        if pos == goal:
            return jalur, len(jalur)-1, len(vis)
        r, c = pos
        for dr, dc in ARAH_4:
            nr, nc = r+dr, c+dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) \
               and grid[nr][nc] == 0 and (nr, nc) not in vis:
                vis.add((nr, nc))
                heapq.heappush(pq, (manhattan((nr, nc), goal),
                                    (nr, nc), jalur + [(nr, nc)]))
    return None, float('inf'), len(vis)


j_grd, c_grd, v_grd = greedy_bfs(GRID, START, GOAL)
print(f"\n  Greedy: {' → '.join(str(p) for p in j_grd)}")
print(f"  Biaya: {c_grd} | Dikunjungi: {v_grd}")


# ─────────────────────────────────────────────
# BAGIAN 3: A* ALGORITHM
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print(" BAGIAN 3: A* — A-STAR ALGORITHM")
print("=" * 55)


def astar(grid, start, goal, heuristic=manhattan):
    """A*: f(n) = g(n) + h(n)."""
    R, C = len(grid), len(grid[0])
    pq   = [(heuristic(start, goal), 0, start, [start])]
    vis  = {}
    while pq:
        f, g, pos, jalur = heapq.heappop(pq)
        if pos in vis:
            continue
        vis[pos] = g
        if pos == goal:
            return jalur, g, len(vis)
        r, c = pos
        for dr, dc in ARAH_4:
            nr, nc = r+dr, c+dc
            if 0 <= nr < R and 0 <= nc < C and \
               grid[nr][nc] == 0 and (nr, nc) not in vis:
                g2 = g + 1
                h2 = heuristic((nr, nc), goal)
                heapq.heappush(pq, (g2+h2, g2, (nr, nc), jalur + [(nr, nc)]))
    return None, float('inf'), len(vis)


j_ast, c_ast, v_ast = astar(GRID, START, GOAL)
print(f"\n  A* jalur  : {' → '.join(str(p) for p in j_ast)}")
print(f"  Biaya     : {c_ast}")
print(f"  Dikunjungi: {v_ast} node")


# ─────────────────────────────────────────────
# BAGIAN 4: BFS UNTUK PERBANDINGAN
# ─────────────────────────────────────────────
def bfs_grid(grid, start, goal):
    q, vis = deque([[start]]), {start}
    while q:
        j = q.popleft()
        r, c = j[-1]
        if (r, c) == goal:
            return j, len(j)-1, len(vis)
        for dr, dc in ARAH_4:
            nr, nc = r+dr, c+dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) \
               and grid[nr][nc] == 0 and (nr, nc) not in vis:
                vis.add((nr, nc))
                q.append(j + [(nr, nc)])
    return None, float('inf'), len(vis)


j_bfs, c_bfs, v_bfs = bfs_grid(GRID, START, GOAL)

print("\n" + "=" * 55)
print(" BAGIAN 4: PERBANDINGAN LENGKAP")
print("=" * 55)

print(f"\n  {'─'*52}")
print(f"  {'Algo':<14} {'Panjang':<10} {'Node vis':<12} {'Optimal?'}")
print(f"  {'─'*52}")
print(f"  {'BFS':<14} {c_bfs:<10} {v_bfs:<12} ✅ Ya")
print(f"  {'Greedy':<14} {c_grd:<10} {v_grd:<12} ❌ Tidak")
print(f"  {'A*':<14} {c_ast:<10} {v_ast:<12} ✅ Ya")
print(f"  {'─'*52}")
print("  → A* optimal DAN lebih efisien dari BFS!")


# ─────────────────────────────────────────────
# BAGIAN 5: A* DENGAN DIAGONAL (8 ARAH)
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print(" BAGIAN 5: A* DIAGONAL (8 ARAH)")
print("=" * 55)

ARAH_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


def astar_8dir(grid, start, goal):
    """A* dengan 8 arah gerak — heuristik Chebyshev."""
    R, C = len(grid), len(grid[0])
    pq   = [(chebyshev(start, goal), 0, start, [start])]
    vis  = {}
    while pq:
        f, g, pos, jalur = heapq.heappop(pq)
        if pos in vis:
            continue
        vis[pos] = g
        if pos == goal:
            return jalur, g, len(vis)
        r, c = pos
        for dr, dc in ARAH_8:
            nr, nc = r+dr, c+dc
            cost_step = 1.414 if (dr != 0 and dc != 0) else 1.0
            if 0 <= nr < R and 0 <= nc < C and \
               grid[nr][nc] == 0 and (nr, nc) not in vis:
                g2 = g + cost_step
                h2 = chebyshev((nr, nc), goal)
                heapq.heappush(pq, (g2+h2, g2, (nr, nc),
                                    jalur + [(nr, nc)]))
    return None, float('inf'), len(vis)


j_8, c_8, v_8 = astar_8dir(GRID, START, GOAL)
print(f"\n  A* 8-arah jalur  : {len(j_8)-1} langkah")
print(f"  A* 4-arah jalur  : {c_ast} langkah")
print(f"  Keuntungan diagonal: {c_ast - (len(j_8)-1)} langkah lebih pendek")


# ─────────────────────────────────────────────
# BAGIAN 6: VISUALISASI
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print(" BAGIAN 6: VISUALISASI PERBANDINGAN")
print("=" * 55)


def visualisasi_pencarian(grid, jalur_map, start, goal):
    n_algo = len(jalur_map)
    fig, axes = plt.subplots(1, n_algo + 1, figsize=(4 * (n_algo + 1), 4))
    fig.patch.set_facecolor("#141414")

    COLORS = {
        'BFS': '#5B9DFF',
        'Greedy': '#FF6FA5',
        'A*': '#5BD68A',
        'A* 8-dir': '#FFD93D',
    }

    R, C = len(grid), len(grid[0])

    def draw_grid(ax, jalur, title, warna):
        img = np.zeros((R, C, 3))
        for r in range(R):
            for c in range(C):
                img[r, c] = [0.15, 0.15, 0.15] if grid[r][c] == 1 \
                             else [0.88, 0.93, 0.88]
        if jalur:
            for pos in jalur[1:-1]:
                r2, c2 = pos
                rgb = [int(warna[1:3], 16)/255,
                       int(warna[3:5], 16)/255,
                       int(warna[5:7], 16)/255]
                img[r2, c2] = rgb
        ax.imshow(img, interpolation='nearest')
        ax.text(start[1], start[0], 'S', ha='center', va='center',
                color='white', fontweight='bold', fontsize=10)
        ax.text(goal[1], goal[0], 'E', ha='center', va='center',
                color='white', fontweight='bold', fontsize=10)
        for x in range(C + 1):
            ax.axvline(x - 0.5, color='#141414', linewidth=1)
        for y in range(R + 1):
            ax.axhline(y - 0.5, color='#141414', linewidth=1)
        panjang = len(jalur) - 1 if jalur else 'N/A'
        ax.set_title(f"{title}\n({panjang} langkah)",
                     color='white', fontsize=9, fontweight='bold')
        ax.set_facecolor('#141414')
        ax.set_xticks([]); ax.set_yticks([])

    # Maze kosong
    draw_grid(axes[0], None, "Maze", '#888888')

    for ax, (nama, jalur) in zip(axes[1:], jalur_map.items()):
        draw_grid(ax, jalur, nama, COLORS.get(nama, '#aaa'))

    # Bar chart node visited
    ax_bar = fig.add_axes([0.02, -0.25, 0.96, 0.18])
    ax_bar.set_facecolor('#1c1c1c')
    algo_names = list(jalur_map.keys())
    visited = [v_bfs, v_grd, v_ast]
    bar_colors = [COLORS[n] for n in algo_names[:3]]
    bars = ax_bar.bar(algo_names[:3], visited[:3],
                      color=bar_colors, edgecolor='#141414', linewidth=2)
    for bar, val in zip(bars, visited[:3]):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(val), ha='center', color='white',
                    fontweight='bold', fontsize=9)
    ax_bar.set_title("Node Dikunjungi", color='white', fontsize=9)
    ax_bar.tick_params(colors='white')
    for spine in ax_bar.spines.values():
        spine.set_edgecolor('#444')
    ax_bar.set_facecolor('#1c1c1c')

    plt.suptitle("Sesi 4 — Perbandingan Algoritma Pencarian Heuristik",
                 color='white', fontsize=12, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig("week04_perbandingan.png", dpi=150,
                bbox_inches='tight', facecolor='#141414')
    plt.show()
    print("  📊 Grafik disimpan ke week04_perbandingan.png")


jalur_map = {
    'BFS':    j_bfs,
    'Greedy': j_grd,
    'A*':     j_ast,
}
visualisasi_pencarian(GRID, jalur_map, START, GOAL)

print("\n" + "=" * 55)
print("  ✅ SELESAI — Semua praktik Sesi 4 berhasil!")
print("=" * 55)
