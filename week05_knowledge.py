# =============================================================
# SESI 5 — KNOWLEDGE REPRESENTATION & LOGIKA PROPOSISIONAL
# Mata Kuliah: Intelegensi Buatan (IF-030)
# Install: pip install networkx matplotlib
# =============================================================

import itertools
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
# BAGIAN 1: KNOWLEDGE BASE + FORWARD CHAINING
# ─────────────────────────────────────────────
print("=" * 55)
print(" BAGIAN 1: KNOWLEDGE BASE & FORWARD CHAINING")
print("=" * 55)


class KnowledgeBase:
    """KB sederhana berbasis fakta (set) dan aturan (list)."""

    def __init__(self):
        self.fakta  = set()
        self.aturan = []

    def tambah_fakta(self, subjek, relasi, objek):
        self.fakta.add((subjek, relasi, objek))

    def tambah_aturan(self, kondisi: list, hasil: tuple):
        self.aturan.append((kondisi, hasil))

    def tanya(self, subjek, relasi, objek=None):
        return [f for f in self.fakta
                if f[0] == subjek and f[1] == relasi
                and (objek is None or f[2] == objek)]

    def forward_chain(self, verbose=True):
        """Inferensi maju: terapkan aturan hingga stabil."""
        baru, iterasi = True, 0
        total_baru = 0
        while baru:
            baru = False
            iterasi += 1
            for kondisi, hasil in self.aturan:
                if all(k in self.fakta for k in kondisi):
                    if hasil not in self.fakta:
                        self.fakta.add(hasil)
                        total_baru += 1
                        if verbose:
                            print(f"  [iter{iterasi}] ✅ Fakta baru: {hasil}")
                        baru = True
        if verbose:
            print(f"\n  Selesai: {total_baru} fakta baru ditambahkan")
        return total_baru

    def backward_chain(self, tujuan: tuple, depth=0) -> bool:
        """Inferensi mundur: buktikan tujuan dari fakta."""
        if tujuan in self.fakta:
            print("  " + "  " * depth + f"✅ Fakta langsung: {tujuan}")
            return True
        for kondisi, hasil in self.aturan:
            if hasil == tujuan:
                print("  " + "  " * depth + f"→ Coba aturan: {kondisi} → {hasil}")
                if all(self.backward_chain(k, depth+1) for k in kondisi):
                    print("  " + "  " * depth + f"✅ Terbukti: {tujuan}")
                    return True
        print("  " + "  " * depth + f"❌ Gagal: {tujuan}")
        return False


# Bangun KB domain hewan
kb = KnowledgeBase()
fakta_awal = [
    ("Tweety",  "adalah",  "Burung"),
    ("Burung",  "adalah",  "Hewan"),
    ("Tweety",  "memiliki","Sayap"),
    ("Rex",     "adalah",  "Anjing"),
    ("Anjing",  "adalah",  "Mamalia"),
    ("Mamalia", "adalah",  "Hewan"),
]
for f in fakta_awal:
    kb.tambah_fakta(*f)

kb.tambah_aturan(
    [("Tweety","adalah","Burung"), ("Burung","adalah","Hewan")],
    ("Tweety","adalah","Hewan")
)
kb.tambah_aturan(
    [("Rex","adalah","Anjing"), ("Anjing","adalah","Mamalia")],
    ("Rex","adalah","Mamalia")
)
kb.tambah_aturan(
    [("Rex","adalah","Mamalia"), ("Mamalia","adalah","Hewan")],
    ("Rex","adalah","Hewan")
)

print(f"\n  Fakta awal: {len(fakta_awal)}")
print("\n── Forward Chaining ──")
kb.forward_chain()

print("\n── Query ──")
print(f"  Tweety adalah Hewan?   {bool(kb.tanya('Tweety','adalah','Hewan'))}")
print(f"  Rex adalah Mamalia?    {bool(kb.tanya('Rex','adalah','Mamalia'))}")
print(f"  Rex adalah Hewan?      {bool(kb.tanya('Rex','adalah','Hewan'))}")

print("\n── Backward Chaining ──")
print("  Buktikan: (Rex, adalah, Hewan)")
kb2 = KnowledgeBase()   # segar tanpa fakta turunan
for f in fakta_awal:
    kb2.tambah_fakta(*f)
kb2.tambah_aturan(
    [("Rex","adalah","Anjing"), ("Anjing","adalah","Mamalia")],
    ("Rex","adalah","Mamalia")
)
kb2.tambah_aturan(
    [("Rex","adalah","Mamalia"), ("Mamalia","adalah","Hewan")],
    ("Rex","adalah","Hewan")
)
kb2.backward_chain(("Rex","adalah","Hewan"))


# ─────────────────────────────────────────────
# BAGIAN 2: TABEL KEBENARAN
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print(" BAGIAN 2: TABEL KEBENARAN OTOMATIS")
print("=" * 55)


def tabel_kebenaran(variabel: list, fn, nama="Ekspresi"):
    """Cetak tabel kebenaran untuk fungsi logika apa pun."""
    hdr = " | ".join(f"{v:^5}" for v in variabel)
    hdr += f" ‖ {nama:^14}"
    print(f"\n── {nama} ──")
    print(hdr)
    print("─" * len(hdr))
    for vals in itertools.product([False, True], repeat=len(variabel)):
        mapping = dict(zip(variabel, vals))
        hasil = fn(mapping)
        row = " | ".join(f"{'T':^5}" if v else f"{'F':^5}" for v in vals)
        row += f" ‖ {'T':^14}" if hasil else f" ‖ {'F':^14}"
        print(row)


tabel_kebenaran(['P', 'Q'],
    lambda m: not m['P'] or m['Q'],
    "P → Q")

tabel_kebenaran(['P', 'Q'],
    lambda m: (not (m['P'] and m['Q'])) == (not m['P'] or not m['Q']),
    "De Morgan ¬(P∧Q)↔(¬P∨¬Q)")

tabel_kebenaran(['P', 'Q', 'R'],
    lambda m: not (m['P'] and m['Q']) or m['R'],
    "(P∧Q)→R")
print("  (Semua T = tautologi!)" if True else "")


# ─────────────────────────────────────────────
# BAGIAN 3: SISTEM PAKAR HEWAN
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print(" BAGIAN 3: SISTEM PAKAR HEWAN")
print("=" * 55)


class SistemPakarHewan:
    """Sistem pakar berbasis aturan IF-THEN untuk identifikasi hewan."""

    def __init__(self):
        self.aturan = [
            ({'berbulu', 'menyusui'},                    'Mamalia'),
            ({'bersayap', 'bertelur', 'terbang'},         'Burung'),
            ({'bersayap', 'bertelur', 'berenang'},        'Penguin'),
            ({'berbulu', 'menyusui', 'terbang'},          'Kelelawar'),
            ({'bersisik', 'berdarah_dingin'},              'Reptil'),
            ({'bersisik', 'berdarah_dingin', 'berenang'}, 'Ikan'),
        ]

    def diagnosa(self, ciri_ciri: list) -> list:
        ciri_set = {c.lower() for c in ciri_ciri}
        return [k for cond, k in self.aturan if cond.issubset(ciri_set)] \
               or ["Tidak dikenali"]

    def cetak_aturan(self):
        print("\n  ── Daftar Aturan ──")
        for i, (cond, hasil) in enumerate(self.aturan, 1):
            print(f"  R{i}: IF {cond} THEN '{hasil}'")


sp = SistemPakarHewan()
sp.cetak_aturan()

test_cases = [
    (['bersayap', 'bertelur', 'berenang'],        "Penguin"),
    (['berbulu', 'menyusui', 'terbang'],           "Kelelawar"),
    (['bersisik', 'berdarah_dingin'],              "Reptil"),
    (['bersisik', 'berdarah_dingin', 'berenang'], "Ikan"),
    (['berbulu', 'menyusui'],                      "Mamalia"),
    (['bersayap', 'bertelur', 'terbang'],          "Burung"),
]

print("\n  ── Uji Sistem Pakar ──")
benar = 0
for ciri, expected in test_cases:
    hasil = sp.diagnosa(ciri)
    ok = "✅" if expected in hasil else "❌"
    if expected in hasil:
        benar += 1
    print(f"  {ok} {expected:<12} ciri={ciri} → {hasil}")
print(f"\n  Akurasi: {benar}/{len(test_cases)} = {benar/len(test_cases):.0%}")


# ─────────────────────────────────────────────
# BAGIAN 4: KNOWLEDGE GRAPH
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print(" BAGIAN 4: KNOWLEDGE GRAPH EKOSISTEM AI")
print("=" * 55)

G = nx.DiGraph()

entitas_warna = {
    'AI': '#FFD93D',
    'Machine Learning': '#FFD93D',
    'Deep Learning': '#FFD93D',
    'NLP': '#5B9DFF',
    'Computer Vision': '#5B9DFF',
    'Python': '#5BD68A',
    'NumPy': '#5BD68A',
    'scikit-learn': '#5BD68A',
    'PyTorch': '#FF6FA5',
    'Transformer': '#FF6FA5',
    'ChatGPT': '#FF6FA5',
}
for e in entitas_warna:
    G.add_node(e)

relasi = [
    ('AI', 'Machine Learning', 'mencakup'),
    ('AI', 'NLP', 'mencakup'),
    ('AI', 'Computer Vision', 'mencakup'),
    ('Machine Learning', 'Deep Learning', 'mencakup'),
    ('Deep Learning', 'NLP', 'digunakan_untuk'),
    ('Deep Learning', 'Computer Vision', 'digunakan_untuk'),
    ('Deep Learning', 'Transformer', 'menggunakan'),
    ('Transformer', 'ChatGPT', 'dasar_dari'),
    ('Python', 'NumPy', 'ekosistem'),
    ('Python', 'scikit-learn', 'ekosistem'),
    ('Python', 'PyTorch', 'ekosistem'),
    ('scikit-learn', 'Machine Learning', 'implementasi'),
    ('PyTorch', 'Deep Learning', 'implementasi'),
]
for s, d, r in relasi:
    G.add_edge(s, d, label=r)

print(f"\n  Nodes : {G.number_of_nodes()}")
print(f"  Edges : {G.number_of_edges()}")
print(f"  Density: {nx.density(G):.3f}")

# Analisis
print("\n  ── Node dengan edge keluar terbanyak ──")
out_deg = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)
for n, d in out_deg[:5]:
    bar = "█" * d
    print(f"  {n:<20} {d}  {bar}")

# Gambar
fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor("#141414")
ax.set_facecolor("#141414")

pos = nx.spring_layout(G, seed=42, k=2.0)
node_colors = [entitas_warna[n] for n in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                       node_size=2000, edgecolors='#141414',
                       linewidths=2.5, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=7.5,
                        font_weight='bold', ax=ax)
nx.draw_networkx_edges(G, pos, edge_color='#aaa',
                       arrows=True, arrowsize=18, ax=ax,
                       connectionstyle='arc3,rad=0.12',
                       width=1.5)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels,
                              font_size=5.5, font_color='#FFD93D', ax=ax)

legend_items = [
    mpatches.Patch(color='#FFD93D', label='AI/ML Core'),
    mpatches.Patch(color='#5B9DFF', label='Aplikasi AI'),
    mpatches.Patch(color='#5BD68A', label='Python Ecosystem'),
    mpatches.Patch(color='#FF6FA5', label='Deep Learning / LLM'),
]
ax.legend(handles=legend_items, loc='lower left',
          labelcolor='white', facecolor='#1c1c1c',
          edgecolor='#555', fontsize=8)
ax.set_title("Knowledge Graph: Ekosistem AI & Python",
             color='white', fontsize=13, fontweight='bold', pad=12)
ax.axis('off')
plt.tight_layout()
plt.savefig("week05_knowledge_graph.png", dpi=150,
            bbox_inches='tight', facecolor='#141414')
plt.show()
print("  📊 Graf disimpan ke week05_knowledge_graph.png")

print("\n" + "=" * 55)
print("  ✅ SELESAI — Semua praktik Sesi 5 berhasil!")
print("=" * 55)
