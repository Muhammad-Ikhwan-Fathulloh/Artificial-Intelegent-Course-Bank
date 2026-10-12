# =============================================================
# SESI 2 — AGEN CERDAS & MODEL ML PERTAMA
# Mata Kuliah: Intelegensi Buatan (IF-030)
# =============================================================
# Install dependensi:
#   pip install numpy pandas scikit-learn matplotlib seaborn
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

# ─────────────────────────────────────────────
# BAGIAN 1: AGEN CERDAS DENGAN OOP
# ─────────────────────────────────────────────
print("=" * 55)
print(" BAGIAN 1: AGEN CERDAS DENGAN OOP PYTHON")
print("=" * 55)


class AgenDasar:
    """
    Agen sederhana berbasis Simple Reflex.
    Siklus: persepsi → keputusan → aksi
    """

    def __init__(self, nama: str):
        self.nama = nama
        self.riwayat: list = []        # memori internal agen

    def persepsi(self, lingkungan: str) -> str:
        """Membaca kondisi lingkungan dan menyimpan ke riwayat."""
        self.riwayat.append(lingkungan)
        return lingkungan

    def keputusan(self, kondisi: str) -> str:
        """Aturan if-then — Simple Reflex Agent."""
        if kondisi == "kotor":
            return "bersihkan"
        elif kondisi == "bersih":
            return "diam"
        else:
            return "eksplorasi"

    def aksi(self, keputusan: str) -> None:
        print(f"  [{self.nama}] Aksi: {keputusan.upper()}")

    def jalankan(self, kondisi_list: list) -> None:
        """Siklus utama: iterasi setiap kondisi."""
        print(f"\n{'─'*45}")
        print(f"  Agen '{self.nama}' mulai beroperasi")
        print(f"{'─'*45}")
        for kondisi in kondisi_list:
            p = self.persepsi(kondisi)
            k = self.keputusan(p)
            self.aksi(k)
        print(f"\n  Riwayat persepsi : {self.riwayat}")
        print(f"  Total siklus     : {len(self.riwayat)}")


# Jalankan Agen Dasar
ruangan = ["kotor", "bersih", "kotor", "tidak dikenal", "bersih"]
agen_vakum = AgenDasar("VacuumBot-01")
agen_vakum.jalankan(ruangan)


# ─────────────────────────────────────────────
# BAGIAN 2: INHERITANCE & POLYMORPHISM
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print(" BAGIAN 2: INHERITANCE & POLYMORPHISM")
print("=" * 55)


class AgenVakum(AgenDasar):
    """Agen vakum — membersihkan jika kotor."""

    def aksi(self, kondisi: str) -> str:      # override
        hasil = "Membersihkan 🧹" if kondisi == "kotor" else "Standby ⏸"
        print(f"  [{self.nama}] {hasil}")
        return hasil


class AgenKeamanan(AgenDasar):
    """Agen keamanan — patroli area."""

    def __init__(self, nama: str, zona_aman: list):
        super().__init__(nama)
        self.zona_aman = zona_aman

    def aksi(self, kondisi: str) -> str:
        if kondisi not in self.zona_aman:
            hasil = f"🚨 ALERT! Area '{kondisi}' tidak aman!"
        else:
            hasil = f"✅ Area '{kondisi}' aman, patroli lanjut."
        print(f"  [{self.nama}] {hasil}")
        return hasil


class AgenNavigasi(AgenDasar):
    """Agen navigasi — bergerak dalam grid."""

    def __init__(self, nama: str):
        super().__init__(nama)
        self.posisi = (0, 0)
        self.arah_map = {
            "utara": (0, 1), "selatan": (0, -1),
            "timur": (1, 0),  "barat": (-1, 0)
        }

    def aksi(self, kondisi: str) -> str:
        if kondisi in self.arah_map:
            dx, dy = self.arah_map[kondisi]
            self.posisi = (self.posisi[0] + dx, self.posisi[1] + dy)
            hasil = f"📍 Pindah ke {self.posisi}"
        else:
            hasil = "🛑 Diam di tempat"
        print(f"  [{self.nama}] {hasil}")
        return hasil


# Demo polimorfisme
agen_list = [
    AgenVakum("VacBot"),
    AgenKeamanan("GuardBot", zona_aman=["lobby", "koridor", "A", "B"]),
    AgenNavigasi("NavBot"),
]
inputs_poly = ["kotor", "gudang", "utara"]

print("\n  === DEMO POLIMORFISME ===")
for agen, inp in zip(agen_list, inputs_poly):
    agen.aksi(inp)


# ─────────────────────────────────────────────
# BAGIAN 3: PANDAS DASAR
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print(" BAGIAN 3: PANDAS — MANIPULASI DATA TABULAR")
print("=" * 55)

data_mhs = {
    "nama":  ["Andi", "Budi", "Citra", "Deni", "Eni"],
    "usia":  [21, 22, 20, 23, 21],
    "kota":  ["Jakarta", "Bandung", "Surabaya", "Medan", "Jakarta"],
    "nilai": [85, 90, 78, 88, 92],
}
df = pd.DataFrame(data_mhs)

print("\n── DataFrame Awal ──")
print(df.to_string(index=False))

print("\n── Statistik kolom 'nilai' ──")
print(df["nilai"].describe().round(2))

print("\n── Filter: nilai >= 85 ──")
top = df[df["nilai"] >= 85][["nama", "nilai"]]
print(top.to_string(index=False))

# Feature engineering
df["lulus"] = df["nilai"].apply(lambda x: "Lulus" if x >= 80 else "Tidak")
df["norm"]  = ((df["nilai"] - df["nilai"].min()) /
               (df["nilai"].max() - df["nilai"].min())).round(2)

print("\n── Dengan Kolom Baru ──")
print(df[["nama", "nilai", "lulus", "norm"]].to_string(index=False))

print("\n── Rata-rata nilai per kota ──")
grp = df.groupby("kota")["nilai"].mean().sort_values(ascending=False)
for kota, rata in grp.items():
    bar = "█" * int(rata / 10)
    print(f"  {kota:<12} {rata:5.1f}  {bar}")


# ─────────────────────────────────────────────
# BAGIAN 4: MODEL ML PERTAMA (KNN)
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print(" BAGIAN 4: MODEL ML PERTAMA — KNN PADA IRIS")
print("=" * 55)

iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names  = iris.target_names

print("\n── Info Dataset ──")
print(f"  Ukuran  : {X.shape} → {X.shape[0]} sampel, {X.shape[1]} fitur")
print(f"  Kelas   : {list(target_names)}")
print(f"  Distribusi: {dict(zip(*np.unique(y, return_counts=True)))}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"\n  Data latih : {len(X_train)} sampel")
print(f"  Data uji   : {len(X_test)} sampel")

# Latih
model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(X_train, y_train)
print("\n  ✅ KNN(k=3) berhasil dilatih!")

# Evaluasi
y_pred = model_knn.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)

print(f"\n── Akurasi : {akurasi:.2%} ──")
print("\n── Laporan Klasifikasi ──")
print(classification_report(y_test, y_pred, target_names=target_names))

# Prediksi data baru
print("── Prediksi Data Baru ──")
sampel_baru = [
    [5.1, 3.5, 1.4, 0.2],   # setosa
    [6.7, 3.0, 5.2, 2.3],   # virginica
    [5.9, 2.9, 4.2, 1.5],   # versicolor
]
for s in sampel_baru:
    pred = model_knn.predict([s])[0]
    proba = model_knn.predict_proba([s])[0]
    print(f"  Fitur {s} → {target_names[pred]:>12}  "
          f"(confidence: {max(proba):.0%})")


# ─────────────────────────────────────────────
# BAGIAN 5: PERBANDINGAN NILAI k
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print(" BAGIAN 5: PERBANDINGAN NILAI k PADA KNN")
print("=" * 55)

k_values  = [1, 3, 5, 7, 9, 11, 15]
akurasi_k = []

for k in k_values:
    m = KNeighborsClassifier(n_neighbors=k)
    m.fit(X_train, y_train)
    acc = m.score(X_test, y_test)
    akurasi_k.append(acc)
    bar = "█" * int(acc * 30)
    print(f"  k={k:>2}  {acc:.2%}  {bar}")

best_k = k_values[akurasi_k.index(max(akurasi_k))]
print(f"\n  ⭐ k terbaik = {best_k} (akurasi {max(akurasi_k):.2%})")


# ─────────────────────────────────────────────
# BAGIAN 6: AGEN + MODEL ML
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print(" BAGIAN 6: AGEN DENGAN 'OTAK' MODEL ML")
print("=" * 55)


class AgenKlasifikasiBunga(AgenDasar):
    """
    Agen yang menggunakan model ML sebagai mesin keputusan,
    menggantikan aturan if-then manual.
    """

    def __init__(self, nama: str):
        super().__init__(nama)
        self.model: KNeighborsClassifier | None = None
        self.label_map: dict | None = None

    def latih(self, X_tr, y_tr, label_map: dict) -> None:
        """Fase pembelajaran: agen melatih model internalnya."""
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.model.fit(X_tr, y_tr)
        self.label_map = label_map
        print(f"  [{self.nama}] ✅ Model berhasil dilatih!")

    def keputusan(self, fitur) -> str:
        """ML menggantikan aturan if-then."""
        if self.model is None:
            return "ERROR: Model belum dilatih!"
        pred = self.model.predict([fitur])[0]
        return self.label_map[pred]

    def aksi(self, hasil: str) -> None:
        emoji = {"setosa": "🌸", "versicolor": "🌼", "virginica": "🌺"}
        e = emoji.get(hasil, "🌿")
        print(f"  [{self.nama}] {e} Identifikasi → {hasil.upper()}")


# Jalankan
bot = AgenKlasifikasiBunga("FloraBot-v2")
bot.latih(X_train, y_train, dict(enumerate(target_names)))

print(f"\n── Agen Mengklasifikasi {len(X_test)} Sampel ──")
benar = 0
for fitur, label_asli in zip(X_test, y_test):
    hasil = bot.keputusan(fitur)
    bot.aksi(hasil)
    if hasil == target_names[label_asli]:
        benar += 1

print(f"\n  Skor agen : {benar}/{len(X_test)} = {benar/len(X_test):.2%}")


# ─────────────────────────────────────────────
# BAGIAN 7: VISUALISASI (matplotlib)
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print(" BAGIAN 7: VISUALISASI HASIL")
print("=" * 55)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor("#141414")
COLORS = ["#FFD93D", "#FF6FA5", "#5B9DFF"]

# — Plot 1: Akurasi vs k —
ax = axes[0]
ax.set_facecolor("#1c1c1c")
ax.plot(k_values, [a * 100 for a in akurasi_k],
        color="#5BD68A", linewidth=3, marker="o",
        markersize=8, markerfacecolor="#FFD93D",
        markeredgecolor="#141414", markeredgewidth=2)
ax.axvline(best_k, color="#FF6FA5", linestyle="--", linewidth=1.5,
           label=f"best k={best_k}")
ax.set_title("Akurasi vs Nilai k (KNN)", color="white",
             fontweight="bold", pad=10)
ax.set_xlabel("k", color="#aaa")
ax.set_ylabel("Akurasi (%)", color="#aaa")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#444")
ax.legend(labelcolor="white", facecolor="#1c1c1c", edgecolor="#444")
ax.grid(True, color="#333", linewidth=0.5)

# — Plot 2: Scatter fitur petal —
ax = axes[1]
ax.set_facecolor("#1c1c1c")
for i, cls in enumerate(target_names):
    mask = y == i
    ax.scatter(X[mask, 2], X[mask, 3],
               c=COLORS[i], label=cls, s=50,
               edgecolors="#141414", linewidths=0.8, alpha=0.85)
ax.set_title("Iris — Petal Length vs Width", color="white",
             fontweight="bold", pad=10)
ax.set_xlabel("Petal Length (cm)", color="#aaa")
ax.set_ylabel("Petal Width (cm)", color="#aaa")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#444")
ax.legend(labelcolor="white", facecolor="#1c1c1c", edgecolor="#444")

# — Plot 3: Confusion Matrix —
ax = axes[2]
ax.set_facecolor("#1c1c1c")
cm = confusion_matrix(y_test, y_pred)
im = ax.imshow(cm, cmap="YlOrRd", vmin=0, vmax=cm.max())
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(target_names, rotation=20, color="white", fontsize=8)
ax.set_yticklabels(target_names, color="white", fontsize=8)
for r in range(3):
    for c in range(3):
        ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                color="white" if cm[r, c] < cm.max() * 0.6 else "black",
                fontweight="bold", fontsize=12)
ax.set_title("Confusion Matrix KNN", color="white",
             fontweight="bold", pad=10)
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#444")

plt.suptitle("Sesi 2 — Visualisasi KNN pada Iris",
             color="white", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("week02_hasil_knn.png", dpi=150,
            bbox_inches="tight", facecolor="#141414")
plt.show()
print("  📊 Grafik disimpan ke week02_hasil_knn.png")

print("\n" + "=" * 55)
print("  ✅ SELESAI — Semua praktik Sesi 2 berhasil!")
print("=" * 55)
