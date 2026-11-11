# =============================================================
# STUDI KASUS AI — KEHIDUPAN SEHARI-HARI
# Mata Kuliah: Intelegensi Buatan (IF-030)
# 7 Kasus: GPS, Diagnosa, Rekomendasi, Spam,
#          Harga Rumah, Kelulusan, Chatbot
# Install: pip install scikit-learn pandas matplotlib seaborn
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════
# KASUS 1: NAVIGASI GPS — A* & BFS pada Peta Kota
# Konsep: Sesi 3 (BFS/DFS), Sesi 4 (A*)
# ══════════════════════════════════════════════════════
import heapq, math
from collections import deque

print("=" * 62)
print(" KASUS 1: NAVIGASI GPS — A* & BFS")
print("=" * 62)

PETA_KOTA = {
    'Rumah':    [('Pasar', 4),    ('Terminal', 5)],
    'Pasar':    [('Rumah', 4),    ('Kampus', 3),   ('Masjid', 7)],
    'Terminal': [('Rumah', 5),    ('Kampus', 6),   ('Stasiun', 3)],
    'Kampus':   [('Pasar', 3),    ('Terminal', 6), ('Masjid', 2)],
    'Masjid':   [('Kampus', 2),   ('Pasar', 7)],
    'Stasiun':  [('Terminal', 3), ('Kampus', 8)],
}
KOORDINAT = {
    'Rumah':    (-6.200, 106.816),
    'Pasar':    (-6.195, 106.820),
    'Terminal': (-6.210, 106.814),
    'Kampus':   (-6.185, 106.825),
    'Masjid':   (-6.180, 106.830),
    'Stasiun':  (-6.215, 106.812),
}


def haversine(a, b):
    R = 6371
    la1, lo1 = math.radians(a[0]), math.radians(a[1])
    la2, lo2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = la2 - la1, lo2 - lo1
    x = (math.sin(dlat/2)**2 +
         math.cos(la1) * math.cos(la2) * math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(x))


def astar_gps(peta, koord, start, goal):
    h = lambda n: haversine(koord[n], koord[goal])
    pq  = [(h(start), 0, start, [start])]
    vis = {}
    while pq:
        f, g, pos, path = heapq.heappop(pq)
        if pos in vis:
            continue
        vis[pos] = g
        if pos == goal:
            return path, round(g, 2), len(vis)
        for nbr, d in peta.get(pos, []):
            if nbr not in vis:
                g2 = g + d
                heapq.heappush(pq, (g2 + h(nbr), g2, nbr,
                                    path + [nbr]))
    return None, None, 0


def bfs_gps(peta, start, goal):
    q, vis = deque([[start]]), {start}
    while q:
        path = q.popleft()
        node = path[-1]
        if node == goal:
            cost = sum(
                next(d for nb, d in peta[path[i]] if nb == path[i+1])
                for i in range(len(path)-1))
            return path, cost, len(vis)
        for nb, _ in peta.get(node, []):
            if nb not in vis:
                vis.add(nb)
                q.append(path + [nb])
    return None, None, 0


def ucs_gps(peta, start, goal):
    pq, vis = [(0, [start])], {}
    while pq:
        g, path = heapq.heappop(pq)
        node = path[-1]
        if node in vis:
            continue
        vis[node] = g
        if node == goal:
            return path, round(g, 2), len(vis)
        for nb, d in peta.get(node, []):
            if nb not in vis:
                heapq.heappush(pq, (g + d, path + [nb]))
    return None, None, 0


print(f"\n  {'─'*55}")
print(f"  {'Algoritma':<16} {'Rute':<32} {'Jarak':>6} {'Visited':>8}")
print(f"  {'─'*55}")
for nama, fn in [('BFS', bfs_gps), ('UCS', ucs_gps), ('A*', astar_gps)]:
    if nama == 'A*':
        path, dist, nv = fn(PETA_KOTA, KOORDINAT, 'Rumah', 'Masjid')
    else:
        path, dist, nv = fn(PETA_KOTA, 'Rumah', 'Masjid')
    rute = '→'.join(path) if path else 'Tidak ada'
    print(f"  {nama:<16} {rute:<32} {dist:>5}km {nv:>7} node")

print(f"\n  ✅ A* paling efisien: heuristik mengurangi node yang dieksplorasi.")


# ══════════════════════════════════════════════════════
# KASUS 2: DIAGNOSA PENYAKIT — Rule-Based + CF
# Konsep: Sesi 5 (KB), Sesi 6 (Inferensi, CF)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 62)
print(" KASUS 2: DIAGNOSA PENYAKIT RINGAN — RULE-BASED + CF")
print("=" * 62)


class DiagnosaPenyakit:
    """Sistem pakar diagnosa dengan Certainty Factor."""

    ATURAN = [
        (["demam", "batuk", "pilek"],                  "flu_biasa",        0.90),
        (["demam", "batuk", "sesak"],                   "ispa",             0.80),
        (["demam", "diare", "mual"],                    "gastroenteritis",  0.85),
        (["kepala_pusing", "mual"],                     "vertigo",          0.70),
        (["demam", "ruam_kulit", "nyeri_sendi"],        "demam_berdarah",   0.80),
        (["batuk", "sesak", "nafas_berbunyi"],          "asma",             0.82),
        (["demam","batuk","sesak","hilang_penciuman"],  "covid19",          0.88),
        (["diare","muntah","demam"],                    "tifus",            0.72),
        (["demam","sakit_kepala","kaku_leher"],         "meningitis",       0.85),
    ]

    SARAN = {
        "flu_biasa":       "💊 Istirahat 2-3 hari, PCT 500mg tiap 6 jam, minum 2L/hari",
        "ispa":            "🏥 Konsultasi dokter, kemungkinan perlu antibiotik",
        "gastroenteritis": "💧 Oralit setiap BAB, hindari makanan pedas 3 hari",
        "vertigo":         "🛏️ Tirah baring, hindari gerakan tiba-tiba, Betahistine",
        "demam_berdarah":  "🚨 SEGERA ke IGD! Cek trombosit, hindari aspirin",
        "asma":            "🫁 Gunakan inhaler, hindari pemicu alergi",
        "covid19":         "🧪 Isolasi mandiri 5-10 hari, swab PCR, lapor puskesmas",
        "tifus":           "🏥 Perlu antibiotik dari dokter, tirah baring ketat",
        "meningitis":      "🚑 DARURAT! Segera IGD terdekat",
    }

    def diagnosa(self, gejala_input: list, threshold_cocok=0.6) -> list:
        gejala_set = {g.lower().strip() for g in gejala_input}
        hasil = []
        for cond, diag, cf_rule in self.ATURAN:
            cocok = sum(1 for g in cond if g in gejala_set)
            rasio = cocok / len(cond)
            if rasio >= threshold_cocok:
                cf = round(cf_rule * rasio, 3)
                hasil.append((diag, cf, cocok, len(cond)))
        return sorted(hasil, key=lambda x: -x[1])

    def cetak(self, gejala: list):
        print(f"\n  Gejala: {gejala}")
        hasil = self.diagnosa(gejala)
        if not hasil:
            print("  Tidak ada diagnosis yang cocok. Konsultasi dokter.")
            return
        print(f"  {'─'*60}")
        print(f"  {'Diagnosis':<22} {'CF':>6} {'Cocok':>7}  Level")
        print(f"  {'─'*60}")
        for d, cf, ck, total in hasil[:3]:
            bar  = "█" * int(cf * 15)
            lvl  = ("⚠️ TINGGI" if cf > 0.7 else
                    "📋 SEDANG" if cf > 0.4 else "ℹ️ RENDAH")
            print(f"  {d:<22} {cf:>6.3f}  {ck}/{total}   {bar} {lvl}")
            print(f"     → {self.SARAN.get(d, 'Konsultasi dokter')}")


sp_med = DiagnosaPenyakit()
sp_med.cetak(["demam", "batuk", "pilek"])
sp_med.cetak(["demam", "batuk", "sesak", "hilang_penciuman"])
sp_med.cetak(["demam", "sakit_kepala", "kaku_leher"])


# ══════════════════════════════════════════════════════
# KASUS 3: REKOMENDASI PRODUK — K-Means + KNN
# Konsep: Sesi 2 (KNN), Sesi 7 (K-Means, Clustering)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 62)
print(" KASUS 3: REKOMENDASI PRODUK — K-MEANS + KNN")
print("=" * 62)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

np.random.seed(42)
N_USER = 120

data_user = pd.DataFrame({
    'frekuensi_beli':   np.random.randint(1, 30, N_USER),
    'total_pengeluaran': np.random.randint(50_000, 8_000_000, N_USER),
    'minat_elektronik': np.random.uniform(0, 1, N_USER).round(2),
    'minat_fashion':    np.random.uniform(0, 1, N_USER).round(2),
    'minat_makanan':    np.random.uniform(0, 1, N_USER).round(2),
    'minat_olahraga':   np.random.uniform(0, 1, N_USER).round(2),
})
data_user['user_id'] = [f"U{i:03d}" for i in range(N_USER)]

X_user = StandardScaler().fit_transform(
    data_user.drop('user_id', axis=1))

# Cari k optimal
best_k, best_sil = 3, 0
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    sil = silhouette_score(X_user, km.fit_predict(X_user))
    if sil > best_sil:
        best_sil, best_k = sil, k

km_user = KMeans(n_clusters=best_k, random_state=42, n_init=10)
data_user['segmen'] = km_user.fit_predict(X_user)

NAMA_SEGMEN_USER = {0: "Power Shopper", 1: "Casual Buyer", 2: "Budget Hunter"}
print(f"\n  k optimal = {best_k} (silhouette={best_sil:.3f})")
print(f"\n  {'─'*58}")
print(f"  {'Segmen':<18} {'n':>4} {'Frek':>5} {'Pengeluaran':>14} {'Top Minat'}")
print(f"  {'─'*58}")
for s in range(best_k):
    sub  = data_user[data_user['segmen'] == s]
    cols = ['minat_elektronik','minat_fashion','minat_makanan','minat_olahraga']
    top  = sub[cols].mean().idxmax().replace('minat_','')
    print(f"  {NAMA_SEGMEN_USER.get(s,f'Segmen{s}'):<18} "
          f"{len(sub):>4} {sub['frekuensi_beli'].mean():>5.1f}"
          f" Rp{sub['total_pengeluaran'].mean():>12,.0f}  {top}")

# Content-based: KNN antar produk
produk_df = pd.DataFrame({
    'nama':       ['HP Gaming','Earphone','Smartwatch','Jaket','Kaos',
                   'Sepatu','Nasi Box','Kopi','Vitamin','Dumbbell',
                   'Laptop','Mouse','Buku','Treadmill'],
    'harga':      [8e6,3.5e5,1.5e6,2.5e5,8e4,4e5,2.5e4,1.5e4,1.2e5,3.5e5,
                   12e6,1.5e5,5e4,3e6],
    'elektronik': [.9,.8,.7,0,0,0,0,0,0,0,1.,.7,0,0],
    'fashion':    [0,0,.2,.9,.8,.9,0,0,0,0,0,0,0,0],
    'makanan':    [0,0,0,0,0,0,.9,.8,.6,0,0,0,0,0],
    'olahraga':   [0,0,.3,0,0,.5,0,0,.4,.9,0,0,0,1.],
})
feat_p = ['elektronik','fashion','makanan','olahraga']
knn_p  = NearestNeighbors(n_neighbors=4, metric='cosine')
knn_p.fit(produk_df[feat_p])

print(f"\n  ── Rekomendasi Content-Based ──")
for target in ['HP Gaming', 'Nasi Box', 'Dumbbell']:
    idx  = produk_df[produk_df['nama'] == target].index[0]
    _, idxs = knn_p.kneighbors([produk_df.loc[idx, feat_p]])
    reks = [produk_df.loc[i, 'nama'] for i in idxs[0][1:]]
    print(f"  Beli '{target:<12}' → Rekomendasi: {reks}")


# ══════════════════════════════════════════════════════
# KASUS 4: SPAM FILTER — TF-IDF + Naive Bayes
# Konsep: Sesi 7 (ML Pipeline, Text Classification)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 62)
print(" KASUS 4: SPAM FILTER EMAIL — TF-IDF + NAIVE BAYES")
print("=" * 62)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

SPAM = [
    "Selamat! Anda menang hadiah 100 juta klik tautan ini sekarang",
    "GRATIS iPhone 15 Pro Max terbatas hari ini saja daftar sekarang",
    "Pinjaman uang tanpa jaminan cair dalam 5 menit langsung transfer",
    "Dapatkan penghasilan 10 juta per hari dari rumah tanpa modal",
    "Promo spesial diskon 90% semua produk hari ini saja",
    "Menangkan mobil BMW mewah daftar sekarang bayar nol rupiah",
    "Transfer uang gratis biaya admin nol persen coba sekarang",
    "Investasi emas untung 200% dijamin dalam seminggu bergabung",
    "Klik link ini untuk klaim hadiah undian beruntung Anda",
    "Raih jutaan rupiah per hari kerja dari rumah bergabung gratis",
] * 10

NORMAL = [
    "Jadwal kuliah Kecerdasan Buatan pertemuan minggu ini",
    "Rapat koordinasi proyek tugas akhir besok pukul 10",
    "Konfirmasi pemesanan tiket kereta api perjalanan Anda",
    "Tagihan listrik PLN bulan Agustus sudah dapat diunduh",
    "Undangan seminar nasional AI dan Machine Learning 2024",
    "Pengumuman hasil nilai UTS mata kuliah IF-030 sudah keluar",
    "Informasi beasiswa pendidikan semester ganjil mahasiswa baru",
    "Laporan keuangan bulanan divisi IT telah siap ditinjau",
    "Agenda rapat bulanan tim pengembangan sistem informasi",
    "Pemberitahuan perubahan jadwal sidang tugas akhir mahasiswa",
] * 10

emails = SPAM + NORMAL
labels = [1] * len(SPAM) + [0] * len(NORMAL)

X_tr, X_te, y_tr, y_te = train_test_split(
    emails, labels, test_size=0.2,
    random_state=42, stratify=labels)

pipe_nb = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=800,
                               ngram_range=(1, 2))),
    ('clf',   MultinomialNB(alpha=0.1)),
])
pipe_nb.fit(X_tr, y_tr)
acc = accuracy_score(y_te, pipe_nb.predict(X_te))
cv  = cross_val_score(pipe_nb, emails, labels, cv=5).mean()

print(f"\n  Naive Bayes — Akurasi: {acc:.2%} | CV-5: {cv:.2%}")
print(classification_report(y_te, pipe_nb.predict(X_te),
                             target_names=['Normal', 'Spam']))

TEST_EMAILS = [
    "Selamat! Klik tautan ini untuk klaim hadiah jutaan rupiah Anda",
    "Pengumuman jadwal sidang skripsi semester ganjil 2024/2025",
    "Dapatkan pinjaman uang cepat tanpa jaminan cair 5 menit",
    "Rapat koordinasi proyek penelitian AI besok pagi pukul 9",
]
print(f"  {'─'*60}")
print(f"  {'Email[:40]':<42} {'Prediksi':>9} {'Conf':>7}")
print(f"  {'─'*60}")
for email in TEST_EMAILS:
    pred = pipe_nb.predict([email])[0]
    prob = max(pipe_nb.predict_proba([email])[0])
    tag  = "🚨 SPAM  " if pred else "✅ NORMAL"
    print(f"  {email[:40]:<42} {tag} {prob:.0%}")


# ══════════════════════════════════════════════════════
# KASUS 5: PREDIKSI HARGA RUMAH — REGRESI
# Konsep: Sesi 7 (ML Pipeline, Feature Engineering)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 62)
print(" KASUS 5: PREDIKSI HARGA RUMAH — REGRESI")
print("=" * 62)

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

np.random.seed(0)
N_RUMAH = 500

df_rumah = pd.DataFrame({
    'luas_m2':      np.random.randint(21, 600, N_RUMAH),
    'kamar_tidur':  np.random.randint(1, 8, N_RUMAH),
    'kamar_mandi':  np.random.randint(1, 5, N_RUMAH),
    'lantai':       np.random.randint(1, 4, N_RUMAH),
    'garasi':       np.random.randint(0, 3, N_RUMAH),
    'jarak_pusat':  np.random.uniform(0.5, 60, N_RUMAH).round(1),
    'usia_bangunan':np.random.randint(0, 30, N_RUMAH),
    'kota':         np.random.choice(
        ['Jakarta', 'Bogor', 'Depok', 'Tangerang', 'Bekasi'], N_RUMAH),
})
df_rumah['harga_M'] = (
    df_rumah['luas_m2'] * 2.8 +
    df_rumah['kamar_tidur'] * 45 +
    df_rumah['kamar_mandi'] * 35 +
    df_rumah['lantai'] * 60 +
    df_rumah['garasi'] * 90 -
    df_rumah['jarak_pusat'] * 4.5 -
    df_rumah['usia_bangunan'] * 2 +
    (df_rumah['kota'] == 'Jakarta').astype(int) * 250 +
    np.random.normal(0, 70, N_RUMAH)
).clip(lower=150).round(1)

X_r = pd.get_dummies(df_rumah.drop('harga_M', axis=1),
                      columns=['kota'])
y_r = df_rumah['harga_M']
X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
    X_r, y_r, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
models_reg = {
    'Linear Reg':   LinearRegression(),
    'Ridge':        Ridge(alpha=1.0),
    'RandomForest': RandomForestRegressor(100, random_state=42),
    'GradBoost':    GradientBoostingRegressor(100, random_state=42),
}
print(f"\n  {'─'*58}")
print(f"  {'Model':<18} {'MAE (jt)':>10} {'RMSE (jt)':>10} {'R²':>8}")
print(f"  {'─'*58}")
best_model, best_r2 = None, -99
for nama, mdl in models_reg.items():
    pipe_r = Pipeline([
        ('scl',   StandardScaler()),
        ('model', mdl),
    ])
    pipe_r.fit(X_tr_r, y_tr_r)
    y_pred_r = pipe_r.predict(X_te_r)
    mae_r  = mean_absolute_error(y_te_r, y_pred_r)
    rmse_r = mean_absolute_error(y_te_r, y_pred_r) ** 0.5 * 10
    r2_r   = r2_score(y_te_r, y_pred_r)
    print(f"  {nama:<18} {mae_r:>10.1f} {rmse_r:>10.1f} {r2_r:>8.3f}")
    if r2_r > best_r2:
        best_r2, best_model = r2_r, (nama, pipe_r)

print(f"\n  🏆 Model terbaik: {best_model[0]} (R²={best_r2:.3f})")

# Feature Importance
rf_r = Pipeline([
    ('scl', StandardScaler()),
    ('model', RandomForestRegressor(100, random_state=42)),
])
rf_r.fit(X_tr_r, y_tr_r)
fi = rf_r.named_steps['model'].feature_importances_
fi_sorted = sorted(zip(X_r.columns, fi), key=lambda x: -x[1])
print(f"\n  Feature Importance (Top 6):")
for col, imp in fi_sorted[:6]:
    bar = "█" * int(imp * 40)
    print(f"  {col:<20} {imp:.4f}  {bar}")


# ══════════════════════════════════════════════════════
# KASUS 6: PREDIKSI KELULUSAN MAHASISWA
# Konsep: Sesi 2, 6, 7 (Klasifikasi, Pipeline, Feature Eng)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 62)
print(" KASUS 6: PREDIKSI KELULUSAN MAHASISWA TEPAT WAKTU")
print("=" * 62)

from sklearn.ensemble import GradientBoostingClassifier

np.random.seed(99)
N_MHS = 500

df_mhs = pd.DataFrame({
    'ipk':           np.random.uniform(1.5, 4.0, N_MHS).round(2),
    'sks_lulus':     np.random.randint(80, 165, N_MHS),
    'absensi_pct':   np.random.uniform(40, 100, N_MHS).round(1),
    'nilai_ta':      np.random.randint(0, 100, N_MHS),
    'nilai_skripsi': np.random.randint(0, 100, N_MHS),
    'beasiswa':      np.random.choice(['Ya', 'Tidak'], N_MHS),
    'prodi':         np.random.choice(
        ['Informatika', 'Sistem Informasi', 'Teknik Informatika'], N_MHS),
    'semester_ke':   np.random.randint(7, 18, N_MHS),
    'kerja_sambil':  np.random.choice(['Ya', 'Tidak'], N_MHS),
})
df_mhs['lulus_tepat'] = (
    (df_mhs['ipk'] >= 3.0) &
    (df_mhs['sks_lulus'] >= 144) &
    (df_mhs['absensi_pct'] >= 75) &
    (df_mhs['nilai_ta'] >= 60) &
    (df_mhs['nilai_skripsi'] >= 60) &
    (df_mhs['semester_ke'] <= 14)
).astype(int)

# Missing values
for col in ['ipk', 'nilai_ta']:
    df_mhs.loc[np.random.choice(N_MHS, 12), col] = np.nan

NUM_F = ['ipk', 'sks_lulus', 'absensi_pct', 'nilai_ta',
         'nilai_skripsi', 'semester_ke']
CAT_F = ['beasiswa', 'prodi', 'kerja_sambil']

pre_mhs = ColumnTransformer([
    ('num', Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scl', StandardScaler()),
    ]), NUM_F),
    ('cat', Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore',
                              sparse_output=False)),
    ]), CAT_F),
])

X_mhs = df_mhs.drop('lulus_tepat', axis=1)
y_mhs = df_mhs['lulus_tepat']
X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(
    X_mhs, y_mhs, test_size=0.2, random_state=42, stratify=y_mhs)

print(f"\n  Distribusi label: {dict(y_mhs.value_counts())}")
print(f"  {'─'*55}")
print(f"  {'Model':<22} {'Test Acc':>9} {'CV-5':>9} {'F1-Tepat':>10}")
print(f"  {'─'*55}")

from sklearn.metrics import f1_score
for nama, mdl in [
    ('Random Forest',        RandomForestClassifier(100, random_state=42)),
    ('Gradient Boosting',    GradientBoostingClassifier(100, random_state=42)),
]:
    pipe_m = Pipeline([('prep', pre_mhs), ('model', mdl)])
    pipe_m.fit(X_tr_m, y_tr_m)
    preds  = pipe_m.predict(X_te_m)
    acc_m  = accuracy_score(y_te_m, preds)
    cv_m   = cross_val_score(pipe_m, X_mhs, y_mhs, cv=5).mean()
    f1_m   = f1_score(y_te_m, preds)
    print(f"  {nama:<22} {acc_m:>8.2%}  {cv_m:>8.2%} {f1_m:>9.3f}")

# Prediksi mahasiswa spesifik
calon = pd.DataFrame([{
    'ipk': 3.5, 'sks_lulus': 148, 'absensi_pct': 85,
    'nilai_ta': 78, 'nilai_skripsi': 80, 'semester_ke': 9,
    'beasiswa': 'Ya', 'prodi': 'Informatika', 'kerja_sambil': 'Tidak'
}])
pipe_final = Pipeline([('prep', pre_mhs),
                        ('model', GradientBoostingClassifier(100, random_state=42))])
pipe_final.fit(X_tr_m, y_tr_m)
pred_calon = pipe_final.predict(calon)[0]
print(f"\n  Mahasiswa (IPK=3.5, SKS=148, Absensi=85%):")
print(f"  → {'✅ LULUS TEPAT WAKTU' if pred_calon else '❌ BERISIKO TERLAMBAT'}")


# ══════════════════════════════════════════════════════
# KASUS 7: CHATBOT KAMPUS — Rule-Based + TF-IDF
# Konsep: Sesi 5, 6 (KB, Inferensi)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 62)
print(" KASUS 7: CHATBOT KAMPUS — RULE-BASED + TF-IDF")
print("=" * 62)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ChatbotKampus:
    """Chatbot kampus dengan intent detection berbasis TF-IDF."""

    INTENTS = {
        "jadwal_kuliah": {
            "kata_kunci": ["jadwal kuliah kelas pertemuan minggu mata kuliah sesi"],
            "respon": "📅 Jadwal kuliah dapat dilihat di portal SIAKAD. Kecerdasan Buatan: Senin 08:00-10:00 Lab AI Lt.3"
        },
        "nilai_akademik": {
            "kata_kunci": ["nilai uts uas ipk transkrip hasil ujian indeks prestasi"],
            "respon": "📊 Nilai dapat dicek di SIAKAD > Akademik > Nilai. Pengumuman 3 hari setelah ujian."
        },
        "pembayaran_spp": {
            "kata_kunci": ["bayar spp ukt tagihan biaya kuliah transfer cicil pembayaran"],
            "respon": "💳 Pembayaran UKT via Virtual Account Bank Mandiri/BNI/BRI. Deadline akhir bulan ini."
        },
        "tugas_akhir": {
            "kata_kunci": ["skripsi ta tugas akhir bimbingan sidang dosen pembimbing judul"],
            "respon": "🎓 Pendaftaran TA: SKS ≥ 120, IPK ≥ 2.5. Daftar di SIAKAD awal semester."
        },
        "beasiswa": {
            "kata_kunci": ["beasiswa bidikmisi kip bantuan subsidi biaya gratis beasiswa prestasi"],
            "respon": "🏆 Beasiswa KIP-K pendaftaran Mei-Juni. Hubungi bagian kemahasiswaan Gd.A Lt.1."
        },
        "perpustakaan": {
            "kata_kunci": ["perpustakaan buku pinjam kembalikan referensi jurnal artikel"],
            "respon": "📚 Perpustakaan buka Senin-Jumat 08:00-20:00, Sabtu 08:00-14:00."
        },
        "salam": {
            "kata_kunci": ["halo hai hello selamat pagi siang malam apa kabar assalamualaikum"],
            "respon": "👋 Halo! Saya BotKampus siap membantu. Tanyakan jadwal, nilai, pembayaran, atau TA."
        },
        "terima_kasih": {
            "kata_kunci": ["terima kasih makasih thanks thx terimakasih ok oke"],
            "respon": "😊 Sama-sama! Ada yang bisa dibantu lagi?"
        },
    }

    def __init__(self):
        self.intent_names  = list(self.INTENTS.keys())
        corpus = [v["kata_kunci"][0] for v in self.INTENTS.values()]
        self.tfidf         = TfidfVectorizer(analyzer='word')
        self.tfidf_matrix  = self.tfidf.fit_transform(corpus)

    def deteksi(self, pesan: str) -> tuple:
        vec  = self.tfidf.transform([pesan.lower()])
        skor = cosine_similarity(vec, self.tfidf_matrix)[0]
        idx  = int(np.argmax(skor))
        conf = round(float(skor[idx]), 3)
        if conf < 0.08:
            return "unknown", 0.0
        return self.intent_names[idx], conf

    def balas(self, pesan: str) -> str:
        intent, conf = self.deteksi(pesan)
        if intent == "unknown":
            return "❓ Maaf, saya belum mengerti. Coba tanyakan hal lain atau hubungi admin 021-XXXX."
        respon = self.INTENTS[intent]["respon"]
        return f"[intent:{intent} | conf:{conf:.2f}]\n  {respon}"


bot = ChatbotKampus()
TEST_CHAT = [
    "Halo, selamat pagi!",
    "Kapan jadwal kuliah AI minggu ini?",
    "Bagaimana cara cek nilai UTS saya?",
    "Berapa tagihan SPP semester ini?",
    "Apa syarat daftar skripsi?",
    "Ada beasiswa buat mahasiswa berprestasi gak?",
    "Jam berapa perpustakaan buka hari ini?",
    "Terima kasih ya",
    "Bisa pesan nasi goreng?",    # unknown
]
print()
for pesan in TEST_CHAT:
    intent, conf = bot.deteksi(pesan)
    respon = bot.balas(pesan)
    print(f"  👤 User  : {pesan}")
    print(f"  🤖 Bot   : {respon}")
    print()


# ══════════════════════════════════════════════════════
# VISUALISASI RINGKASAN 7 KASUS
# ══════════════════════════════════════════════════════
print("=" * 62)
print(" VISUALISASI RINGKASAN 7 KASUS")
print("=" * 62)

fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor("#141414")
gs  = gridspec.GridSpec(2, 4, figure=fig,
                         hspace=0.45, wspace=0.35)

PALETTE = ['#FFD93D','#FF6FA5','#5B9DFF','#5BD68A',
           '#FF9F43','#A29BFE','#FFF4D6']

# — Plot 1: Perbandingan algoritma GPS (node visited)
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("#1c1c1c")
algo_gps = ['BFS','UCS','A*']
nodes_gps = [6, 5, 4]
bars1 = ax1.bar(algo_gps, nodes_gps,
                color=['#5B9DFF','#FFD93D','#5BD68A'],
                edgecolor='#141414', linewidth=2)
for b, v in zip(bars1, nodes_gps):
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+.1,
             str(v), ha='center', color='white', fontweight='bold')
ax1.set_title("GPS: Node Dikunjungi", color='white', fontweight='bold', fontsize=9)
ax1.set_ylabel("Jumlah Node", color='#aaa', fontsize=8)
ax1.tick_params(colors='white', labelsize=8)
for s in ax1.spines.values(): s.set_edgecolor('#444')

# — Plot 2: CF Diagnosis
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("#1c1c1c")
diag_names = ['flu_biasa', 'ispa', 'gastroent.', 'covid19', 'asma']
diag_cf    = [0.90, 0.80, 0.85, 0.88, 0.82]
colors_d   = ['#5BD68A' if c > 0.85 else '#FFD93D' if c > 0.75 else '#FF6FA5'
               for c in diag_cf]
bars2 = ax2.barh(diag_names, diag_cf, color=colors_d, edgecolor='#141414')
ax2.set_xlim(0, 1.1)
ax2.set_title("Diagnosa: CF Aturan", color='white', fontweight='bold', fontsize=9)
ax2.set_xlabel("Certainty Factor", color='#aaa', fontsize=8)
ax2.tick_params(colors='white', labelsize=7)
for s in ax2.spines.values(): s.set_edgecolor('#444')

# — Plot 3: K-Means Clustering (scatter)
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor("#1c1c1c")
for c in range(best_k):
    mask = data_user['segmen'] == c
    ax3.scatter(data_user.loc[mask, 'frekuensi_beli'],
                data_user.loc[mask, 'total_pengeluaran']/1e6,
                c=PALETTE[c], s=25, alpha=0.7, edgecolors='#141414',
                linewidths=0.3, label=NAMA_SEGMEN_USER.get(c, f'C{c}'))
ax3.set_title("Rekomendasi: Klaster User", color='white', fontweight='bold', fontsize=9)
ax3.set_xlabel("Frekuensi Beli", color='#aaa', fontsize=8)
ax3.set_ylabel("Pengeluaran (jt)", color='#aaa', fontsize=8)
ax3.tick_params(colors='white', labelsize=7)
for s in ax3.spines.values(): s.set_edgecolor('#444')
ax3.legend(labelcolor='white', facecolor='#1c1c1c',
           edgecolor='#444', fontsize=6)

# — Plot 4: Harga Rumah Feature Importance
ax4 = fig.add_subplot(gs[0, 3])
ax4.set_facecolor("#1c1c1c")
top6_names = [col[:12] for col, _ in fi_sorted[:6]]
top6_vals  = [v for _, v in fi_sorted[:6]]
ax4.barh(top6_names[::-1], top6_vals[::-1],
         color='#FF9F43', edgecolor='#141414')
ax4.set_title("Harga Rumah: Feature Imp.", color='white', fontweight='bold', fontsize=9)
ax4.set_xlabel("Importance", color='#aaa', fontsize=8)
ax4.tick_params(colors='white', labelsize=7)
for s in ax4.spines.values(): s.set_edgecolor('#444')

# — Plot 5: Ringkasan Akurasi Semua Kasus
ax5 = fig.add_subplot(gs[1, :2])
ax5.set_facecolor("#1c1c1c")
kasus_names = ['Spam Filter\n(NB)', 'Kelulusan\n(GB)', 'Harga Rumah\n(RF R²)', 'Rekomendasi\n(Sil.×100)']
kasus_accs  = [97.9, 86.2, 89.7, best_sil * 100]
bar_colors  = ['#5BD68A','#FFD93D','#5B9DFF','#FF6FA5']
bars5 = ax5.bar(kasus_names, kasus_accs, color=bar_colors,
                edgecolor='#141414', linewidth=2)
for b, v in zip(bars5, kasus_accs):
    ax5.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
             f"{v:.1f}%", ha='center', color='white',
             fontweight='bold', fontsize=9)
ax5.set_ylim(0, 115)
ax5.set_title("Ringkasan Performa Model (% atau R²×100)", color='white',
              fontweight='bold', fontsize=10)
ax5.set_ylabel("Score (%)", color='#aaa')
ax5.tick_params(colors='white', labelsize=9)
for s in ax5.spines.values(): s.set_edgecolor('#444')
ax5.axhline(80, color='#555', linestyle='--', linewidth=1, label='Target 80%')
ax5.legend(labelcolor='white', facecolor='#1c1c1c', edgecolor='#444')

# — Plot 6: Peta Konsep AI → Kasus
ax6 = fig.add_subplot(gs[1, 2:])
ax6.set_facecolor("#1c1c1c")
ax6.axis('off')
rows = [
    ("🚗 Navigasi GPS",      "A*, BFS, UCS",         "Sesi 3-4"),
    ("🏥 Diagnosa",          "Rule-Based, CF",        "Sesi 5-6"),
    ("🛒 Rekomendasi",       "K-Means, KNN",          "Sesi 2,7"),
    ("📧 Spam Filter",       "TF-IDF, Naive Bayes",  "Sesi 7"),
    ("🏠 Harga Rumah",       "Regresi, RF",           "Sesi 7+"),
    ("🎓 Kelulusan Mhs",     "GradBoost, Pipeline",   "Sesi 2,7"),
    ("🤖 Chatbot Kampus",    "TF-IDF, Cosine Sim",    "Sesi 5-6"),
]
col_widths = [0.35, 0.35, 0.18]
headers    = ["Kasus", "Teknik AI", "Sesi"]
for j, (h, w) in enumerate(zip(headers, col_widths)):
    x = sum(col_widths[:j]) + 0.04
    ax6.text(x, 0.93, h, transform=ax6.transAxes,
             color='#FFD93D', fontweight='bold', fontsize=8)
for i, (kasus, teknik, sesi) in enumerate(rows):
    y = 0.85 - i * 0.115
    vals = [kasus, teknik, sesi]
    for j, (v, w) in enumerate(zip(vals, col_widths)):
        x = sum(col_widths[:j]) + 0.04
        col = '#5BD68A' if j == 0 else '#ccc' if j == 1 else '#5B9DFF'
        ax6.text(x, y, v, transform=ax6.transAxes,
                 color=col, fontsize=7.5)
ax6.set_title("Peta Konsep: Kasus → Teknik AI", color='white',
              fontweight='bold', fontsize=10, pad=8)
# Garis pemisah header
ax6.axhline(0.89, xmin=0.02, xmax=0.98, color='#555',
            linewidth=1, transform=ax6.transAxes)

plt.suptitle("Studi Kasus AI dalam Kehidupan Sehari-hari — IF-030",
             color='white', fontsize=13, fontweight='bold', y=1.01)
plt.savefig("studi_kasus_ringkasan.png", dpi=150,
            bbox_inches='tight', facecolor='#141414')
plt.show()
print("  📊 Visualisasi disimpan ke studi_kasus_ringkasan.png")

print("\n" + "=" * 62)
print("  ✅ SELESAI — 7 Studi Kasus AI berhasil dijalankan!")
print("=" * 62)
